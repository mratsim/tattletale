## SPIR-V barrier emission: workgroup-scoped OpControlBarrier, end to end.
##
## Test that a statement `threadgroup_barrier()` in a `spirv:` block:
##
## - emits workgroup-scoped OpControlBarrier
## - passes spirv-val
## - completes a 2-workgroup MoltenVK launch over the barrier
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_barrier.nim

import std/[os, osproc, strutils, tables]
import workspace/crucible

const barrierSpv = spirv:
  proc barrierKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 1, 1).} =
    let tid = uint32(thread_position_in_grid.x)
    output[tid] = tid
    threadgroup_barrier()
    output[tid] = tid + 100'u32

proc validateSpirv(spv: seq[uint32], name: string): string =
  # spirv-val the module, then hand back the spirv-dis text for operand checks
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / (name & ".spv")
  # File handle closes at block exit, before the subprocesses read it
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr spv[0], spv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the emitted module:\n" & valOut
  let spirvDis = findExe("spirv-dis")
  doAssert spirvDis.len > 0,
    "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let (disOut, disCode) = execCmdEx(spirvDis & " " & spvPath)
  doAssert disCode == 0, "spirv-dis failed on the emitted module:\n" & disOut
  disOut

proc controlBarrierOperands(dis: string): array[3, int] =
  ## Resolves the three OpControlBarrier operands to their constant values
  ## through the disassembly's own OpConstant definitions.
  var consts: Table[string, int]
  for line in dis.splitLines:
    let t = line.strip
    # `%uint_2 = OpConstant %uint 2` names a 32-bit constant
    if "= OpConstant %uint " in t:
      let parts = t.splitWhitespace
      consts[parts[0]] = parseInt(parts[4])
  var n = 0
  for line in dis.splitLines:
    let t = line.strip
    if t.startsWith("OpControlBarrier "):
      let parts = t.splitWhitespace
      doAssert parts.len == 4, "OpControlBarrier operand count drifted: " & t
      for i in 1 ..< 4:
        result[n] = consts[parts[i]]
        n.inc
  doAssert n == 3, "no OpControlBarrier in the disassembly"

proc runTest() =
  let dis = validateSpirv(barrierSpv, "t_spirv_barrier")
  # Operand order: execution scope, memory scope, memory semantics.
  # 2 = Workgroup: a Device execution scope deadlocks any multi-workgroup launch.
  # 264 = 0x108 = WorkgroupMemory | AcquireRelease, matching the atomic emission's UniformMemory | AcquireRelease form.
  let operands = controlBarrierOperands(dis)
  doAssert operands == [2, 2, 264],
    "OpControlBarrier operands drifted: " & $operands
  echo "  emission: OpControlBarrier operands " & $operands & " (scope, scope, semantics)"

  block twoWorkgroupLaunch:
    var engine = newEngine(runVulkan)
    engine.ingest(barrierSpv)
    let oDev = engine.allocDevice(uint32, 8)
    # 2 workgroups of 4: a Device-scoped barrier deadlocks this launch,
    # a Workgroup-scoped one lets every invocation pass and store
    engine.launchKernel<<(grid: (2, 1, 1), blk: (4, 1, 1))>>("barrierKernel", oDev, ())
    var got: array[8, uint32]
    engine.copyToHostFromDevice(got.toOpenArray(0, 8 - 1), oDev)
    for i in 0 ..< 8:
      doAssert got[i] == uint32(i) + 100'u32,
        "element " & $i & " got " & $got[i] & ", want " & $(uint32(i) + 100) &
        " (an invocation never passed the barrier)"
    echo "  launch: 2 workgroups x 4 invocations over the barrier, readback exact"

when isMainModule:
  runTest()
  echo "t_spirv_barrier: all checks passed"
