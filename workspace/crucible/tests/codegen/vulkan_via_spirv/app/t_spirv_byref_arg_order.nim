## Direct SPIR-V: by-ref argument temporaries versus evaluation order
## Test that documents the argument-temporary limitation of the by-ref lowering
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_byref_arg_order.nim

import std/[os, osproc]
import workspace/crucible

type
  Count = object
    n: int32
    pad: array[16, int32]

const byrefArgOrderSpv = spirv:
  proc bump(p: ptr UncheckedArray[int32], cell: int32): int32 {.device.} =
    p[cell] = p[9]
    p[9] = p[9] + 1'i32
    p[cell]

  proc sumAt(w: int32, c: Count): int32 {.device.} =
    w + c.n

  proc orderKernel(output: ptr UncheckedArray[int32]) {.global.} =
    output[8] = sumAt(bump(output, 6'i32), Count(n: bump(output, 7'i32)))

const byrefTernSpv = spirv:
  proc bump(p: ptr UncheckedArray[int32], cell: int32): int32 {.device.} =
    p[cell] = p[9]
    p[9] = p[9] + 1'i32
    p[cell]

  proc pickSlot(c: Count): int32 {.device.} =
    c.n * 2'i32

  proc ternKernel(output: ptr UncheckedArray[int32]) {.global.} =
    output[4] = if output[0] > 100'i32: pickSlot(Count(n: bump(output, 5'i32))) else: 5'i32

# Documented by-ref lowering limitation.
# Nim evaluates arguments left to right and only the selected ternary branch.
# By-ref lowering hoists a non-lvalue struct argument into a temporary instead.
# Temporary effects precede an earlier argument's effects.
# Temporary effects also run inside an unselected ternary branch.
# Fix: expression-level statement extraction in a dedicated pass.
proc runTest() =
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)

  var zeros10: array[10, int32]

  block order:
    let spvPath = outDir / "t_spirv_byref_arg_order.spv"
    var f = open(spvPath, fmWrite)
    discard f.writeBuffer(addr byrefArgOrderSpv[0], byrefArgOrderSpv.len * 4)
    f.close()
    let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
    doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

    var engine = newEngine(runVulkan)
    engine.ingest(byrefArgOrderSpv)
    let oOrder = engine.allocDevice(int32, 10)
    engine.copyToDeviceFromHost(oOrder, zeros10)
    engine.launchKernel("orderKernel", oOrder, ())
    var got: array[10, int32]
    engine.copyToHostFromDevice(got.toOpenArray(0, 9), oOrder)
    # documented limitation: argument 2's temporary evaluates before argument 1,
    # cell7 takes counter value 0 and cell6 takes 1
    doAssert got[6] == 1'i32 and got[7] == 0'i32,
      "the hoisted temporary evaluates before the earlier argument, got " &
      "cell6=" & $got[6] & " cell7=" & $got[7]

  block tern:
    let spvPath = outDir / "t_spirv_byref_arg_tern.spv"
    var f = open(spvPath, fmWrite)
    discard f.writeBuffer(addr byrefTernSpv[0], byrefTernSpv.len * 4)
    f.close()
    let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
    doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

    var engine = newEngine(runVulkan)
    engine.ingest(byrefTernSpv)
    let oTern = engine.allocDevice(int32, 10)
    engine.copyToDeviceFromHost(oTern, zeros10)
    engine.launchKernel("ternKernel", oTern, ())
    var got2: array[10, int32]
    engine.copyToHostFromDevice(got2.toOpenArray(0, 9), oTern)
    doAssert got2[4] == 5'i32, "the unselected branch yields its else value, got " & $got2[4]
    # documented limitation: branch selection happens after the hoisted
    # temporary runs and the bump inside the unselected branch advances the counter
    doAssert got2[9] == 1'i32,
      "the unselected branch runs its by-ref argument's bump, got counter " & $got2[9]
  echo "[OK] direct SPIR-V: documented by-ref temporary evaluation-order limitation"

when isMainModule:
  runTest()
  echo "t_spirv_byref_arg_order: all checks passed"
