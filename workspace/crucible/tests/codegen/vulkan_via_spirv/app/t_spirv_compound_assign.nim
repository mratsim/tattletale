## SPIR-V direct: compound assignments lower to load/binop/store
## Test that compound assignments compile through the SPIR-V pipeline and run one binop per site on MoltenVK
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_compound_assign.nim

import std/[os, osproc, strutils]
import workspace/crucible

const scalarSpv = spirv:
  proc compoundScalarKernel(o: ptr UncheckedArray[float32]) {.global.} =
    let i = thread_position_in_grid.x
    var acc = o[i]
    acc += 2.5'f32
    o[i] = acc
    var d = o[i]
    d -= 1.0'f32
    o[i] = d

const indexedSpv = spirv:
  proc compoundIndexedKernel(o: ptr UncheckedArray[float32]) {.global.} =
    let i = thread_position_in_grid.x
    o[i] += 4.0'f32
    o[i] *= 3.0'f32
    o[i] /= 2.0'f32

proc disassembly(spv: seq[uint32], name: string): string =
  # spirv-val the module before it reaches the engine: a mis-encoded word
  # sequence must fail here, not as a driver-side opaque launch error
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / (name & ".spv")
  # the file closes at block exit, before the subprocesses read it
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr spv[0], spv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut
  let spirvDis = findExe("spirv-dis")
  doAssert spirvDis.len > 0,
    "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let (disOut, disCode) = execCmdEx(spirvDis & " " & spvPath)
  doAssert disCode == 0, "spirv-dis failed on " & name
  disOut

proc countOpcode(dis: string, opcode: string): int =
  for line in dis.splitLines():
    if opcode & " " in line:
      result.inc

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block scalarCompound:
    # one binop per compound site: += over the local is OpFAdd,
    # -= over the local is OpFSub, no other arithmetic opcode leaks
    let dis = disassembly(scalarSpv, "t_spirv_compound_assign_scalar")
    doAssert dis.countOpcode("OpFAdd") == 1,
      "the += site is exactly one OpFAdd"
    doAssert dis.countOpcode("OpFSub") == 1,
      "the -= site is exactly one OpFSub"
    doAssert "OpFMul" notin dis and "OpFDiv" notin dis and "OpIAdd" notin dis

    # the compound target is the output buffer, read back after the launch
    var engine = newEngine(runVulkan)
    engine.ingest(scalarSpv)
    let oDev = engine.allocDevice(float32, 4)
    var oHost: array[4, float32] = [1.0'f32, 2.0, 3.0, 4.0]
    engine.copyToDeviceFromHost(oDev, addr oHost[0], oHost.len)

    engine.launchKernel("compoundScalarKernel", oDev, ())

    engine.copyToHostFromDevice(addr oHost[0], oDev, oHost.len)
    for j in 0 ..< 4:
      doAssert oHost[j] == float32(j + 1) + 1.5'f32,
        "slot " & $j & " got " & $oHost[j] & ", want the input + 2.5 - 1.0"
    echo "  scalar compound assign: += 2.5 then -= 1.0 exact on MoltenVK"

  block indexedCompound:
    # the indexed rewrite keeps one load and one store per site:
    # += is one OpFAdd, *= is one OpFMul, /= is one OpFDiv
    let dis = disassembly(indexedSpv, "t_spirv_compound_assign_indexed")
    doAssert dis.countOpcode("OpFAdd") == 1,
      "the += site is exactly one OpFAdd"
    doAssert dis.countOpcode("OpFMul") == 1,
      "the *= site is exactly one OpFMul"
    doAssert dis.countOpcode("OpFDiv") == 1,
      "the /= site is exactly one OpFDiv"

    var engine = newEngine(runVulkan)
    engine.ingest(indexedSpv)
    let oDev = engine.allocDevice(float32, 4)
    var oHost: array[4, float32] = [2.0'f32, 4.0, 6.0, 8.0]
    engine.copyToDeviceFromHost(oDev, addr oHost[0], oHost.len)

    engine.launchKernel("compoundIndexedKernel", oDev, ())

    engine.copyToHostFromDevice(addr oHost[0], oDev, oHost.len)
    for j in 0 ..< 4:
      doAssert oHost[j] == (float32(j + 1) * 2.0'f32 + 4.0'f32) * 3.0'f32 / 2.0'f32,
        "slot " & $j & " got " & $oHost[j] & ", want (input + 4) * 3 / 2"
    echo "  indexed compound assign: += 4, *= 3, /= 2 exact on MoltenVK"

when isMainModule:
  runTest()
  echo "t_spirv_compound_assign: all checks passed"
