## SPIR-V binary emission: pointer and by-value params in device fns
##
## Test that a kernel calling device fns with `var` and by-value parameters
## emits valid binary SPIR-V, and that the kernel computes the expected
## value on Vulkan through its callees' pointer writes.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_param_element_chains.nim

import std/os
import std/osproc
import workspace/crucible

type
  Reg4 = object
    data: array[4, float32]

proc fill(dst: var Reg4, v: float32) {.device.} =
  var i = 0
  while i < 4:
    dst.data[i] = v + float32(i)
    i = i + 1

proc scale(v: var float32, k: float32) {.device.} =
  v = v * k

proc gather(acc: var float32, tile: Reg4) {.device.} =
  var i = 0
  while i < 4:
    acc = acc + tile.data[i]
    i = i + 1

const paramSpv = spirv:
  proc paramKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var r: Reg4
    fill(r, 2.0'f32)
    var x: float32 = 3.0'f32
    scale(x, 2.0'f32)
    var acc: float32 = 0.0'f32
    gather(acc, r)
    output[0] = x + acc + r.data[0]

proc paramKernelLaunch(output: ptr UncheckedArray[float32]) =
  var r: Reg4
  fill(r, 2.0'f32)
  var x: float32 = 3.0'f32
  scale(x, 2.0'f32)
  var acc: float32 = 0.0'f32
  gather(acc, r)
  output[0] = x + acc + r.data[0]

proc runTest() =
  block emission:
    doAssert paramSpv.len > 0, "`spirv:` macro emitted no words"
    doAssert paramSpv[0] == 0x07230203'u32, "SPIR-V magic number missing"

  block validation:
    let spirvVal = findExe("spirv-val")
    doAssert spirvVal.len > 0,
      "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
    let spirvDis = findExe("spirv-dis")

    let outDir = "build/tests"
    createDir(outDir)
    let spvPath = outDir / "t_spirv_param_element_chains.spv"
    # File closes at block exit, before the subprocesses read it
    block:
      var f = open(spvPath, fmWrite)
      defer: f.close()
      discard f.writeBuffer(addr paramSpv[0], paramSpv.len * 4)

    let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
    if valCode != 0:
      if spirvDis.len > 0:
        let (disOut, _) = execCmdEx(spirvDis & " " & spvPath)
        echo "spirv-dis dump:\n", disOut
      doAssert false, "spirv-val rejected the emitted module:\n" & valOut
    # spirv-dis only feeds the failure dump above. The check is spirv-val itself.
    doAssert spirvDis.len > 0,
      "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"

  block numerics:
    # scale writes x through a pointer parameter, gather writes acc through one
    # and reads r by value: 6.0 + 14.0 + 2.0
    var engine = newEngine(runVulkan)
    let oDev = engine.allocDevice(float32, 1)
    engine.launchDeviceFn paramKernelLaunch(oDev) do:
      lang = spirv
      grid = (1, 1)
      blk = (1, 1)
    var got: array[1, float32]
    engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
    doAssert got[0] == 22.0'f32,
      "kernel computed " & $got[0] & ", want 22.0 (6.0 + 14.0 + 2.0)"

when isMainModule:
  runTest()
  echo "t_spirv_param_element_chains: spirv-val accepted the emitted module, kernel computed 22.0"
