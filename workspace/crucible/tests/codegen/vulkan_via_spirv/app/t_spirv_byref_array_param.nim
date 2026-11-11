## Direct SPIR-V: `{.device.}` function parameters carrying arrays pass as pointers
## Test that a kernel passing a local array to an array-parameter device
## function reads back exact values
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_byref_array_param.nim

import std/[os, osproc]
import workspace/crucible

const byrefArraySpv = spirv:
  proc doubledAt(a: array[64, float32], first: int): float32 {.device.} =
    a[first] * 2.0'f32

  proc arrayKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var buf: array[64, float32]
    for i in 0 ..< 64:
      buf[i] = float32(i + 1)
    output[0] = doubledAt(buf, 9)

proc runTest() =
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_byref_array_param.spv"
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr byrefArraySpv[0], byrefArraySpv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

  var engine = newEngine(runVulkan)
  engine.ingest(byrefArraySpv)
  let oDev = engine.allocDevice(float32, 1)

  engine.launchKernel("arrayKernel", oDev, ())
  var got: array[1, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 20.0'f32, "got " & $got[0] & ", want 2.0 * 10.0 = 20.0"
  echo "[OK] direct SPIR-V: bare array parameter crosses as a pointer"

when isMainModule:
  runTest()
  echo "t_spirv_byref_array_param: all checks passed"
