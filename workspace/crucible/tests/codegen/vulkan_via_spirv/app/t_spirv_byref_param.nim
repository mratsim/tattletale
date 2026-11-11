## Direct SPIR-V: `{.device.}` function parameters carrying arrays pass as pointers
## Test that a kernel calling a device function taking a struct by value reads back exact values
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_byref_param.nim

import std/[os, osproc]
import workspace/crucible

type
  Staged = object
    values: array[256, float32]

const byrefSpv = spirv:
  proc scaledSum(s: Staged, weight: float32, first: int): float32 {.device.} =
    weight * s.values[first]

  proc storeSumKernel(output: ptr UncheckedArray[float32], weight: float32, first: int) {.global.} =
    var s: Staged
    for i in 0 ..< 256:
      s.values[i] = float32(i + 1)
    output[0] = scaledSum(s, weight, first)

proc runTest() =
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_byref_param.spv"
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr byrefSpv[0], byrefSpv.len * 4)
  let (valOut, valCode) = execCmdEx(quoteShell(spirvVal) & " --target-env vulkan1.4 " & quoteShell(spvPath))
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

  var engine = newEngine(runVulkan)
  engine.ingest(byrefSpv)
  let oDev = engine.allocDevice(float32, 1)

  engine.launchKernel("storeSumKernel", oDev, (2.0'f32, 0))
  var got: array[1, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 2.0'f32, "first=0: got " & $got[0] & ", want 2.0 * 1.0 = 2.0"

  engine.launchKernel("storeSumKernel", oDev, (0.5'f32, 255))
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 128.0'f32, "first=255: got " & $got[0] & ", want 0.5 * 256.0 = 128.0"
  echo "[OK] direct SPIR-V: struct with array parameter crosses as a pointer"

when isMainModule:
  runTest()
  echo "t_spirv_byref_param: all checks passed"
