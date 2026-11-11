## Direct SPIR-V: a by-ref device call can take a kernel's push-block parameter
## Test that a kernel calling a device function taking a struct by reference reads back exact values
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_byref_push_param.nim

import std/[os, osproc]
import workspace/crucible

type
  Count = object
    n: int32
    pad: array[16, int32]

const byrefPushParamSpv = spirv:
  proc readCount(c: Count): int32 {.device.} =
    c.n + 1'i32

  proc pushKernel(output: ptr UncheckedArray[int32], w: Count) {.global.} =
    output[0] = readCount(w)

proc runTest() =
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_byref_push_param.spv"
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr byrefPushParamSpv[0], byrefPushParamSpv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

  var engine = newEngine(runVulkan)
  engine.ingest(byrefPushParamSpv)
  let oDev = engine.allocDevice(int32, 1)

  var w: Count
  w.n = 41'i32
  engine.launchKernel("pushKernel", oDev, (w,))
  var got: array[1, int32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 42'i32, "got " & $got[0] & ", want 41 + 1 = 42"
  echo "[OK] direct SPIR-V: push-block struct parameter crosses to a by-ref call"

when isMainModule:
  runTest()
  echo "t_spirv_byref_push_param: all checks passed"
