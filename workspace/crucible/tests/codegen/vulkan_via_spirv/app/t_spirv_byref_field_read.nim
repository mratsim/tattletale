## Direct SPIR-V: `{.device.}` function parameters carrying structs pass as pointers
## Test that a scalar field read of a by-ref struct parameter emits valid SPIR-V
##
## The callee reads a by-ref parameter through a dereference of the renamed
## pointer symbol. OpLoad yields the pointee struct and the read chains over it.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_byref_field_read.nim

import std/[os, osproc]
import workspace/crucible

type
  Sample = object
    weight: float32
    bias: float32
    values: array[64, float32]

const byrefFieldReadSpv = spirv:
  proc sampleWeight(s: Sample, first: int): float32 {.device.} =
    s.weight * float32(first)

  proc sampleKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var s: Sample
    s.weight = 2.5'f32
    for i in 0 ..< 64:
      s.values[i] = float32(i + 1)
    output[0] = sampleWeight(s, 3)

proc runTest() =
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_byref_field_read.spv"
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr byrefFieldReadSpv[0], byrefFieldReadSpv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

  var engine = newEngine(runVulkan)
  engine.ingest(byrefFieldReadSpv)
  let oDev = engine.allocDevice(float32, 1)

  engine.launchKernel("sampleKernel", oDev, ())
  var got: array[1, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 7.5'f32, "got " & $got[0] & ", want 2.5 * 3.0 = 7.5"
  echo "[OK] direct SPIR-V: scalar field read of a by-ref struct parameter"

when isMainModule:
  runTest()
  echo "t_spirv_byref_field_read: all checks passed"
