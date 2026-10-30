## Vulkan (GLSL): non-finite float literals emit the uintBitsToFloat form
##
## Test that a kernel whose constants fold to +inf, -inf and NaN compiles
## through glslang and returns infinities and NaN
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/functions/t_vulkan_inf_nan.nim

import std/[math, strutils]
import workspace/crucible

const infNanGlsl = glsl:
  proc infNanKernel(output: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1, 1).} =
    const posInf = Inf
    const negInf = -Inf
    const nanVal = NaN
    output[0] = float32(posInf)
    output[1] = float32(negInf)
    output[2] = float32(nanVal)

proc runTest() =
  var engine = newEngine(runVulkan)
  engine.ingest(infNanGlsl)
  doAssert "uintBitsToFloat(0x7f800000u)" in engine.getArtifact(),
    "the +inf literal did not emit the uintBitsToFloat bit pattern"
  doAssert "uintBitsToFloat(0xff800000u)" in engine.getArtifact(),
    "the -inf literal did not emit the uintBitsToFloat bit pattern"
  doAssert "uintBitsToFloat(0x7fc00000u)" in engine.getArtifact(),
    "the NaN literal did not emit the uintBitsToFloat bit pattern"
  let oDev = engine.allocDevice(float32, 3)
  engine.launchKernel("infNanKernel", oDev, ())
  var res: array[3, float32]
  engine.copyToHostFromDevice(addr res[0], oDev, 3)
  doAssert res[0].classify == fcInf, "output[0] = " & $res[0]
  doAssert res[1].classify == fcNegInf, "output[1] = " & $res[1]
  doAssert res[2].classify == fcNan, "output[2] = " & $res[2]
  echo "  OK - t_vulkan_inf_nan"
  flushFile(stdout)

when isMainModule:
  runTest()
