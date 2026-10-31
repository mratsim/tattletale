## WebGPU (WGSL): non-finite float literals emit the bitcast form
##
## Test that a kernel whose constants fold to +inf, -inf and NaN produces
## WGSL that the WebGPU implementation accepts and returns infinities and NaN
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/functions/t_webgpu_inf_nan.nim

import std/math
import workspace/crucible

const infNanWgsl = webgpu:
  proc infNanKernel(output: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1, 1).} =
    const posInf = Inf
    const negInf = -Inf
    const nanVal = NaN
    output[0] = float32(posInf)
    output[1] = float32(negInf)
    output[2] = float32(nanVal)

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(infNanWgsl)
  let oDev = engine.allocDevice(float32, 3)
  engine.launchKernel("infNanKernel", oDev, ())
  var res: array[3, float32]
  engine.copyToHostFromDevice(addr res[0], oDev, 3)
  doAssert res[0].classify == fcInf, "output[0] = " & $res[0]
  doAssert res[1].classify == fcNegInf, "output[1] = " & $res[1]
  doAssert res[2].classify == fcNan, "output[2] = " & $res[2]
  echo "  OK - t_webgpu_inf_nan"
  flushFile(stdout)

when isMainModule:
  runTest()
