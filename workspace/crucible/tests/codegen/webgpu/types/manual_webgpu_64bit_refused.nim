## WebGPU: 64-bit scalars refuse at compile time.
##
## Test that a `webgpu:` kernel with a float64 parameter fails to compile,
## the refusal names the shader-f64 device feature the runtime cannot request.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/types/manual_webgpu_64bit_refused.nim

import workspace/crucible

const code = webgpu:
  proc k(output: ptr UncheckedArray[float64]) {.global, workgroup: (1, 1, 1).} =
    output[0] = 1.0'f64

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(code)
  let oDev = engine.allocDevice(float64, 1)
  engine.launchKernel("k", oDev, ())
  echo "UNREACHABLE — a 64-bit kernel must refuse at compile time"

when isMainModule:
  runTest()
