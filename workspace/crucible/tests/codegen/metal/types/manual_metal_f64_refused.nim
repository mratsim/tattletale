## Metal: an f64 leaf refuses at compile time.
##
## Test that a `metal:` kernel with a float64 parameter fails to compile,
## the refusal names the f64 cell MSL does not name.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/manual_metal_f64_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[float32], x: float64) {.global.} =
    output[0] = float32(x)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runMetal)
  engine.ingest(code)
  let oDev = engine.allocDevice(float32, 1)
  engine.launchKernel("k", oDev, (1.5'f64,))
  echo "UNREACHABLE — an f64 kernel must refuse at compile time"

when isMainModule:
  runTest()
