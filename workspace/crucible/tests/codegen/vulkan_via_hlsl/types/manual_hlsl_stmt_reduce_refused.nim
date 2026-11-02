## HLSL: a refused builtin kind in statement position refuses.
##
## Test that `discard reduceAdd(...)` on an `hlsl:` kernel fails to compile, the refusal names the missing subgroup emission.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/types/manual_hlsl_stmt_reduce_refused.nim

import workspace/crucible

const code = hlsl:
  proc k(o: ptr UncheckedArray[float32]) {.global.} =
    let v = 1.5'f32
    discard reduceAdd(v)
    o[0] = v

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(code)
  let oDev = engine.allocDevice(float32, 1)
  engine.launchKernel("k", oDev, ())
  echo "UNREACHABLE — a statement-position refused builtin must refuse at compile time"

when isMainModule:
  runTest()
