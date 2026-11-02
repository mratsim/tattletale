## WebGPU: a refused builtin kind in statement position refuses.
##
## Test that `discard atomicFetchAdd(...)` on a `webgpu:` kernel fails to compile, the refusal names the WGSL atomic restriction.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/types/manual_webgpu_stmt_atomic_refused.nim

import workspace/crucible

const code = webgpu:
  proc k(C: ptr UncheckedArray[int32]) {.global, workgroup: (4, 1, 1).} =
    discard atomicFetchAdd(addr C[0], 1'i32)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runWebGPU)
  engine.ingest(code)
  let cDev = engine.allocDevice(int32, 1)
  engine.launchKernel("k", cDev, ())
  echo "UNREACHABLE — a statement-position refused builtin must refuse at compile time"

when isMainModule:
  runTest()
