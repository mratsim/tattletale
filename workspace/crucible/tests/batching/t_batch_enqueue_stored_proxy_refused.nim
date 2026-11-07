## batch enqueue refuses a stored enqueue proxy
##
## Test that the `>>` stage rejects an enqueue proxy bound to a variable at compile time.
## Test that the inline chevron form still enqueues and launches on Metal.
##
## A proxy bound to a variable reaches `>>` as a symbol with no children, so the block size for the workgroup pragma is unreadable.
## The stored form is refused at macro expansion.
##
## Run from the worktree root (the stored-proxy compile must fail):
##   nim c --hints:off --warnings:off -d:batchStoredProxy --path:"$PWD" \
##     --nimcache:nimcache/wip-batchcore --outdir:build/wip-batchcore \
##     workspace/crucible/tests/batching/t_batch_enqueue_stored_proxy_refused.nim
## Run from the worktree root (the inline form compiles and launches):
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip-batchcore --outdir:build/wip-batchcore \
##     workspace/crucible/tests/batching/t_batch_enqueue_stored_proxy_refused.nim

import workspace/crucible
import workspace/crucible/src/runtime/batching/batch
import workspace/crucible/src/runtime/engines/engines_api

const N = 64

proc scaleKernel(outP, x: ptr UncheckedArray[float32], s: float32) =
  let i = int(thread_position_in_grid.x)
  outP[i] = s * x[i]

proc runInline[E: HwEngine](engine: E) =
  var xD = engine.allocDevice(float32, N)
  var outD = engine.allocDevice(float32, N)
  var xs: array[N, float32]
  for i in 0 ..< N:
    xs[i] = float32(i)
  engine.copyToDeviceFromHost(xD, addr xs[0], N)
  var batch = engine.recordBatch()
  batch.enqueue<<(1, 64)>> scaleKernel(outD, xD, 3.0'f32)
  batch.submit()
  var got: array[N, float32]
  engine.copyToHostFromDevice(addr got[0], outD, N)
  for i in 0 ..< N:
    doAssert got[i] == 3.0'f32 * xs[i], "scale mismatch at " & $i & ": " & $got[i]
  echo "[OK] inline enqueue launches"

when defined(batchStoredProxy):
  proc runStored[E: HwEngine](engine: E) =
    var xD = engine.allocDevice(float32, N)
    var outD = engine.allocDevice(float32, N)
    var batch = engine.recordBatch()
    let proxy = batch.enqueue << (1, 64)
    proxy >> scaleKernel(outD, xD, 3.0'f32)

when isMainModule:
  var engine = newEngine(runMetal)
  when defined(batchStoredProxy):
    runStored(engine)
  else:
    runInline(engine)
