## Metal batch: the stored enqueue proxy compiles into a refusal.
##
## Test that binding the `<<` chevron result to a variable fails to compile.
##
## Run from the worktree root (the compile must fail):
##   nim c --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/batching/manual_metal_batch_stored_proxy_refused.nim

import workspace/crucible

proc scaleKernel(outP, x: ptr UncheckedArray[float32], s: float32) =
  let i = int(thread_position_in_grid.x)
  outP[i] = s * x[i]

proc runTest[E: HwEngine](engine: E) =
  var xD = engine.allocDevice(float32, 64)
  var outD = engine.allocDevice(float32, 64)
  var batch = engine.recordBatch()
  let proxy = batch.enqueue << (1, 64)
  proxy >> scaleKernel(outD, xD, 3.0'f32)

when isMainModule:
  runTest(newEngine(runMetal))
