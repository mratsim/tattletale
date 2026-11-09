## Metal batch: the inline chevron form enqueues and launches.
##
## Test that the inline chevron records, submits and launches one entry whose
## numerics match the CPU reference.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/batching/t_metal_batch_enqueue_inline.nim

import workspace/crucible
import workspace/crucible/src/runtime/batching/batch
import workspace/crucible/src/runtime/engines/engines_api

const N = 64

proc scaleKernel(outP, x: ptr UncheckedArray[float32], s: float32) =
  let i = int(thread_position_in_grid.x)
  outP[i] = s * x[i]

proc runTest[E: HwEngine](engine: E) =
  var xD = engine.allocDevice(float32, N)
  var outD = engine.allocDevice(float32, N)
  var xs: array[N, float32]
  for i in 0 ..< N:
    xs[i] = float32(i)
  engine.copyToDeviceFromHost(xD, xs.toOpenArray(0, N - 1))
  var batch = engine.recordBatch()
  batch.enqueue<<(1, 64)>> scaleKernel(outD, xD, 3.0'f32)
  batch.submit()
  var got: array[N, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, N - 1), outD)
  for i in 0 ..< N:
    doAssert got[i] == 3.0'f32 * xs[i], "scale mismatch at " & $i & ": " & $got[i]
  echo "[OK] inline enqueue launches"

when isMainModule:
  runTest(newEngine(runMetal))
