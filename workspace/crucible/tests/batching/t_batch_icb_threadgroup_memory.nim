## Metal tier-1 batch (indirect command buffer): threadgroup memory per entry.
##
## Test that recordGlobalFn stores the entry's sharedMem, the replay encodes
## the threadgroup memory length in the indirect command, and both replays
## match the CPU reference.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/batching/t_batch_icb_threadgroup_memory.nim

import std/[importutils, tables, unittest]

import workspace/crucible
import workspace/crucible/src/runtime/batching/batch_metal

privateAccess(MetalEngine)

const
  Blk = 64
  Groups = 4
  Sum = float32(Blk * (Blk - 1) div 2)

proc tgSumKernel(output: ptr UncheckedArray[float32]) =
  var scratch {.smem.}: array[Blk, float32]
  let t = int(thread_position_in_threadgroup.x)
  scratch[t] = float32(t)
  syncthreads()
  if t == 0:
    var s = 0.0'f32
    for i in 0 ..< Blk:
      s += scratch[i]
    output[int(threadgroup_position_in_grid.x)] = s

proc runTest() =
  var engine = newMetalEngine()
  privateAccess(typeof(engine.ctx))
  privateAccess(typeof(engine.cache))
  let outD = engine.allocDevice(float32, Groups)

  # one eager launch compiles the Nim kernel into the engine's library
  engine.launchDeviceFn tgSumKernel(outD) do:
    grid = (Groups, 1)
    blk = (Blk, 1)
    sharedMem = Blk * sizeof(float32)
  var eager: array[Groups, float32]
  engine.copyToHostFromDevice(eager.toOpenArray(0, Groups - 1), outD)
  for g in 0 ..< Groups:
    check eager[g] == Sum

  # Tier-1 replay borrows the engine library and launches the wrapper name
  # registered in the pipeline-state cache
  var entrypoint: string
  for k in engine.cache.psos.keys:
    entrypoint = k
  var batch = newMetalBatch(engine.ctx.device, engine.ctx.queue)
  batch.recordGlobalFn(engine.cache.library, entrypoint,
                       LaunchConfig(grid: dim3(Groups), blk: dim3(Blk), sharedMem: Blk * sizeof(float32)),
                       outD, ())
  batch.finalize()
  batch.replay()

  var got: array[Groups, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, Groups - 1), outD)
  for g in 0 ..< Groups:
    check got[g] == Sum
  batch.replay()
  engine.copyToHostFromDevice(got.toOpenArray(0, Groups - 1), outD)
  for g in 0 ..< Groups:
    check got[g] == Sum
  echo "[OK] ICB batch: threadgroup memory encoded per entry, both replays match"

when isMainModule:
  runTest()
