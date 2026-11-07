## Metal batch: threadgroup memory in the chevron and the replay.
##
## Test that a kernel with `{.smem.}` locals replays with a per-entry
## threadgroup memory length. The batched result matches the eager launch
## and the CPU reference.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/batching/t_batch_threadgroup_memory.nim

import std/unittest

import workspace/crucible
import workspace/crucible/src/runtime/batching/batch
import workspace/crucible/src/runtime/engines/engines_api

const
  Blk = 64
  Groups = 4
  Sum = float32(Blk * (Blk - 1) div 2)  # 0 + 1 + ... + 63

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

proc runBatch[E: HwEngine](engine: E) =
  let outD = engine.allocDevice(float32, Groups)

  # eager reference: same kernel, same config, the launchDeviceFn route
  engine.launchDeviceFn tgSumKernel(outD) do:
    grid = (Groups, 1)
    blk = (Blk, 1)
    sharedMem = Blk * sizeof(float32)
  var eager: array[Groups, float32]
  engine.copyToHostFromDevice(addr eager[0], outD, Groups)
  for g in 0 ..< Groups:
    check eager[g] == Sum

  # batch route: the smem length is encoded per entry in the indirect command
  var batch = engine.recordBatch()
  batch.enqueue<<(Groups, Blk, Blk * sizeof(float32))>> tgSumKernel(outD)
  batch.submit()
  var batched: array[Groups, float32]
  engine.copyToHostFromDevice(addr batched[0], outD, Groups)
  for g in 0 ..< Groups:
    check batched[g] == Sum

proc runTest() =
  suite "Metal batch - threadgroup memory":

    test "batched smem kernel matches the eager launch":
      var engine = newEngine(runMetal)
      engine.runBatch()

when isMainModule:
  runTest()
