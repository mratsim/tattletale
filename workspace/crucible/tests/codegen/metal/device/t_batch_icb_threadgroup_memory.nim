## Metal tier-1 batch (indirect command buffer): threadgroup memory per entry.
##
## Test that recordGlobalFn stores the entry's sharedMem, the replay encodes
## the threadgroup memory length in the indirect command, and both replays
## match the CPU reference.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/metal/device/t_batch_icb_threadgroup_memory.nim

import std/[importutils, unittest]

import workspace/crucible
import workspace/crucible/src/runtime/batching/batch_metal

privateAccess(MetalEngine)

const
  Blk = 64
  Groups = 4
  Sum = float32(Blk * (Blk - 1) div 2)

const tgSumSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void tgSum(device float* output [[buffer(0)]],
                  uint3 tid [[thread_position_in_threadgroup]],
                  uint3 bid [[threadgroup_position_in_grid]]) {
  threadgroup float scratch[64];
  scratch[tid.x] = float(tid.x);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid.x == 0) {
    float s = 0.0;
    for (int i = 0; i < 64; i++) s += scratch[i];
    output[bid.x] = s;
  }
}
"""

proc runTest() =
  var engine = newMetalEngine()
  privateAccess(typeof(engine.ctx))
  privateAccess(typeof(engine.cache))
  engine.ingest(tgSumSrc)
  let outD = engine.allocDevice(float32, Groups)

  var batch = newMetalBatch(engine.ctx.device, engine.ctx.queue)
  batch.recordGlobalFn(engine.cache.library, "tgSum",
                       LaunchConfig(grid: dim3(Groups), blk: dim3(Blk), sharedMem: Blk * sizeof(float32)),
                       outD, ())
  batch.finalize()
  batch.replay()

  var got: array[Groups, float32]
  engine.copyToHostFromDevice(addr got[0], outD, Groups)
  for g in 0 ..< Groups:
    check got[g] == Sum
  batch.replay()
  engine.copyToHostFromDevice(addr got[0], outD, Groups)
  for g in 0 ..< Groups:
    check got[g] == Sum
  echo "[OK] ICB batch: threadgroup memory encoded per entry, both replays match"

when isMainModule:
  runTest()
