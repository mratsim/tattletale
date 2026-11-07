## Vulkan batch: pipeline reuse requires an identical entrypoint and module.
##
## Test that two entries of the same kernel name with different baked workgroup
## sizes dispatch on distinct pipelines. The kernel reads
## thread_position_in_threadgroup, which identifies the LocalSize baked into
## the pipeline that ran.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/wip-vleg --nimcache:nimcache/wip-vleg \
##     workspace/crucible/tests/batching/t_vulkan_batch_pipeline_key.nim

import std/unittest

import workspace/crucible
import workspace/crucible/src/runtime/backends/vulkan_common
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/batching/batch_vulkan

const N = 128

proc tidKernel(outP: ptr UncheckedArray[uint32]) =
  let i = int(thread_position_in_grid.x)
  let t = uint32(thread_position_in_threadgroup.x)
  if i < N:
    outP[i] = t

proc runTest() =
  suite "Vulkan batch - pipelines keyed by module, not entrypoint name":

    test "same kernel name, two workgroup sizes, each entry runs its own pipeline":
      var ctx = newVulkanContext()
      let outD = DeviceMem[VulkanBuffer, uint32](mem: ctx.allocBuffer(N * sizeof(uint32)))
      var batch = newVulkanBatch(ctx)
      # entry 1: two workgroups of 32 -> positions 0..31,0..31 at out[0..63]
      batch.recordGlobalFn(tidKernel, LaunchConfig(grid: dim3(2), blk: dim3(32)), outD, ())
      # entry 2: two workgroups of 64 -> positions 0..63,0..63 at out[0..127]
      batch.recordGlobalFn(tidKernel, LaunchConfig(grid: dim3(2), blk: dim3(64)), outD, ())
      batch.finalize()
      batch.replay()
      let got = readBuffer[uint32](ctx, outD.mem)
      for i in 0 ..< N:
        check got[i] == uint32(i mod 64)

when isMainModule:
  runTest()
