## Vulkan batch: overload resolution runs on the full call, output included.
##
## Test that two overloads of one kernel resolve by recorded argument count.
## The output counts as the first argument. Each entry runs its own overload.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/wip-vleg --nimcache:nimcache/wip-vleg \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_batch_overload.nim

import std/unittest

import workspace/crucible
import workspace/crucible/src/runtime/backends/vulkan_common
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/batching/batch_vulkan

const N = 8

proc fillKernel(outP: ptr UncheckedArray[uint32], v: uint32) =
  let i = int(thread_position_in_grid.x)
  if i < N:
    outP[i] = v

proc fillKernel(outP: ptr UncheckedArray[uint32], v, scale: uint32) =
  let i = int(thread_position_in_grid.x)
  if i < N:
    outP[i] = v * scale

proc runTest() =
  suite "Vulkan batch - overload resolution counts the output":

    test "one-arg and two-arg overloads of one kernel both record and run":
      var ctx = newVulkanContext()
      let outD = DeviceMem[VulkanBuffer, uint32](mem: ctx.allocBuffer(N * sizeof(uint32)))
      let proD = DeviceMem[VulkanBuffer, uint32](mem: ctx.allocBuffer(N * sizeof(uint32)))
      var batch = newVulkanBatch(ctx)
      batch.recordGlobalFn(fillKernel, LaunchConfig(grid: dim3(1), blk: dim3(N)), outD, (5'u32,))
      batch.recordGlobalFn(fillKernel, LaunchConfig(grid: dim3(1), blk: dim3(N)), proD, (3'u32, 4'u32))
      batch.finalize()
      batch.replay()
      let got = readBuffer[uint32](ctx, outD.mem)
      let product = readBuffer[uint32](ctx, proD.mem)
      for i in 0 ..< N:
        check got[i] == 5'u32
        check product[i] == 12'u32

when isMainModule:
  runTest()
