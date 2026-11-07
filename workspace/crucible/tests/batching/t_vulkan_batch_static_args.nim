## Vulkan batch: compile-time arguments are baked into the wrapper, never packed.
##
## Test that a static parameter occupies no push-constant bytes and the shader
## reads runtime arguments at their own offsets.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/wip-vleg --nimcache:nimcache/wip-vleg \
##     workspace/crucible/tests/batching/t_vulkan_batch_static_args.nim

import std/unittest

import workspace/crucible
import workspace/crucible/src/runtime/backends/vulkan_common
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/batching/batch_vulkan

const N = 16

proc fillKernel(outP: ptr UncheckedArray[uint32], n: static int, v: uint32) =
  let i = int(thread_position_in_grid.x)
  if i < n:
    outP[i] = v + uint32(n)

proc setKernel(outP: ptr UncheckedArray[uint32], n: static int) =
  let i = int(thread_position_in_grid.x)
  if i < n:
    outP[i] = uint32(n)

proc runTest() =
  suite "Vulkan batch - static arguments excluded from push constants":

    test "static int between runtime args leaves the push block runtime-only":
      var ctx = newVulkanContext()
      let outD = DeviceMem[VulkanBuffer, uint32](mem: ctx.allocBuffer(N * sizeof(uint32)))
      var batch = newVulkanBatch(ctx)
      batch.recordGlobalFn(fillKernel, LaunchConfig(grid: dim3(1), blk: dim3(N)), outD, (N, 7'u32))
      # 8-byte device address + one uint32, the static int contributes nothing
      check batch.getScalarSize == 12
      batch.finalize()

      var zeros: array[N, uint32]
      var outBuf = outD.mem
      ctx.writeBuffer(outBuf, zeros[0].addr, N * sizeof(uint32))
      batch.replay()
      let got = readBuffer[uint32](ctx, outD.mem)
      for i in 0 ..< N:
        check got[i] == uint32(N + 7)

    test "all-static tail packs only the output address":
      var ctx = newVulkanContext()
      let outD = DeviceMem[VulkanBuffer, uint32](mem: ctx.allocBuffer(N * sizeof(uint32)))
      var batch = newVulkanBatch(ctx)
      batch.recordGlobalFn(setKernel, LaunchConfig(grid: dim3(1), blk: dim3(N)), outD, (N,))
      check batch.getScalarSize == 8
      batch.finalize()

      var zeros: array[N, uint32]
      var outBuf = outD.mem
      ctx.writeBuffer(outBuf, zeros[0].addr, N * sizeof(uint32))
      batch.replay()
      let got = readBuffer[uint32](ctx, outD.mem)
      for i in 0 ..< N:
        check got[i] == uint32(N)

when isMainModule:
  runTest()
