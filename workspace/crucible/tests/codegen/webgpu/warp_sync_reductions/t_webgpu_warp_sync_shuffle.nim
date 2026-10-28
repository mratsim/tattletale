## WebGPU: `simd_shuffle` and `simd_shuffle_xor` cross-lane shuffles.
## Test that `subgroupShuffle` broadcasts the subgroup's lane 0 while
## `subgroupShuffleXor` exchanges with the lane at callerLane xor mask,
## bitwise over 32 lanes
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/warp_sync_reductions/t_webgpu_warp_sync_shuffle.nim

import std/unittest
import workspace/crucible

const LaneWidth = 32

const broadcastWgsl = webgpu:
  proc broadcastLane0(outBuf: ptr UncheckedArray[uint32]) {.global, workgroup: (32, 1, 1).} =
    let lane = thread_index_in_threadgroup
    let v = uint32(0xABCD0000'u32 or uint32(lane))
    outBuf[lane] = subgroupShuffle(v, 0)

const butterflyWgsl = webgpu:
  proc butterflyXor1(outBuf: ptr UncheckedArray[uint32]) {.global, workgroup: (32, 1, 1).} =
    let lane = thread_index_in_threadgroup
    # xor-1 butterfly: lane i exchanges with lane i xor 1
    outBuf[lane] = subgroupShuffleXor(uint32(lane), 1)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # Requires an adapter exposing the subgroups device feature
  suite "WebGPU - cross-lane shuffles":

    test "simd_shuffle broadcasts lane 0 to every lane":
      var engine = newEngine(runWebGPU)
      engine.ingest(broadcastWgsl)
      let oDev = engine.allocDevice(uint32, LaneWidth)
      engine.launchKernel<<(grid: (1, 1), blk: (LaneWidth, 1))>>("broadcastLane0", oDev, ())
      var res: array[LaneWidth, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
      for lane in 0 ..< LaneWidth:
        check res[lane] == 0xABCD0000'u32

    test "simd_shuffle_xor pairs each lane with lane i xor 1":
      var engine = newEngine(runWebGPU)
      engine.ingest(butterflyWgsl)
      let oDev = engine.allocDevice(uint32, LaneWidth)
      engine.launchKernel<<(grid: (1, 1), blk: (LaneWidth, 1))>>("butterflyXor1", oDev, ())
      var res: array[LaneWidth, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
      for lane in 0 ..< LaneWidth:
        check res[lane] == uint32(lane xor 1)

when isMainModule:
  runTest()
