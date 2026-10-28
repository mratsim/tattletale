## CUDA: `simd_shuffle` and `simd_shuffle_xor` cross-lane shuffles.
## Test that `shfl_sync` broadcasts the warp's lane 0 and `shfl_xor_sync`
## exchanges with the lane at callerLane xor mask, bitwise over 32 lanes.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/warp_sync_reductions/t_cuda_warp_sync_shuffle.nim

import std/unittest
import workspace/crucible

const LaneWidth = 32

const broadcastCuda = cuda:
  proc broadcastLane0(outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let v = uint32(0xABCD0000'u32 or uint32(lane))
    outBuf[lane] = shfl_sync(v, 0)

const butterflyCuda = cuda:
  proc butterflyXor1(outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    # xor-1 butterfly: lane i exchanges with lane i xor 1
    outBuf[lane] = shfl_xor_sync(uint32(lane), 1)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "CUDA - cross-lane shuffles":

    test "shfl_sync broadcasts lane 0 to every lane":
      var engine = newEngine(runCuda)
      engine.ingest(broadcastCuda)
      let oDev = engine.allocDevice(uint32, LaneWidth)
      engine.launchKernel<<(grid: (1, 1), blk: (LaneWidth, 1))>>("broadcastLane0", oDev, ())
      var res: array[LaneWidth, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
      for lane in 0 ..< LaneWidth:
        check res[lane] == 0xABCD0000'u32

    test "shfl_xor_sync pairs each lane with lane i xor 1":
      var engine = newEngine(runCuda)
      engine.ingest(butterflyCuda)
      let oDev = engine.allocDevice(uint32, LaneWidth)
      engine.launchKernel<<(grid: (1, 1), blk: (LaneWidth, 1))>>("butterflyXor1", oDev, ())
      var res: array[LaneWidth, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
      for lane in 0 ..< LaneWidth:
        check res[lane] == uint32(lane xor 1)

when isMainModule:
  runTest()
