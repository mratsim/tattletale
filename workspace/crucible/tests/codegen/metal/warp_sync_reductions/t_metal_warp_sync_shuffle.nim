## Metal: `simd_shuffle` and `simd_shuffle_xor` cross-lane shuffles.
## Test that `simd_shuffle` gathers from an absolute lane index while
## `simd_shuffle_xor` exchanges with the lane at callerLane xor mask,
## bitwise against host-side expectations.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/warp_sync_reductions/t_metal_warp_sync_shuffle.nim

import std/[strutils, unittest]
import workspace/crucible

const LaneWidth = 32

const broadcastMsl = metal:
  proc broadcastLane0(outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let v = uint32(0xABCD0000'u32 or uint32(lane))
    outBuf[lane] = simd_shuffle(v, 0)

const butterflyMsl = metal:
  proc butterflyXor1(outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    # xor-1 butterfly: lane i exchanges with lane i xor 1
    outBuf[lane] = simd_shuffle_xor(uint32(lane), 1)

const butterflyHighMsl = metal:
  proc butterflyXor16(outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    # xor-16 butterfly: lane i exchanges with lane i xor 16
    outBuf[lane] = simd_shuffle_xor(uint32(lane), 16)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - cross-lane shuffles":


    test "simd_shuffle broadcasts lane 0 to every lane":
      var engine = newEngine(runMetal)
      engine.ingest(broadcastMsl)
      let oDev = engine.allocDevice(uint32, LaneWidth)
      engine.launchKernel<<(grid: (1, 1), blk: (LaneWidth, 1))>>("broadcastLane0", oDev, ())
      var res: array[LaneWidth, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
      for lane in 0 ..< LaneWidth:
        check res[lane] == 0xABCD0000'u32

    test "simd_shuffle_xor pairs each lane with lane i xor 1":
      var engine = newEngine(runMetal)
      engine.ingest(butterflyMsl)
      let oDev = engine.allocDevice(uint32, LaneWidth)
      engine.launchKernel<<(grid: (1, 1), blk: (LaneWidth, 1))>>("butterflyXor1", oDev, ())
      var res: array[LaneWidth, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
      for lane in 0 ..< LaneWidth:
        check res[lane] == uint32(lane xor 1)

    test "simd_shuffle_xor pairs each lane with lane i xor 16":
      var engine = newEngine(runMetal)
      engine.ingest(butterflyHighMsl)
      let oDev = engine.allocDevice(uint32, LaneWidth)
      engine.launchKernel<<(grid: (1, 1), blk: (LaneWidth, 1))>>("butterflyXor16", oDev, ())
      var res: array[LaneWidth, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
      for lane in 0 ..< LaneWidth:
        check res[lane] == uint32(lane xor 16)

when isMainModule:
  runTest()
