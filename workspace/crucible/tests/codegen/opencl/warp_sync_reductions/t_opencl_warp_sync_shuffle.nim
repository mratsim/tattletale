## OpenCL: `simd_shuffle` and `simd_shuffle_xor` cross-lane shuffles.
## Test that the shuffle builtins emit `sub_group_shuffle` and `sub_group_shuffle_xor`
## and exchange values bitwise against host-side expectations over 32 lanes. Requires OpenCL 2.1+ hardware (sub-group shuffles are core there).
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/warp_sync_reductions/t_opencl_warp_sync_shuffle.nim

import std/[strutils, unittest]
import workspace/crucible

const LaneWidth = 32

const broadcastCl = opencl:
  proc broadcastLane0(outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let v = uint32(0xABCD0000'u32 or uint32(lane))
    outBuf[lane] = sub_group_shuffle(v, 0)

const butterflyCl = opencl:
  proc butterflyXor1(outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    # xor-1 butterfly: lane i exchanges with lane i xor 1
    outBuf[lane] = sub_group_shuffle_xor(uint32(lane), 1)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "OpenCL - cross-lane shuffles":


    test "simd_shuffle broadcasts lane 0 to every lane":
      var engine = newEngine(runOpenCL)
      engine.ingest(broadcastCl)
      let oDev = engine.allocDevice(uint32, LaneWidth)
      engine.launchKernel<<(grid: (1, 1), blk: (LaneWidth, 1))>>("broadcastLane0", oDev, ())
      var res: array[LaneWidth, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
      for lane in 0 ..< LaneWidth:
        check res[lane] == 0xABCD0000'u32

    test "simd_shuffle_xor pairs each lane with lane i xor 1":
      var engine = newEngine(runOpenCL)
      engine.ingest(butterflyCl)
      let oDev = engine.allocDevice(uint32, LaneWidth)
      engine.launchKernel<<(grid: (1, 1), blk: (LaneWidth, 1))>>("butterflyXor1", oDev, ())
      var res: array[LaneWidth, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
      for lane in 0 ..< LaneWidth:
        check res[lane] == uint32(lane xor 1)

when isMainModule:
  runTest()
