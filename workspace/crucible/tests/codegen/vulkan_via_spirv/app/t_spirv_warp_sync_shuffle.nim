## SPIR-V binary emission: shuffle kernels execute through the Vulkan engine
## Test that `simd_shuffle` gathers from an absolute lane index while
## `simd_shuffle_xor` exchanges with the lane at callerLane xor mask
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_warp_sync_shuffle.nim

import workspace/crucible

const LaneWidth = 32

const broadcastSpv = spirv:
  proc broadcastLane0(outBuf: ptr UncheckedArray[uint32]) {.global, workgroup: (32, 1, 1).} =
    let lane = thread_index_in_threadgroup
    let v = 1000'u32 + uint32(lane)
    outBuf[lane] = simd_shuffle(v, 0)

const butterflySpv = spirv:
  proc butterflyXor1(outBuf: ptr UncheckedArray[uint32]) {.global, workgroup: (64, 1, 1).} =
    let lane = thread_index_in_threadgroup
    # xor-1 butterfly: lane i exchanges with lane i xor 1
    outBuf[lane] = simd_shuffle_xor(uint32(lane), 1)

proc runTest() =
  doAssert broadcastSpv.len > 0 and broadcastSpv[0] == 0x07230203'u32,
    "`spirv:` macro emitted no words or lost the magic number"

  var engine = newEngine(runVulkan)

  block broadcast:
    engine.ingest(broadcastSpv)
    let lanes = 32
    let oDev = engine.allocDevice(uint32, lanes)
    engine.launchKernel("broadcastLane0", oDev, ())
    var res: array[LaneWidth, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
    for lane in 0 ..< LaneWidth:
      doAssert res[lane] == 1000'u32,
        "broadcast: lane " & $lane & " read " & $res[lane] & ", expected 1000"

  block butterfly:
    engine.ingest(butterflySpv)
    let lanes = 64
    let oDev = engine.allocDevice(uint32, lanes)
    engine.launchKernel("butterflyXor1", oDev, ())
    var res: array[64, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 64)
    for lane in 0 ..< 64:
      doAssert res[lane] == uint32(lane xor 1),
        "butterfly: lane " & $lane & " read " & $res[lane] & ", expected " & $(lane xor 1)

  echo "  OK - t_spirv_warp_sync_shuffle"

when isMainModule:
  runTest()
