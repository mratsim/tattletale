## HLSL: `simd_shuffle` cross-lane shuffle through the DXC route.
## Test that a `WaveReadLaneAt` kernel compiles through DXC and executes
## on the device, every lane reading lane 0 of its subgroup
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/warp_sync_reductions/t_hlsl_warp_sync_shuffle.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const LaneWidth = 32

const broadcastHlsl = hlsl:
  proc broadcastLane0(output: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let v = 1000'u32 + uint32(lane)
    output[lane] = WaveReadLaneAt(v, 0)

proc runTest() =
  let bindings: seq[BindingExpectation] = @[("output", 0, false)]
  let words = compileCheck("shuffle_broadcast", broadcastHlsl, bindings)

  var engine = newEngine(runVulkan)
  engine.ingest(words)
  let lanes = 32
  let oDev = engine.allocDevice(uint32, lanes)
  engine.launchKernel("main", oDev, ())
  var res: array[LaneWidth, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, LaneWidth)
  for lane in 0 ..< LaneWidth:
    doAssert res[lane] == 1000'u32,
      "broadcast: lane " & $lane & " read " & $res[lane] & ", expected 1000"

  echo "  OK - t_hlsl_warp_sync_shuffle"

when isMainModule:
  runTest()
