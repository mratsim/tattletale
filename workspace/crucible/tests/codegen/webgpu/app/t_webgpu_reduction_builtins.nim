## WebGPU: the subgroup shuffles (`simdShuffleDown`, `simdShuffle`) emit WGSL reduction builtins,
## device execution requires `wgslSubgroupsExecutable = true`.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/app/t_webgpu_reduction_builtins.nim

import workspace/crucible

## Device-execution switch: the WGSL carries the `enable subgroups` directive,
## rejected at shader parse by the vendored wgpu-native naga (wgpu#5555).
## The const stays false until the runtime accepts subgroups.
const wgslSubgroupsExecutable* = false

const reductionWgsl = webgpu:
  proc reductionKernel(output: ptr UncheckedArray[float32]) {.global, workgroup: (2, 1, 1).} =
    let lane = thread_position_in_threadgroup.x
    let acc = float32(lane) + 10.0'f32
    # Lane 0 gathers lane 1's value through the relative shuffle.
    let v = simdShuffleDown(acc, 1)
    # Both lanes gather lane 0's shuffled-down value through the absolute shuffle.
    let w = simdShuffle(v, 0)
    if lane == 0:
      output[0] = v
    output[lane + 1] = w

proc runTest() =   # private, tests run in a proc so engines are destroyed at return
  when wgslSubgroupsExecutable:
    block: # reductionKernel: out = [acc(lane 1), v(lane 0), v(lane 0)]
      var engine = bkWGSL.init()
      engine.ingest(reductionWgsl)

      var res: array[3, float32]
      let oDev = engine.allocDevice(float32, res.len)
      engine.launchKernel("reductionKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)

      echo "  reductionKernel: output = [", res[0], ", ", res[1], ", ", res[2], "]"
      doAssert res[0] == 11.0'f32  # lane 0: shuffleDown gathers lane 1's acc
      doAssert res[1] == 11.0'f32  # lane 0: shuffle(v, 0) gathers its own v, acc(lane 1)
      doAssert res[2] == 11.0'f32  # lane 1: shuffle(v, 0) gathers lane 0's v, acc(lane 1)
      echo "  OK"

    echo "All reduction builtin execution tests passed"
  else:
    # Emission check: the WGSL is ingested and read back without dispatch.
    var engine = bkWGSL.init()
    engine.ingest(reductionWgsl)
    discard engine.getArtifact()
    echo "  SKIP: WGSL subgroup execution unavailable, naga rejects enable subgroups (wgpu#5555)"

when isMainModule:
  runTest()
