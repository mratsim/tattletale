## Vulkan: `reduceAdd` warp-synchronous reduction.
## Test that `reduceAdd`/`reduceMax` reduce across the 32 lanes and broadcast to every lane.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/warp_sync_reductions/t_vulkan_warp_sync_sum_max.nim

import std/[strutils, unittest]
import workspace/crucible

const scalarVk = glsl:
  proc sumScalar(outBuf: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    let lane = thread_index_in_threadgroup
    let v = float32(lane)
    outBuf[lane] = reduceAdd(v)

const partialsVk = glsl:
  proc sumPartials(outBuf: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    let lane = thread_index_in_threadgroup
    for j in 0 ..< 4:
      let v = float32(lane) + float32(j)
      outBuf[lane * 4 + j] = reduceAdd(v)

const maxScalarVk = glsl:
  proc maxScalar(outBuf: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    let lane = thread_index_in_threadgroup
    let v = float32(lane)
    outBuf[lane] = reduceMax(v)

const maxPartialsVk = glsl:
  proc maxPartials(outBuf: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    let lane = thread_index_in_threadgroup
    for j in 0 ..< 4:
      let v = float32(lane) + float32(j)
      outBuf[lane * 4 + j] = reduceMax(v)

  # Worked example with 4 threads and 2 partials per thread
  # (the kernels below use 32 threads and 4 partials):
  #
  #   thread:      t0   t1   t2   t3
  #   partial 0:    0    1    2    3      thread t holds t + 0
  #   partial 1:    1    2    3    4      thread t holds t + 1
  #
  #   partial 0 summed across threads: 0+1+2+3 = 6
  #   partial 1 summed across threads: 1+2+3+4 = 10
  #   every thread ends holding both totals.

  # Worked example with 4 threads and 2 partials per thread
  # (the kernels below use 32 threads and 4 partials):
  #
  #   thread:      t0   t1   t2   t3
  #   partial 0:    0    1    2    3      thread t holds t + 0
  #   partial 1:    1    2    3    4      thread t holds t + 1
  #
  #   partial 0 maxed across threads: max(0, 1, 2, 3) = 3
  #   partial 1 maxed across threads: max(1, 2, 3, 4) = 4
  #   every thread ends holding both maxima.

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Vulkan - warp-synchronous sum and max":

    test "reduceAdd emits subgroupAdd and the subgroup extensions":
      check "subgroupAdd(" in scalarVk
      check "#extension GL_KHR_shader_subgroup_basic : enable" in scalarVk
      check "#extension GL_KHR_shader_subgroup_arithmetic : enable" in scalarVk

    test "scalar reduce sums the 32 lanes and broadcasts to each":
      var engine = newEngine(runVulkan)
      engine.ingest(scalarVk)
      let oDev = engine.allocDevice(float32, 32)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("sumScalar", oDev, ())
      var res: array[32, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32)
      # sum of lanes 0..31 = 496, every lane holds the total
      for lane in 0 ..< 32:
        check res[lane] == 496.0

    test "reduceAdd: 4 partials per thread, each summed across the 32 threads":
      var engine = newEngine(runVulkan)
      engine.ingest(partialsVk)
      let oDev = engine.allocDevice(float32, 32 * 4)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("sumPartials", oDev, ())
      var res: array[32 * 4, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32 * 4)
      # partial j: thread t holds t + j, summed over t = 496 + 32*j
      for lane in 0 ..< 32:
        for j in 0 ..< 4:
          check res[lane * 4 + j] == float32(496 + 32 * j)

    test "reduceMax takes the max of the 32 lanes and broadcasts it to each":
      var engine = newEngine(runVulkan)
      engine.ingest(maxScalarVk)
      let oDev = engine.allocDevice(float32, 32)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("maxScalar", oDev, ())
      var res: array[32, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32)
      # max of lanes 0..31 = 31, every lane holds it
      for lane in 0 ..< 32:
        check res[lane] == 31.0

    test "reduceMax: 4 partials per thread, each maxed across the 32 threads":
      var engine = newEngine(runVulkan)
      engine.ingest(maxPartialsVk)
      let oDev = engine.allocDevice(float32, 32 * 4)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("maxPartials", oDev, ())
      var res: array[32 * 4, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32 * 4)
      # partial j: thread t holds t + j, maxed over t = 31 + j
      for lane in 0 ..< 32:
        for j in 0 ..< 4:
          check res[lane * 4 + j] == float32(31 + j)

when isMainModule:
  runTest()
