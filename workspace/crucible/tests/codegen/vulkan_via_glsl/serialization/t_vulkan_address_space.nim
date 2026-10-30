## Vulkan (GLSL): address-space semantics for locals and module-scope `{.smem.}`.
## Unannotated locals are private per-thread registers, `{.smem.}` storage is
## workgroup-shared: one thread's write is visible to the group after a barrier.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/serialization/t_vulkan_address_space.nim

import std/unittest
import workspace/crucible

const addressSpaceVk = glsl:
  var scratch {.smem.}: array[8, uint32]
  proc addressSpaceKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    var x: uint32 = 7
    scratch[0] = x
    output[0] = scratch[0]

const scratchSumVk = glsl:
  var scratch {.smem.}: array[8, uint32]
  proc scratchSumKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (8, 1).} =
    let tid = int(thread_position_in_threadgroup.x)
    scratch[tid] = uint32(tid)
    threadgroup_barrier()
    if tid == 0:
      var acc: uint32 = 0
      for i in 0 ..< 8:
        acc += scratch[i]
      output[0] = acc

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Vulkan - address-space semantics":
    test "an unannotated local and a shared scratch store and load":
      var engine = newEngine(runVulkan)
      engine.ingest(addressSpaceVk)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel<<(grid: (1, 1), blk: (4, 2))>>("addressSpaceKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 7'u32

    test "the shared scratch is one workgroup-wide array":
      var engine = newEngine(runVulkan)
      engine.ingest(scratchSumVk)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel<<(grid: (1, 1), blk: (8, 1))>>("scratchSumKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 28'u32

when isMainModule:
  runTest()
