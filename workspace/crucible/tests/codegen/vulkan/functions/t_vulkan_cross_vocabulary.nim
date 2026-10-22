## Vulkan: cross-vocabulary coordinate + barrier kernel (GEMM-idiom slice).
## CUDA idiom `blockIdx * blockDim + threadIdx` and OpenCL idiom `get_global_id(d)`
## agree in one kernel, `thread_index_in_threadgroup` maps to native
## `gl_LocalInvocationIndex`, `syncthreads()` lowers to GLSL `barrier()`.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/functions/t_vulkan_cross_vocabulary.nim

import std/[strutils, unittest]
import workspace/crucible

const kernel2d = vulkan:
  proc crossVocab2d(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    let gx = blockIdx.x * blockDim.x + threadIdx.x
    let gy = blockIdx.y * blockDim.y + threadIdx.y
    C[gy * 8 + gx] = uint32(gy * 8 + gx)
    # OpenCL idiom in a Vulkan kernel: get_global_id(d) must agree with the CUDA-idiom global id above.
    if gx == get_global_id(0) and gy == get_global_id(1):
      # Position-derived flat-local value: with blk (4,2) the x-major flat
      # index maps (tx, ty) -> ty*4 + tx, so slot i holds tx + 10*ty for
      # the (tx, ty) at that slot. A y-major mapping permutes these values.
      C[64 + thread_index_in_threadgroup] =
        uint32(thread_position_in_threadgroup.x + 10 * thread_position_in_threadgroup.y)
    else:
      C[64 + thread_index_in_threadgroup] = 99'u32
    syncthreads()

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel2d

  suite "Vulkan - cross-vocabulary coordinates + barrier":

    test "CUDA + OpenCL idioms agree; flat local is native; barrier runs":
      var engine = bkVulkan.newEngine()
      engine.ingest(kernel2d)
      # The generated GLSL source (the `vulkan:` macro output) must carry
      # the native flat-index spelling. The Vulkan engine's getArtifact
      # returns compiled SPIR-V instead of source text, so the source is
      # grepped directly.
      check "gl_LocalInvocationIndex" in kernel2d
      let oDev = engine.allocDevice(uint32, 72)
      engine.launchKernel<<(grid: (2, 4), blk: (4, 2))>>("crossVocab2d", oDev, ())
      var res: array[64 + 8, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 72)
      for i in 0 ..< 64:
        check res[i] == uint32(i)
      # The flat-local values are position-derived, so they pin the x-major
      # order: blk (4,2) maps (tx, ty) -> ty*4 + tx, and each slot holds
      # tx + 10*ty for the (tx, ty) at that slot.
      let expectedFlat = [0'u32, 1, 2, 3, 10, 11, 12, 13]
      for i in 0 ..< 8:
        check res[64 + i] == expectedFlat[i]

when isMainModule:
  runTest()
