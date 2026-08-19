## Vulkan: cross-vocabulary coordinate + barrier kernel (GEMM-idiom slice).
##
## A single kernel written in foreign idioms runs on the Vulkan backend:
##   - CUDA idiom `blockIdx.x * blockDim.x + threadIdx.x` for the global id
##   - OpenCL idiom `get_global_id(d)`: must agree with the CUDA spelling
##   - `syncthreads()` (CUDA barrier alias) -> GLSL `barrier()`
##   - `thread_index_in_threadgroup` -> native `gl_LocalInvocationIndex`
## The host compares the global flat id and the flat local index.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_cross_vocabulary.nim

import std/[strutils, unittest]
import workspace/crucible

const kernel2d = vulkan:
  proc crossVocab2d(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    let gx = blockIdx.x * blockDim.x + threadIdx.x
    let gy = blockIdx.y * blockDim.y + threadIdx.y
    C[gy * 8'u32 + gx] = gy * 8'u32 + gx
    # OpenCL idiom in a Vulkan kernel: get_global_id(d) must agree with the CUDA-idiom global id above.
    if gx == get_global_id(0'u32) and gy == get_global_id(1'u32):
      # Position-derived flat-local value: with blk (4,2) the x-major flat
      # index maps (tx, ty) -> ty*4 + tx, so slot i holds tx + 10*ty for
      # the (tx, ty) at that slot. A y-major mapping permutes these values.
      C[64'u32 + thread_index_in_threadgroup] =
        thread_position_in_threadgroup.x + 10'u32 * thread_position_in_threadgroup.y
    else:
      C[64'u32 + thread_index_in_threadgroup] = 99'u32
    syncthreads()

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel2d

  suite "Vulkan - cross-vocabulary coordinates + barrier":

    test "CUDA + OpenCL idioms agree; flat local is native; barrier runs":
      var engine = bkVulkan.init()
      engine.ingest(kernel2d)
      # The generated GLSL source (the `vulkan:` macro output) must carry
      # the native flat-index spelling. The Vulkan engine's getArtifact
      # returns compiled SPIR-V instead of source text, so the source is
      # grepped directly.
      check "gl_LocalInvocationIndex" in kernel2d
      var res: array[64 + 8, uint32]
      engine.run<<(grid: (2, 4), blk: (4, 2))>>("crossVocab2d", res, ())
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
