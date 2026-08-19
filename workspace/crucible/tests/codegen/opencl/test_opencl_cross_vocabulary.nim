## OpenCL: cross-vocabulary coordinate + barrier kernel (GEMM-idiom slice).
##
## A single kernel written in foreign idioms runs on the OpenCL backend:
##   - CUDA idiom `blockIdx.x * blockDim.x + threadIdx.x` for the global id
##   - `syncthreads()` (CUDA barrier alias)
##   - `thread_index_in_threadgroup` (canonical flat local index), lowered
##     to the x-major linearization of get_local_id / get_local_size,
##     since OpenCL has no native flat local index.
## The host compares both the global flat id and the flat local index.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_cross_vocabulary.nim

import std/[strutils, unittest]
import workspace/crucible

const kernel2d = opencl:
  proc crossVocab2d(C: ptr UncheckedArray[uint32]) {.global.} =
    # CUDA idiom in an OpenCL kernel: blockIdx/blockDim/threadIdx expand
    # to canonical names, which the printer lowers to get_group_id /
    # get_local_size / get_local_id.
    let gx = blockIdx.x * blockDim.x + threadIdx.x
    let gy = blockIdx.y * blockDim.y + threadIdx.y
    C[gy * 8'u32 + gx] = gy * 8'u32 + gx
    # canonical flat local index -> linearization formula
    C[64'u32 + thread_index_in_threadgroup] = thread_index_in_threadgroup
    syncthreads()

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel2d

  suite "OpenCL - cross-vocabulary coordinates + barrier":

    test "CUDA idiom + flat local + barrier execute with correct values":
      var engine = bkOpenCL.init()
      engine.ingest(kernel2d)
      var res: array[64 + 8, uint32]
      engine.run<<(grid: (2, 4), blk: (4, 2))>>("crossVocab2d", res, ())
      # global flat id = gy * 8 + gx for each of the 64 work-items
      for i in 0 ..< 64:
        check res[i] == uint32(i)
      # flat local index: every work-group writes its own linearized local
      # index to the same 8 slots, so each slot holds its own index
      for i in 0 ..< 8:
        check res[64 + i] == uint32(i)

when isMainModule:
  runTest()
