## Metal: cross-vocabulary coordinates + barrier kernel (GEMM-idiom slice).
##
## A single kernel written in foreign idioms runs on the Metal backend:
##   - CUDA idiom `blockIdx.x * blockDim.x + threadIdx.x` for the global id
##   - `syncthreads()` (CUDA barrier alias)
##   - `thread_index_in_threadgroup` -> the native MSL scalar attribute
##     `uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]`
##     (the printer's scalar branch: the five coordinate vectors are
##     `uint3` attribute params, the flat index is `uint`).
## The host compares the global flat id and the flat local index.
## The emitted MSL must contain the scalar attribute declaration.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/metal/test_metal_cross_vocabulary_flat.nim

import std/[strutils, unittest]
import workspace/crucible

const kernel1d = metal:
  proc crossVocabFlat(C: ptr UncheckedArray[uint32]) {.global.} =
    let gid = blockIdx.x * blockDim.x + threadIdx.x
    C[gid] = gid
    C[64'u32 + thread_index_in_threadgroup] = thread_index_in_threadgroup
    syncthreads()

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel1d

  suite "Metal - cross-vocabulary coordinates + barrier":

    test "CUDA idiom + native scalar flat index + barrier execute correctly":
      var engine = bkMetal.init()
      engine.ingest(kernel1d)
      let msl = engine.getArtifact()
      check "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]" in msl
      check "threadgroup_barrier(mem_flags::mem_threadgroup)" in msl
      var res: array[64 + 8, uint32]
      engine.run<<(grid: (8, 1), blk: (8, 1))>>("crossVocabFlat", res, ())
      for i in 0 ..< 64:
        check res[i] == uint32(i)
      for i in 0 ..< 8:
        check res[64 + i] == uint32(i)

when isMainModule:
  runTest()
