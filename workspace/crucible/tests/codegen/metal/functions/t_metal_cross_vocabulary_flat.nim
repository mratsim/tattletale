## Metal: cross-vocabulary coordinates + barrier kernel (GEMM-idiom slice).
## One kernel mixes the CUDA global-id idiom `blockIdx.x * blockDim.x + threadIdx.x`,
## the native scalar `thread_index_in_threadgroup`, and `syncthreads()`.
## The host asserts both the global flat id and the flat local index.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_cross_vocabulary_flat.nim

import std/[strutils, unittest]
import workspace/crucible

const kernel1d = metal:
  proc crossVocabFlat(C: ptr UncheckedArray[uint32]) {.global.} =
    let gid = blockIdx.x * blockDim.x + threadIdx.x
    C[gid] = uint32(gid)
    C[64 + thread_index_in_threadgroup] = uint32(thread_index_in_threadgroup)
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
      let oDev = engine.allocDevice(uint32, (64 + 8))
      engine.launchKernel<<(grid: (8, 1), blk: (8, 1))>>("crossVocabFlat", oDev, ())
      var res: array[64 + 8, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, (64 + 8))
      engine.freeDevice(oDev)
      for i in 0 ..< 64:
        check res[i] == uint32(i)
      for i in 0 ..< 8:
        check res[64 + i] == uint32(i)

when isMainModule:
  runTest()
