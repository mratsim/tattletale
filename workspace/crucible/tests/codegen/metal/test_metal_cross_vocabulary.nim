## Metal: cross-vocabulary kernel using the CUDA idiom
## (`blockIdx.x` + `syncthreads()`) inside a `metal:` kernel.
## The alias templates expand to canonical names during sem.
## The emitted MSL must contain `threadgroup_position_in_grid.x`
## and `threadgroup_barrier(mem_flags::mem_threadgroup)`.
## The kernel runs with real barrier semantics. 64 threads fill shared scratch,
## barrier, and each thread then reads a slot written by a different thread.
## Measured on this M4 Max: the run checks pass even with the barrier removed,
## so the emitted-MSL barrier assertion is the effective barrier check.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_cross_vocabulary.nim

import std/[strutils, unittest]
import workspace/crucible

const crossVocabMsl = metal:
  proc crossVocabKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.shared.}: array[64, uint32]
    scratch[thread_position_in_threadgroup.x] =
      blockIdx.x * 64'u32 + thread_position_in_threadgroup.x
    syncthreads()
    # Thread t reads the slot written by thread (63 - t), not its own.
    # Values are per-thread distinct, so a same-slot read fails the checks.
    output[blockIdx.x * 64'u32 + thread_position_in_threadgroup.x] =
      scratch[63'u32 - thread_position_in_threadgroup.x]

proc runTest() =
  suite "Metal - cross-vocabulary builtin aliases":

    test "CUDA idiom blockIdx + syncthreads lowers to canonical MSL and runs":
      var engine = bkMetal.init()
      engine.ingest(crossVocabMsl)
      let msl = engine.getArtifact()
      check "threadgroup_position_in_grid.x" in msl
      check "threadgroup_barrier(mem_flags::mem_threadgroup)" in msl
      var res: array[128, uint32]
      engine.run<<(grid: (2, 1), blk: (64, 1))>>("crossVocabKernel", res, ())
      for i in 0 ..< 64:
        check res[i] == uint32(63 - i)
      for i in 64 ..< 128:
        check res[i] == uint32(64 + 63 - (i - 64))

when isMainModule:
  runTest()
