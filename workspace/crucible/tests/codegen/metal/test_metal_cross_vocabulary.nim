## Metal: cross-vocabulary kernel, the CUDA idiom `blockIdx.x` + `syncthreads()` inside a `metal:` block.
## Emitted MSL carries `threadgroup_position_in_grid.x` and `threadgroup_barrier(mem_flags::mem_threadgroup)`,
## the run holds the cross-thread exchange through real barrier semantics.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_cross_vocabulary.nim

import std/[strutils, unittest]
import workspace/crucible

const crossVocabMsl = metal:
  proc crossVocabKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[64, uint32]
    let t = thread_position_in_threadgroup.x
    let base = blockIdx.x * 64
    scratch[t] = uint32(base + t)
    syncthreads()
    # Thread t reads the slot written by thread (63 - t), not its own.
    # Values are per-thread distinct, so a same-slot read fails the checks.
    output[base + t] = scratch[63 - t]

proc runTest() =
  suite "Metal - cross-vocabulary builtin aliases":

    test "CUDA idiom blockIdx + syncthreads lowers to canonical MSL and runs":
      var engine = bkMetal.init()
      engine.ingest(crossVocabMsl)
      let msl = engine.getArtifact()
      check "threadgroup_position_in_grid.x" in msl
      check "threadgroup_barrier(mem_flags::mem_threadgroup)" in msl
      let oDev = engine.allocDevice(128 * sizeof(uint32))
      engine.run<<(grid: (2, 1), blk: (64, 1))>>("crossVocabKernel", oDev, ())
      var res: array[128, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 128 * sizeof(uint32))
      engine.freeDevice(oDev)
      for i in 0 ..< 64:
        check res[i] == uint32(63 - i)
      for i in 64 ..< 128:
        check res[i] == uint32(64 + 63 - (i - 64))

when isMainModule:
  runTest()
