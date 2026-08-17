## Metal: shared memory + `syncthreads` barrier.
## A 64-thread threadgroup fills a `{.shared.}` scratch array, barriers, then
## reads it back reversed through `thread_position_in_threadgroup`.
## Runs through `engine.run()` and asserts byte-exact output.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_shared_memory.nim

import std/unittest
import workspace/crucible

const sharedMsl = metal:
  proc sharedKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.shared.}: array[64, uint32]
    scratch[thread_position_in_threadgroup.x] = thread_position_in_threadgroup.x
    syncthreads()
    output[thread_position_in_threadgroup.x] = scratch[63'u32 - thread_position_in_threadgroup.x]

proc runTest() =
  suite "Metal - shared memory + barrier":
    test "64 threads reverse-fill through threadgroup memory":
      var engine = bkMetal.init()
      engine.ingest(sharedMsl)
      var res: array[64, uint32]
      engine.run<<(grid: (64, 1), blk: (64, 1))>>("sharedKernel", res, ())
      for i in 0 ..< 64:
        check res[i] == uint32(63 - i)

when isMainModule:
  runTest()
