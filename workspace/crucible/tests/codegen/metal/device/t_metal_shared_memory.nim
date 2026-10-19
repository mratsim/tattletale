## Metal: shared memory + `syncthreads` barrier. A 64-thread threadgroup
## fills a `{.smem.}` scratch array, barriers, then reads it back reversed
## through `thread_position_in_threadgroup`, the output is byte-exact.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_shared_memory.nim

import std/unittest
import workspace/crucible

const sharedMsl = metal:
  proc sharedKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[64, uint32]
    let t = thread_position_in_threadgroup.x
    scratch[t] = uint32(t)
    syncthreads()
    output[t] = scratch[63 - t]

proc runTest() =
  suite "Metal - shared memory + barrier":
    test "64 threads reverse-fill through threadgroup memory":
      var engine = bkMetal.init()
      engine.ingest(sharedMsl)
      let oDev = engine.allocDevice(uint32, 64)
      engine.launchKernel<<(grid: (64, 1), blk: (64, 1))>>("sharedKernel", oDev, ())
      var res: array[64, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 64)
      for i in 0 ..< 64:
        check res[i] == uint32(63 - i)

when isMainModule:
  runTest()
