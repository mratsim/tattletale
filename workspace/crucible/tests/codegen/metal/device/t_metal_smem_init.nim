## Metal: `{.smem.}` local with an initializer reads back the actual values.
##
## Test that the initializer lowers to explicit threadgroup stores.
## Metal drops the initializer form and honors runtime writes.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_smem_init.nim

import std/unittest
import workspace/crucible

const smemInitMsl = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[4, uint32] = [1'u32, 2, 3, 4]
    output[0] = scratch[0]
    output[1] = scratch[1]
    output[2] = scratch[2]
    output[3] = scratch[3]

proc runTest() =
  suite "Metal - {.smem.} initializer rewrite":
    test "threadgroup storage reads back the initialized values":
      var engine = newEngine(runMetal)
      engine.ingest(smemInitMsl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel<<(grid: (1, 1), blk: (4, 1))>>("k", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(res.toOpenArray(0, 4 - 1), oDev)
      check res == [1'u32, 2, 3, 4]

when isMainModule:
  runTest()
