## Metal: object constructor fields bind by name, not by written order.
##
## Test that an object constructed with out-of-order named fields
## binds each value to its named field.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_objconstr_field_order.nim

import std/unittest
import workspace/crucible

type Pair = object
  x: int32
  y: int32

const ctorOrderMsl = metal:
  proc ctorOrderKernel(o: ptr UncheckedArray[int32]) {.global.} =
    let p = Pair(y: 2, x: 1)
    o[0] = p.x
    o[1] = p.y

proc runTest() =
  suite "Metal - object constructor field order":
    test "out-of-order named fields bind by name, x keeps 1 and y keeps 2":
      var engine = newEngine(runMetal)
      engine.ingest(ctorOrderMsl)
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("ctorOrderKernel", oDev, ())
      var res: array[2, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 1
      check res[1] == 2

when isMainModule:
  runTest()
