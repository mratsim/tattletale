## while-condition splice ordering
##
## Test that a splice in the while condition runs before every condition evaluation, including the first.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/red --nimcache:nimcache/red \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_while_cond_splice_order.nim

import std/unittest
import workspace/crucible

const whileSpliceOnceMsl = metal:
  proc whileSpliceOnce(o: ptr UncheckedArray[int32]) {.global.} =
    var i: int32 = 0
    var ran: int32 = 0
    while (block: ran = 1; i < 0):
      o[2] = 99
    o[0] = ran
    o[1] = i

const whileSplicePerTestMsl = metal:
  proc whileSplicePerTest(o: ptr UncheckedArray[int32]) {.global.} =
    var i: int32 = 0
    var count: int32 = 0
    while (block: i = i + 1; i < 3):
      count = count + 1
    o[0] = count
    o[1] = i

proc runTest() =
  suite "Metal - while condition splice ordering":

    test "splice runs before the first test even when the body never executes":
      var engine = newEngine(runMetal)
      engine.ingest(whileSpliceOnceMsl)
      let oDev = engine.allocDevice(int32, 3)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("whileSpliceOnce", oDev, ())
      var res: array[3, int32]
      engine.copyToHostFromDevice(res.toOpenArray(0, 3 - 1), oDev)
      check res[0] == 1  # splice ran once before the first (failing) test
      check res[1] == 0  # i untouched: the body never ran

    test "splice runs before every test, not only after the first admission":
      var engine = newEngine(runMetal)
      engine.ingest(whileSplicePerTestMsl)
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("whileSplicePerTest", oDev, ())
      var res: array[2, int32]
      engine.copyToHostFromDevice(res.toOpenArray(0, 2 - 1), oDev)
      # tests see i = 1, 2, 3: two admitted iterations
      check res[0] == 2
      check res[1] == 3

when isMainModule:
  runTest()
