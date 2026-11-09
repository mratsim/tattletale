## loop-body hoist position
##
## Test that a hoisted statement runs at its source statement, never at the front or the end of the body.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/red --nimcache:nimcache/red \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_body_hoist_position.nim

import std/unittest
import workspace/crucible

const bodyHoistPosMsl = metal:
  proc bodyHoistPos(o: ptr UncheckedArray[int32]) {.global.} =
    var tick: int32 = 0
    for i in 0 ..< 1:
      o[1] = tick
      tick = tick + 1
      o[3] = (block: o[3] = tick; o[4] = tick; 2)
      tick = tick + 10
    o[0] = tick
    o[2] = o[3]

const bodyHoistIfExprMsl = metal:
  proc bodyHoistIfExpr(o, x: ptr UncheckedArray[int32]) {.global.} =
    var tick: int32 = 0
    for i in 0 ..< 1:
      o[1] = tick
      tick = tick + 1
      o[3] = (if x[0] == 0:
                o[4] = tick
                2'i32
              else:
                3'i32)
      tick = tick + 10
    o[0] = tick

const bodyHoistCondSpliceMsl = metal:
  proc bodyHoistCondSplice(o, x: ptr UncheckedArray[int32]) {.global.} =
    var tick: int32 = 0
    for i in 0 ..< 1:
      o[1] = tick
      tick = tick + 1
      if (block: o[4] = tick; x[0] == 0):
        o[3] = 2
      tick = tick + 10
    o[0] = tick

proc runTest() =
  suite "Metal - body hoist position":

    test "block-expr hoist runs between its predecessor and its own assignment":
      var engine = newEngine(runMetal)
      engine.ingest(bodyHoistPosMsl)
      let oDev = engine.allocDevice(int32, 5)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("bodyHoistPos", oDev, ())
      var res: array[5, int32]
      engine.copyToHostFromDevice(res.toOpenArray(0, 5 - 1), oDev)
      check res[1] == 0   # statement 1 saw tick before any increment
      check res[4] == 1   # hoist saw tick = 1: after stmt 2, before stmt 4
      check res[3] == 2   # hoisted code ran before its own statement's assignment
      check res[0] == 11  # both increments ran exactly once
      check res[2] == 2

    test "if-expression hoists inside the body run at their statement":
      var engine = newEngine(runMetal)
      engine.ingest(bodyHoistIfExprMsl)
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 5)
      var res: array[5, int32]
      var x = [0'i32]
      engine.copyToDeviceFromHost(xDev, x.toOpenArray(0, 1 - 1))
      engine.launchKernel("bodyHoistIfExpr", oDev, (xDev,))
      engine.copyToHostFromDevice(res.toOpenArray(0, 5 - 1), oDev)
      check res[1] == 0   # statement 1 saw tick before any increment
      check res[4] == 1   # branch statement's effect saw tick = 1, not 11
      check res[3] == 2
      check res[0] == 11

    test "spliced if-condition hoist inside the body runs at its statement":
      var engine = newEngine(runMetal)
      engine.ingest(bodyHoistCondSpliceMsl)
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 5)
      var res: array[5, int32]
      var x = [0'i32]
      engine.copyToDeviceFromHost(xDev, x.toOpenArray(0, 1 - 1))
      engine.launchKernel("bodyHoistCondSplice", oDev, (xDev,))
      engine.copyToHostFromDevice(res.toOpenArray(0, 5 - 1), oDev)
      check res[1] == 0   # statement 1 saw tick before any increment
      check res[4] == 1   # condition splice saw tick = 1, not 11
      check res[3] == 2
      check res[0] == 11

when isMainModule:
  runTest()
