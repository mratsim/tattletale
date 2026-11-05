## and/or short-circuit rhs evaluation
##
## Test that the rhs of a false `and` and the rhs of a true `or` never evaluate.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/red --nimcache:nimcache/red \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_and_or_short_circuit.nim

import std/unittest
import workspace/crucible

const andShortCircuitMsl = metal:
  proc andShortCircuit(o, x: ptr UncheckedArray[int32]) {.global.} =
    var ran: int32 = 0
    let r = (x[0] == 0) and (block: ran = 1; true)
    o[0] = ran
    if r:
      o[1] = 1
    else:
      o[1] = 0

const orShortCircuitMsl = metal:
  proc orShortCircuit(o, x: ptr UncheckedArray[int32]) {.global.} =
    var ran: int32 = 0
    let r = (x[0] == 1) or (block: ran = 1; false)
    o[0] = ran
    if r:
      o[1] = 1
    else:
      o[1] = 0

proc runTest() =
  suite "Metal - and/or short-circuit rhs evaluation":

    test "false and skips the rhs side effect":
      var engine = newEngine(runMetal)
      engine.ingest(andShortCircuitMsl)
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 2)
      var res: array[2, int32]
      var x = [1'i32]
      engine.copyToDeviceFromHost(xDev, addr x[0], 1)
      engine.launchKernel("andShortCircuit", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 0  # rhs never ran
      check res[1] == 0  # r is false

    test "true and evaluates the rhs side effect":
      var engine = newEngine(runMetal)
      engine.ingest(andShortCircuitMsl)
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 2)
      var res: array[2, int32]
      var x = [0'i32]
      engine.copyToDeviceFromHost(xDev, addr x[0], 1)
      engine.launchKernel("andShortCircuit", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 1  # rhs ran
      check res[1] == 1  # r is true

    test "true or skips the rhs side effect":
      var engine = newEngine(runMetal)
      engine.ingest(orShortCircuitMsl)
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 2)
      var res: array[2, int32]
      var x = [1'i32]
      engine.copyToDeviceFromHost(xDev, addr x[0], 1)
      engine.launchKernel("orShortCircuit", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 0  # rhs never ran
      check res[1] == 1  # r is true

    test "false or evaluates the rhs side effect":
      var engine = newEngine(runMetal)
      engine.ingest(orShortCircuitMsl)
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 2)
      var res: array[2, int32]
      var x = [0'i32]
      engine.copyToDeviceFromHost(xDev, addr x[0], 1)
      engine.launchKernel("orShortCircuit", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 1  # rhs ran
      check res[1] == 0  # r is false

when isMainModule:
  runTest()
