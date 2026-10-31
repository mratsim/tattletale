## Metal: if/elif/else statement lowering through the frontend, branch selection on the device.
##
## Test that an if/elif/else chain and a no-else if/elif chain reach the device and select the right branch.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_if_else.nim

import std/unittest
import workspace/crucible

const ifElseMsl = metal:
  proc ifElseDispatch(output, x: ptr UncheckedArray[int32]) {.global.} =
    if x[0] == 0:
      output[0] = 10
    elif x[0] == 1:
      output[0] = 20
    elif x[0] == 2:
      output[0] = 30
    else:
      output[0] = 40

const noElseMsl = metal:
  proc ifNoElse(output, x: ptr UncheckedArray[int32]) {.global.} =
    output[0] = 99
    if x[0] == 0:
      output[0] = 1
    elif x[0] == 1:
      output[0] = 2

proc runDispatch(x: int32): array[1, int32] =
  ## One launch per call: device buffers serve a single launch so handles never need a copy.
  var engine = newEngine(runMetal)
  engine.ingest(ifElseMsl)
  let xDev = engine.allocDevice(int32, 1)
  let oDev = engine.allocDevice(int32, 1)
  engine.copyToDeviceFromHost(xDev, unsafeAddr x, 1)
  engine.launchKernel("ifElseDispatch", oDev, (xDev,))
  engine.copyToHostFromDevice(addr result[0], oDev, 1)

proc runNoElse(x: int32): array[1, int32] =
  var engine = newEngine(runMetal)
  engine.ingest(noElseMsl)
  let xDev = engine.allocDevice(int32, 1)
  let oDev = engine.allocDevice(int32, 1)
  engine.copyToDeviceFromHost(xDev, unsafeAddr x, 1)
  engine.launchKernel("ifNoElse", oDev, (xDev,))
  engine.copyToHostFromDevice(addr result[0], oDev, 1)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - if/elif/else statement lowering":
    test "elif chain reaches the device: all four branches distinct":
      check runDispatch(0'i32)[0] == 10
      check runDispatch(1'i32)[0] == 20
      check runDispatch(2'i32)[0] == 30
      check runDispatch(3'i32)[0] == 40
    test "no-else chain keeps the sentinel when no branch takes":
      check runNoElse(0'i32)[0] == 1
      check runNoElse(1'i32)[0] == 2
      check runNoElse(5'i32)[0] == 99

when isMainModule:
  runTest()
