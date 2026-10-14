## Metal: if/elif/else statement lowering, branch selection on the device.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_if_elif.nim

import std/[strutils, unittest]
import workspace/crucible

const elifMsl = metal:
  proc elifDispatch(output: ptr UncheckedArray[int32];
                    x: ptr UncheckedArray[int32]) {.global.} =
    if x[0] == 0:
      output[0] = 10
    elif x[0] == 1:
      output[0] = 20
    elif x[0] == 2:
      output[0] = 30
    else:
      output[0] = 40

const noElseMsl = metal:
  proc elifNoElse(output: ptr UncheckedArray[int32];
                  x: ptr UncheckedArray[int32]) {.global.} =
    output[0] = 99
    if x[0] == 0:
      output[0] = 1
    elif x[0] == 1:
      output[0] = 2

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo elifMsl

  suite "Metal - if/elif/else statement lowering":

    test "elif chain reaches the device: all four branches distinct":
      # Emitted shape: every branch condition present, in source order, as an `else if` chain.
      doAssert "if ((x[0] == 0)) {" in elifMsl
      doAssert "else if ((x[0] == 1)) {" in elifMsl
      doAssert "else if ((x[0] == 2)) {" in elifMsl
      doAssert elifMsl.find("else if ((x[0] == 1))") < elifMsl.find("else if ((x[0] == 2))")
      for v in ["output[0] = 10;", "output[0] = 20;", "output[0] = 30;", "output[0] = 40;"]:
        doAssert v in elifMsl
      var engine = bkMetal.init()
      engine.ingest(elifMsl)
      let xDev = engine.allocDevice(sizeof(int32))
      let oDev = engine.allocDevice(sizeof(int32))
      var res: array[1, int32]
      var x = [0'i32]
      engine.copyToDeviceFromHost(xDev, addr x[0], sizeof(int32))
      engine.run("elifDispatch", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(int32))
      check res[0] == 10
      x[0] = 1
      engine.copyToDeviceFromHost(xDev, addr x[0], sizeof(int32))
      engine.run("elifDispatch", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(int32))
      check res[0] == 20
      x[0] = 2
      engine.copyToDeviceFromHost(xDev, addr x[0], sizeof(int32))
      engine.run("elifDispatch", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(int32))
      check res[0] == 30
      x[0] = 3
      engine.copyToDeviceFromHost(xDev, addr x[0], sizeof(int32))
      engine.run("elifDispatch", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(int32))
      check res[0] == 40
      engine.freeDevice(xDev)
      engine.freeDevice(oDev)

    test "no else in source emits no else block":
      # The elif chain is emitted (`else if`), but no else-BLOCK: the source
      # has no `else`, so a spurious `else { }` must not appear.
      doAssert "else if ((x[0] == 1)) {" in noElseMsl
      doAssert "else {" notin noElseMsl
      var engine = bkMetal.init()
      engine.ingest(noElseMsl)
      let xDev = engine.allocDevice(sizeof(int32))
      let oDev = engine.allocDevice(sizeof(int32))
      var res: array[1, int32]
      var x = [0'i32]
      engine.copyToDeviceFromHost(xDev, addr x[0], sizeof(int32))
      engine.run("elifNoElse", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(int32))
      check res[0] == 1
      x[0] = 1
      engine.copyToDeviceFromHost(xDev, addr x[0], sizeof(int32))
      engine.run("elifNoElse", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(int32))
      check res[0] == 2
      x[0] = 2
      engine.copyToDeviceFromHost(xDev, addr x[0], sizeof(int32))
      engine.run("elifNoElse", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(int32))
      check res[0] == 99  # no branch matched: sentinel preserved
      engine.freeDevice(xDev)
      engine.freeDevice(oDev)

when isMainModule:
  runTest()
