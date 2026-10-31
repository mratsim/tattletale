## OpenCL: if/elif/else statement lowering, all branches reach the device,
## a source without else emits no else block.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/control_flow/t_opencl_if_elif.nim

import std/unittest
import workspace/crucible

const elifCl = opencl:
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

const noElseCl = opencl:
  proc elifNoElse(output: ptr UncheckedArray[int32];
                  x: ptr UncheckedArray[int32]) {.global.} =
    output[0] = 99
    if x[0] == 0:
      output[0] = 1
    elif x[0] == 1:
      output[0] = 2

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo elifCl

  suite "OpenCL - if/elif/else statement lowering":

    test "elif chain reaches the device: all four branches distinct":
      var engine = newEngine(runOpenCL)
      engine.ingest(elifCl)
      var res: array[1, int32]
      var x = [0'i32]
      let xDev = engine.allocDevice(int32, x.len)
      let oDev = engine.allocDevice(int32, 1)
      engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
      engine.launchKernel("elifDispatch", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 10
      x[0] = 1
      engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
      engine.launchKernel("elifDispatch", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 20
      x[0] = 2
      engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
      engine.launchKernel("elifDispatch", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 30
      x[0] = 3
      engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
      engine.launchKernel("elifDispatch", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 40

    test "no else in source leaves the sentinel standing on no match":
      var engine = newEngine(runOpenCL)
      engine.ingest(noElseCl)
      var res: array[1, int32]
      var x = [0'i32]
      let xDev = engine.allocDevice(int32, x.len)
      let oDev = engine.allocDevice(int32, 1)
      engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
      engine.launchKernel("elifNoElse", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 1
      x[0] = 1
      engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
      engine.launchKernel("elifNoElse", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 2
      x[0] = 2
      engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
      engine.launchKernel("elifNoElse", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 99  # no branch matched: sentinel preserved

when isMainModule:
  runTest()
