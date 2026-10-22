## Vulkan: if/elif/else statement lowering, every branch reached on device.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/control_flow/t_vulkan_if_elif.nim

import std/[strutils, unittest]
import workspace/crucible

const elifGlsl = vulkan:
  proc elifDispatch(output, x: ptr UncheckedArray[int32]) {.global.} =
    if x[0] == 0:
      output[0] = 10
    elif x[0] == 1:
      output[0] = 20
    elif x[0] == 2:
      output[0] = 30
    else:
      output[0] = 40

const noElseGlsl = vulkan:
  proc elifNoElse(output, x: ptr UncheckedArray[int32]) {.global.} =
    output[0] = 99
    if x[0] == 0:
      output[0] = 1
    elif x[0] == 1:
      output[0] = 2

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo elifGlsl

  suite "Vulkan - if/elif/else statement lowering":

    test "elif chain reaches the device: all four branches distinct":
      var engine = bkVulkan.newEngine()
      engine.ingest(elifGlsl)
      var x = [0'i32]
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 1)
      engine.copyToDeviceFromHost(xDev, addr x[0], 1)
      var res: array[1, int32]
      const cases: array[4, (int32, int32)] = [(0'i32, 10'i32), (1'i32, 20'i32), (2'i32, 30'i32), (3'i32, 40'i32)]
      for (v, want) in cases:
        x[0] = v
        engine.copyToDeviceFromHost(xDev, addr x[0], 1)
        engine.launchKernel("elifDispatch", oDev, (xDev,))
        engine.copyToHostFromDevice(addr res[0], oDev, 1)
        check res[0] == want

    test "no else in source emits no else block":
      var engine = bkVulkan.newEngine()
      engine.ingest(noElseGlsl)
      var x = [0'i32]
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 1)
      engine.copyToDeviceFromHost(xDev, addr x[0], 1)
      var res: array[1, int32]
      const cases: array[2, (int32, int32)] = [(0'i32, 1'i32), (1'i32, 2'i32)]
      for (v, want) in cases:
        x[0] = v
        engine.copyToDeviceFromHost(xDev, addr x[0], 1)
        engine.launchKernel("elifNoElse", oDev, (xDev,))
        engine.copyToHostFromDevice(addr res[0], oDev, 1)
        check res[0] == want
      x[0] = 2
      engine.copyToDeviceFromHost(xDev, addr x[0], 1)
      engine.launchKernel("elifNoElse", oDev, (xDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 99  # no branch matched: sentinel preserved

when isMainModule:
  runTest()
