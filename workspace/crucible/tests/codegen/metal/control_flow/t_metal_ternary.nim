## Metal: ternary expression lowering. The DSL funnels `if cond: a else: b`
## in an expression slot through the ternary IR shape, emitted as the C-style
## ternary form `(cond ? a : b)`, covering a compile-time constant condition
## and a data-dependent condition.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_ternary.nim


import workspace/crucible

import std/unittest

const code = metal:
  proc ternaryKernel(output: ptr UncheckedArray[uint32];
                     a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = if a[0] > b[0]: a[0] else: b[0]
    output[1] = if true: 1'u32 else: 0'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo code

  suite "Metal - ternary lowering":
    test "data-dependent and constant ternaries":
      var engine = newEngine(runMetal)
      engine.ingest(code)
      var a = [7'u32]
      var b = [3'u32]
      let aDev = engine.allocDevice(uint32, 1)
      let bDev = engine.allocDevice(uint32, 1)
      let oDev = engine.allocDevice(uint32, 2)
      var res: array[2, uint32]
      engine.copyToDeviceFromHost(aDev, addr a[0], 1)
      engine.copyToDeviceFromHost(bDev, addr b[0], 1)
      engine.launchKernel("ternaryKernel", oDev, (aDev, bDev))
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 7
      check res[1] == 1
      a[0] = 2
      b[0] = 9
      engine.copyToDeviceFromHost(aDev, addr a[0], 1)
      engine.copyToDeviceFromHost(bDev, addr b[0], 1)
      engine.launchKernel("ternaryKernel", oDev, (aDev, bDev))
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 9
      check res[1] == 1

when isMainModule:
  runTest()
