## Metal: ternary expression lowering. The DSL funnels `if cond: a else: b`
## in an expression slot through the ternary IR shape.
## Covered conditions:
## - a data-dependent condition
## - a compile-time constant condition
## - a block-expression branch
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_ternary.nim


import workspace/crucible

import std/unittest

const code = metal:
  proc ternaryKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = if a[0] > b[0]: a[0] else: b[0]
    output[1] = if true: 1'u32 else: 0'u32
    output[2] = if a[0] > 0'u32:
      block:
        let y = a[0] + 1'u32
        y * 2'u32
    else:
      7'u32

proc runTernary(a, b: uint32): array[3, uint32] =
  ## One launch per call: device buffers serve a single launch so handles never need a copy.
  var engine = newEngine(runMetal)
  engine.ingest(code)
  let aDev = engine.allocDevice(uint32, 1)
  let bDev = engine.allocDevice(uint32, 1)
  let oDev = engine.allocDevice(uint32, 3)
  engine.copyToDeviceFromHost(aDev, unsafeAddr a, 1)
  engine.copyToDeviceFromHost(bDev, unsafeAddr b, 1)
  engine.launchKernel("ternaryKernel", oDev, (aDev, bDev))
  engine.copyToHostFromDevice(addr result[0], oDev, 3)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - ternary lowering":
    test "data-dependent, constant, and block-expression ternaries":
      let sel = runTernary(7'u32, 3'u32)
      check sel[0] == 7
      check sel[1] == 1
      check sel[2] == 16
      let swap = runTernary(2'u32, 9'u32)
      check swap[0] == 9
      check swap[1] == 1
      check swap[2] == 6
      let zero = runTernary(0'u32, 9'u32)
      check zero[0] == 9
      check zero[1] == 1
      check zero[2] == 7

when isMainModule:
  runTest()
