## Metal: ternary expression lowering. The DSL funnels `if cond: a else: b`
## in an expression slot through the ternary IR shape.
## Covered conditions:
## - a data-dependent condition
## - a compile-time constant condition
## - a block-expression branch
## - a call in a branch value runs only when the branch takes
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

const branchCallMsl = metal:
  proc bump(output: ptr UncheckedArray[uint32]): uint32 {.device.} =
    output[0] = output[0] + 1'u32
    result = output[0]
  proc ternCallKernel(output, cmd: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 0'u32
    output[1] = if cmd[0] == 1'u32: bump(output) else: 7'u32

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

proc runTernCall(cmd: uint32): array[2, uint32] =
  ## One launch per call: device buffers serve a single launch so handles never need a copy.
  var engine = newEngine(runMetal)
  engine.ingest(branchCallMsl)
  let cDev = engine.allocDevice(uint32, 1)
  let oDev = engine.allocDevice(uint32, 2)
  engine.copyToDeviceFromHost(cDev, unsafeAddr cmd, 1)
  engine.launchKernel("ternCallKernel", oDev, (cDev,))
  engine.copyToHostFromDevice(addr result[0], oDev, 2)

proc runTest() =
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

    test "branch call runs only when the branch takes":
      # bump writes output[0] through the pointer and returns it.
      # Eager lowering runs it on both branches.
      # Branch-local evaluation runs it only when the branch takes.
      let taken = runTernCall(1'u32)
      check taken[0] == 1
      check taken[1] == 1
      let skipped = runTernCall(0'u32)
      check skipped[0] == 0
      check skipped[1] == 7

when isMainModule:
  runTest()
