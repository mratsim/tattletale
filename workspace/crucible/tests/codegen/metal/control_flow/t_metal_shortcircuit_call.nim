## Metal: short-circuit evaluation of calls under `and`/`or` in statement position.
##
## Test that a call operand of a statement-position boolean operator runs only
## when the operator's earlier operands do not already decide the result.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_shortcircuit_call.nim

import std/unittest
import workspace/crucible

const shortCircuitMsl = metal:
  proc bump(output: ptr UncheckedArray[uint32]): bool {.device.} =
    output[0] = output[0] + 1'u32
    result = true
  proc andKernel(output, cmd: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 0'u32
    let take = cmd[0] > 0'u32 and bump(output)
    if take:
      output[1] = 11'u32
    else:
      output[1] = 22'u32
  proc orKernel(output, cmd: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 0'u32
    let take = cmd[0] == 0'u32 or bump(output)
    if take:
      output[1] = 11'u32
    else:
      output[1] = 22'u32

proc runAnd(cmd: uint32): array[2, uint32] =
  ## One launch per call: device buffers serve a single launch so handles never need a copy.
  var engine = newEngine(runMetal)
  engine.ingest(shortCircuitMsl)
  let cDev = engine.allocDevice(uint32, 1)
  let oDev = engine.allocDevice(uint32, 2)
  engine.copyToDeviceFromHost(cDev, unsafeAddr cmd, 1)
  engine.launchKernel("andKernel", oDev, (cDev,))
  engine.copyToHostFromDevice(addr result[0], oDev, 2)

proc runOr(cmd: uint32): array[2, uint32] =
  var engine = newEngine(runMetal)
  engine.ingest(shortCircuitMsl)
  let cDev = engine.allocDevice(uint32, 1)
  let oDev = engine.allocDevice(uint32, 2)
  engine.copyToDeviceFromHost(cDev, unsafeAddr cmd, 1)
  engine.launchKernel("orKernel", oDev, (cDev,))
  engine.copyToHostFromDevice(addr result[0], oDev, 2)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - short-circuit of call operands under statement-position and/or":
    test "and skips the call operand when the left side is false":
      let skipped = runAnd(0'u32)
      check skipped[0] == 0
      check skipped[1] == 22
      let taken = runAnd(1'u32)
      check taken[0] == 1
      check taken[1] == 11

    test "or skips the call operand when the left side is true":
      let skipped = runOr(0'u32)
      check skipped[0] == 0
      check skipped[1] == 11
      let taken = runOr(1'u32)
      check taken[0] == 1
      check taken[1] == 11

when isMainModule:
  runTest()
