## Metal: boolean `xor` in a condition with a call-bearing right operand.
##
## Test that `xor` evaluates the right operand's side effects and value
## before the comparison, on the device.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_bool_xor.nim

import std/unittest
import workspace/crucible

const xorMsl = metal:
  proc readFlag(output: ptr UncheckedArray[uint32]): bool {.device.} =
    output[1] = 7'u32
    result = output[0] == 1'u32
  proc xorKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 1'u32
    output[1] = 0'u32
    output[2] = 0'u32
    if false xor readFlag(output):
      output[2] = 1'u32

const xorSymMsl = metal:
  proc pick(output: ptr UncheckedArray[uint32]): bool {.device.} =
    result = output[0] == 1'u32
  proc xorSymKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 1'u32
    output[1] = 0'u32
    output[2] = 0'u32
    if true xor pick(output):
      output[1] = 1'u32
    if pick(output) xor false:
      output[2] = 1'u32

proc runTest() =
  suite "Metal - boolean xor lowering":
    test "xor evaluates the right operand and compares both sides":
      var engine = newEngine(runMetal)
      engine.ingest(xorMsl)
      let oDev = engine.allocDevice(uint32, 3)
      engine.launchKernel("xorKernel", oDev, ())
      var res: array[3, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 3)
      check res[1] == 7'u32  # readFlag ran, its store landed
      check res[2] == 1'u32  # false xor true is true

    test "xor over a symbol operand and a call operand both compare":
      var engine = newEngine(runMetal)
      engine.ingest(xorSymMsl)
      let oDev = engine.allocDevice(uint32, 3)
      engine.launchKernel("xorSymKernel", oDev, ())
      var res: array[3, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 3)
      check res[1] == 0'u32  # true xor true is false
      check res[2] == 1'u32  # true xor false is true

when isMainModule:
  runTest()
