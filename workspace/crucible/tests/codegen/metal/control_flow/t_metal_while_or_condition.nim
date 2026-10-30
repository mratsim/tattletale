## Metal: `while` condition over a call-bearing `or` chain.
##
## Test that the loop guard re-checks both calls per iteration.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_while_or_condition.nim

import std/unittest
import workspace/crucible

const chainWhileMsl = metal:
  proc guardA(output: ptr UncheckedArray[uint32]): bool {.device.} =
    result = output[0] < 3'u32
  proc guardB(output: ptr UncheckedArray[uint32]): bool {.device.} =
    result = output[1] < 5'u32
  proc orWhileKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var i: uint32 = 0
    while guardA(output) or guardB(output):
      output[0] = output[0] + 1'u32
      output[1] = output[1] + 1'u32
      i = i + 1'u32
    output[2] = i

  proc andWhileKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var i: uint32 = 0
    while guardA(output) and i < 4'u32:
      output[0] = output[0] + 1'u32
      i = i + 1'u32
    output[2] = i

proc runTest() =
  suite "Metal - while condition over call-bearing boolean chains":
    test "or chain: the loop runs while either guard holds":
      var engine = newEngine(runMetal)
      engine.ingest(chainWhileMsl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel("orWhileKernel", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[2] == 5'u32
      check res[0] == 5'u32
      check res[1] == 5'u32

    test "and chain: the loop runs while both guards hold":
      var engine = newEngine(runMetal)
      engine.ingest(chainWhileMsl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel("andWhileKernel", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[2] == 3'u32
      check res[0] == 3'u32

when isMainModule:
  runTest()
