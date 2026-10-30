## Metal: `break` and `continue` inside a `while` loop.
##
## Test that break exits the enclosing while loop and that continue
## re-checks the loop condition before the next iteration, on the device.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_while_break_continue.nim

import std/unittest
import workspace/crucible

const breakMsl = metal:
  proc breakKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var j: uint32 = 0
    while j < 8'u32:
      if j == 3'u32:
        break
      output[j] = j
      j = j + 1'u32

const continueMsl = metal:
  proc continueKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var j: uint32 = 0
    while j < 8'u32:
      j = j + 1'u32
      if j == 4'u32:
        continue
      output[j - 1'u32] = j

proc runTest() =
  suite "Metal - while loop break and continue":
    test "break leaves the while loop at the matching iteration":
      var engine = newEngine(runMetal)
      engine.ingest(breakMsl)
      let oDev = engine.allocDevice(uint32, 8)
      engine.launchKernel("breakKernel", oDev, ())
      var res: array[8, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 8)
      for i in 0 ..< 3:
        check res[i] == uint32(i)
      for i in 3 ..< 8:
        check res[i] == 0'u32

    test "continue re-enters the while condition, one iteration skipped":
      var engine = newEngine(runMetal)
      engine.ingest(continueMsl)
      let oDev = engine.allocDevice(uint32, 8)
      engine.launchKernel("continueKernel", oDev, ())
      var res: array[8, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 8)
      for i in 0 ..< 8:
        let expected = if i == 3: 0'u32 else: uint32(i + 1)
        check res[i] == expected

when isMainModule:
  runTest()
