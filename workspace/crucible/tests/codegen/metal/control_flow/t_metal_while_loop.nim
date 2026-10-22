## Metal: `while` loop emission. A single-thread kernel writes
## `output[i] = i` for `i in 0 ..< 4` via a `while` loop with a manually incremented counter.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_while_loop.nim

import std/unittest
import workspace/crucible

const whileMsl = metal:
  proc whileKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var i: uint32 = 0
    while i < 4'u32:
      output[i] = i
      i = i + 1'u32

proc runTest() =
  suite "Metal - while loop":
    test "output[i] = i for i in 0..<4":
      var engine = bkMetal.newEngine()
      engine.ingest(whileMsl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel("whileKernel", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      for i in 0 ..< 4:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
