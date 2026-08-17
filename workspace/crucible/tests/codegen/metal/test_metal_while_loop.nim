## Metal: `while` loop emission.
## A single-thread kernel writes `output[i] = i` for `i in 0 ..< 4` via a
## `while` loop with a manually incremented counter. Runs through
## `engine.run()` and asserts the output byte-exact.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_while_loop.nim

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
      var engine = bkMetal.init()
      engine.ingest(whileMsl)
      var res: array[4, uint32]
      engine.run("whileKernel", res, ())
      for i in 0 ..< 4:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
