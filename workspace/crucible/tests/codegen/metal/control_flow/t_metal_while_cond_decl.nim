## Metal: while condition declaring a local
##
## Test that a while condition declaring a local re-computes it before
## every test and terminates at the recomputed bound.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_while_cond_decl.nim

import std/unittest
import workspace/crucible

const declCondMsl = metal:
  proc declCondKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var i = 0'u32
    var tests = 0'u32
    while (block:
      var n = i
      tests = tests + 1'u32
      n < 3'u32):
      i = i + 1'u32
    output[0] = i
    output[1] = tests

proc runTest() =   # private: engines are destroyed at proc return
  suite "Metal - while condition declaring a local":
    test "a declared while condition reads its recomputed value per test":
      var engine = newEngine(runMetal)
      engine.ingest(declCondMsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel("declCondKernel", oDev, ())
      var output: array[2, uint32]
      engine.copyToHostFromDevice(output.toOpenArray(0, 2 - 1), oDev)
      # Nim re-runs the condition block before every test: tests = 3 admitted
      # iterations plus the failing test, i stops at the recomputed bound.
      check output[0] == 3
      check output[1] == 4

when isMainModule:
  runTest()
