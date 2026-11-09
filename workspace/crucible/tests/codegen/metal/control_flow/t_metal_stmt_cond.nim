## Metal: statement splices in loop and branch conditions
##
## Test that a `block:` splice in a `while` condition fires
## before every condition evaluation, the failing test included.
## An `if`-condition splice fires once before the branch test.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_stmt_cond.nim

import std/unittest
import workspace/crucible

const stmtCondMsl = metal:
  proc countKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var i = 0'u32
    while (block:
      output[0] = output[0] + 1'u32
      i < 3'u32):
      i = i + 1'u32
    output[1] = i

  proc feedKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var n = 0'u32
    var i = 0'u32
    while (block:
      n = n + 1'u32
      output[0] = output[0] + 1'u32
      n < 3'u32):
      i = i + 1'u32
    output[1] = n
    output[2] = i

  proc ifKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var i = 0'u32
    if (block:
      i = 2'u32
      i == 2'u32):
      output[0] = 11'u32
    else:
      output[0] = 22'u32

proc runTest() =
  suite "Metal - statement splices in conditions":
    test "a while-condition splice fires before every test, the failing one included":
      var engine = newEngine(runMetal)
      engine.ingest(stmtCondMsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel("countKernel", oDev, ())
      var output: array[2, uint32]
      engine.copyToHostFromDevice(output.toOpenArray(0, 2 - 1), oDev)
      # 4 condition-splice firings, one before each condition evaluation.
      # i counts the admitted iterations in the loop body, so it stops at 3
      check output[0] == 4
      check output[1] == 3

    test "a while-condition splice feeding the test matches the Nim firing count":
      var engine = newEngine(runMetal)
      engine.ingest(stmtCondMsl)
      let oDev = engine.allocDevice(uint32, 3)
      engine.launchKernel("feedKernel", oDev, ())
      var output: array[3, uint32]
      engine.copyToHostFromDevice(output.toOpenArray(0, 3 - 1), oDev)
      # Nim fires the splice before the failing test too: n = 3 at the failing test.
      # A pre-splice n = 0 admits no iteration on the first test.
      # i counts 2 admitted iterations
      check output[0] == 3
      check output[1] == 3
      check output[2] == 2

    test "an if-condition splice runs once, before the branch test":
      var engine = newEngine(runMetal)
      engine.ingest(stmtCondMsl)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("ifKernel", oDev, ())
      var output: array[1, uint32]
      engine.copyToHostFromDevice(output.toOpenArray(0, 1 - 1), oDev)
      # Splice data: the assignment fed the test,
      # and the then branch took it
      check output[0] == 11

when isMainModule:
  runTest()
