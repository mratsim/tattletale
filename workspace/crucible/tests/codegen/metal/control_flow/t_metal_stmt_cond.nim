## Metal: statement splices in loop and branch conditions evaluate in place.
##
## Test that a `block:` splice in a `while` condition fires once per iteration
## at the front of the loop body. A splice in an `if` condition fires once
## before the branch test.
##
## Nim runs a while-condition splice on every test, including the failing test
## that exits the loop. The released lowering never does.
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

proc runTest() =   # private: engines are destroyed at proc return
  suite "Metal - statement splices in conditions":
    test "a while-condition splice runs once per iteration, never on the failing test":
      var engine = newEngine(runMetal)
      engine.ingest(stmtCondMsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel("countKernel", oDev, ())
      var output: array[2, uint32]
      engine.copyToHostFromDevice(addr output[0], oDev, 2)
      # 3 condition-splice firings, one per iteration, none on the failing test
      # i counts the iterations in the loop body, so it also stops at 3
      check output[0] == 3
      check output[1] == 3

    test "a while-condition splice feeding the test stops the loop one splice short of Nim":
      var engine = newEngine(runMetal)
      engine.ingest(stmtCondMsl)
      let oDev = engine.allocDevice(uint32, 3)
      engine.launchKernel("feedKernel", oDev, ())
      var output: array[3, uint32]
      engine.copyToHostFromDevice(addr output[0], oDev, 3)
      # Boundary: Nim increments n on the failing test too,
      # n = 4 under Nim and n = 3 under the released host
      check output[0] == 3
      check output[1] == 3
      check output[2] == 3

    test "an if-condition splice runs once, before the branch test":
      var engine = newEngine(runMetal)
      engine.ingest(stmtCondMsl)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("ifKernel", oDev, ())
      var output: array[1, uint32]
      engine.copyToHostFromDevice(addr output[0], oDev, 1)
      # Splice data: the assignment fed the test,
      # and the then branch took it
      check output[0] == 11

when isMainModule:
  runTest()
