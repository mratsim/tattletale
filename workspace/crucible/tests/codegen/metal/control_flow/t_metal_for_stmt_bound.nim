## Metal: for-loop stop bound containing statements
##
## Test that a `countup` stop bound with statements evaluates once
## before the loop runs, and the loop stops at that first bound.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_for_stmt_bound.nim

import std/unittest
import workspace/crucible

const forBoundMsl = metal:
  proc onceBoundKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var acc = 0'u32
    var evals = 0'u32
    var limit = 3'u32
    for i in countup(0'u32, (block:
        evals = evals + 1'u32
        limit)):
      acc = acc + 1'u32
    output[0] = acc
    output[1] = evals

proc runTest() =
  suite "Metal - for-loop stop bound containing statements":
    test "a stop bound with statements evaluates once":
      var engine = newEngine(runMetal)
      engine.ingest(forBoundMsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel("onceBoundKernel", oDev, ())
      var output: array[2, uint32]
      engine.copyToHostFromDevice(addr output[0], oDev, 2)
      # Nim evaluates a for bound once: countup is inclusive, 4 iterations,
      # one bound evaluation.
      check output[0] == 4
      check output[1] == 1

when isMainModule:
  runTest()
