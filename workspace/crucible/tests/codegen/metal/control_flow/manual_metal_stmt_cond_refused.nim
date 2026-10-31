## Metal: a loop condition that lowers to statements refuses at compile time.
##
## Test that a `metal:` kernel whose loop condition lowers to statements fails
## to compile with the frontend refusal, conditions must be expressions.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/manual_metal_stmt_cond_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var i = 0'u32
    while (block:
      output[0] = output[0] + 1'u32
      i < 3'u32):
      i = i + 1'u32
    output[0] = i

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runMetal)
  engine.ingest(code)
  echo "UNREACHABLE: a statement-bearing condition must refuse at compile time"

when isMainModule:
  runTest()
