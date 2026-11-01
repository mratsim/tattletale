## Metal: calling a void lambda in statement position refuses at compile time.
##
## Test that a `metal:` kernel invoking a void lambda as a statement fails
##   to compile with the frontend refusal naming the callee "is not a proc symbol"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_lambda_stmt_call_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    (proc() = discard)()
    output[0] = 1

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel invoking a void lambda as a statement fails to compile with the frontend refusal naming the callee 'is not a proc symbol'"

when isMainModule:
  runTest()
