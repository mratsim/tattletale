## Metal: calling a lambda literal refuses at compile time.
##
## Test that a `metal:` kernel immediately invoking a lambda fails to compile
##   with the frontend refusal naming the callee "is not a proc symbol"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_lambda_call_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    let y = (proc(x: uint32): uint32 = x + 1)(5'u32)
    output[0] = y

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel immediately invoking a lambda fails to compile with the frontend refusal naming the callee 'is not a proc symbol'"

when isMainModule:
  runTest()
