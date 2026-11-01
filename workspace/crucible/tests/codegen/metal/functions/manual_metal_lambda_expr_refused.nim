## Metal: a lambda literal as a value refuses at compile time.
##
## Test that a `metal:` kernel binding a lambda to a variable fails to compile
##   with the frontend refusal "expression construct not supported: nnkLambda"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_lambda_expr_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    let f = proc(x: uint32): uint32 = x + 1
    output[0] = 0

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel binding a lambda to a variable fails to compile with the frontend refusal 'expression construct not supported: nnkLambda'"

when isMainModule:
  runTest()
