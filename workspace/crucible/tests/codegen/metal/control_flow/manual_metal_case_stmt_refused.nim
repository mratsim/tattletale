## Metal: a `case` statement refuses at compile time.
##
## Test that a `metal:` kernel with a `case` statement fails to compile
##   with the frontend refusal "case statement: outside the supported set"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/manual_metal_case_stmt_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var i: uint32 = 1
    case i
    of 0: output[0] = 10
    of 1: output[0] = 11
    else: output[0] = 12

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel with a `case` statement fails to compile with the frontend refusal 'case statement: outside the supported set'"

when isMainModule:
  runTest()
