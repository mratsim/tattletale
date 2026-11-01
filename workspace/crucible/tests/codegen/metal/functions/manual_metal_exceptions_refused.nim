## Metal: `try`/`except` refuses at compile time.
##
## Test that a `metal:` kernel with a `try` statement fails to compile with the frontend refusal "exceptions are outside the supported set"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_exceptions_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    try:
      output[0] = 1
    except:
      output[0] = 2

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel with a `try` statement fails to compile with the frontend refusal 'exceptions are outside the supported set'"

when isMainModule:
  runTest()
