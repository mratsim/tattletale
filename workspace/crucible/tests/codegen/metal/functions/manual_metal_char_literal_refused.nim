## Metal: a char literal refuses at compile time.
##
## Test that a `metal:` kernel with a char literal in expression position fails
##   to compile with the frontend refusal "char literal: outside the supported set"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_char_literal_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    discard 'a'
    output[0] = 0

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel with a char literal in expression position fails to compile with the frontend refusal 'char literal: outside the supported set'"

when isMainModule:
  runTest()
