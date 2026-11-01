## Metal: a string literal refuses at compile time.
##
## Test that a `metal:` kernel with a string literal in expression position fails
##   to compile with the frontend refusal "string literal: GPU kernels carry no strings"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_string_literal_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    discard "hi"
    output[0] = 0

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel with a string literal in expression position fails to compile with the frontend refusal 'string literal: GPU kernels carry no strings'"

when isMainModule:
  runTest()
