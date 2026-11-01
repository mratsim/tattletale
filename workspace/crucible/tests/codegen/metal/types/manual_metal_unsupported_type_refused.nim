## Metal: a seq-typed kernel parameter refuses at compile time.
##
## Test that a `metal:` kernel taking a `seq` parameter fails to compile
##   with the frontend refusal "type not supported: seq[uint32] (ntySequence)"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/manual_metal_unsupported_type_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32], s: seq[uint32]) {.global.} =
    output[0] = 0

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel taking a `seq` parameter fails to compile with the frontend refusal 'type not supported: seq[uint32] (ntySequence)'"

when isMainModule:
  runTest()
