## Metal: a case object kernel parameter refuses at compile time.
##
## Test that a `metal:` kernel taking a variant-record parameter fails to compile
##   with the frontend refusal "case object: variant records have no C representation"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/manual_metal_case_object_refused.nim

import workspace/crucible

type Shape = object
  kind: uint32
  case isRound: bool
  of true:
    radius: uint32
  of false:
    side: uint32

const code = metal:
  proc k(output: ptr UncheckedArray[uint32], s: Shape) {.global.} =
    output[0] = s.side

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel taking a variant-record parameter fails to compile with the frontend refusal 'case object: variant records have no C representation'"

when isMainModule:
  runTest()
