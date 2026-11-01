## Metal: an unexpected pragma on a local refuses at compile time.
##
## Test that a `metal:` kernel local annotated `{.global.}` fails to compile
##   with the frontend refusal "unexpected pragma global on a local declaration"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/manual_metal_unknown_pragma_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var x {.global.}: uint32 = 3
    output[0] = x

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel local annotated `{.global.}` fails to compile with the frontend refusal 'unexpected pragma global on a local declaration'"

when isMainModule:
  runTest()
