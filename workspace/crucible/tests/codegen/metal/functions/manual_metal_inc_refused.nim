## Metal: `inc` in statement position refuses at compile time.
##
## Test that a `metal:` kernel calling `inc` as a statement fails to compile
##   with the frontend refusal "magic call inc in statement position without a destination"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_inc_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var i: uint32 = 0
    inc(i)
    output[0] = i

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel calling `inc` as a statement fails to compile with the frontend refusal 'magic call inc in statement position without a destination'"

when isMainModule:
  runTest()
