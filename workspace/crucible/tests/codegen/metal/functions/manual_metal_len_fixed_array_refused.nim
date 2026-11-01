## Metal: `len` over a fixed-size array parameter refuses at compile time.
##
## Test that a `metal:` kernel calling `len` over an `array` parameter fails to compile
##   with the frontend refusal "magic call len (LengthArray) is outside the supported set"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_len_fixed_array_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32], arr: array[4, uint32]) {.global.} =
    output[0] = uint32(len(arr))

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel calling `len` over an `array` parameter fails to compile with the frontend refusal 'magic call len (LengthArray) is outside the supported set'"

when isMainModule:
  runTest()
