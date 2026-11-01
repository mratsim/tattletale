## Metal: `toOpenArray` over a fixed-size array parameter refuses at compile time.
##
## Test that a `metal:` kernel building a span over an `array` parameter fails to compile with the frontend
##   refusal "toOpenArray over a ctArray is outside the supported set, pass a pointer to the storage"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_toopenarray_array_param_refused.nim

import workspace/crucible

proc use(s: openArray[uint32]) =
  discard

const code = metal:
  proc k(output: ptr UncheckedArray[uint32], arr: array[4, uint32]) {.global.} =
    use(toOpenArray(arr, 0, 3))
    output[0] = 0

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel building a span over an `array` parameter fails to compile with the frontend refusal 'toOpenArray over a ctArray is outside the supported set, pass a pointer to the storage'"

when isMainModule:
  runTest()
