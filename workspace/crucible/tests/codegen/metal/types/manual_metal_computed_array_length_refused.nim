## Metal: a computed array length refuses at compile time.
##
## Test that a `metal:` kernel declaring `array[int32(3), uint32]` fails to compile
##   with the frontend refusal "compile-time value expected, got node kind nnkCall"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/manual_metal_computed_array_length_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var arr: array[int32(3), uint32]
    arr[0] = 1
    output[0] = arr[0]

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel declaring `array[int32(3), uint32]` fails to compile with the frontend refusal 'compile-time value expected, got node kind nnkCall'"

when isMainModule:
  runTest()
