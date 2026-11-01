## Metal: a bare enum field symbol refuses at compile time.
##
## Test that a `metal:` kernel reading an enum field without `ord` fails to compile
##   with the frontend refusal "symbol cGreen (nskEnumField) is not a value"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/manual_metal_enum_field_refused.nim

import workspace/crucible

type Color = enum cRed, cGreen

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = uint32(cGreen)

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel reading an enum field without `ord` fails to compile with the frontend refusal 'symbol cGreen (nskEnumField) is not a value'"

when isMainModule:
  runTest()
