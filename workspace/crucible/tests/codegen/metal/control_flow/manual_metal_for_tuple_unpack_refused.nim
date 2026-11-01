## Metal: tuple unpacking in a `for` loop refuses at compile time.
##
## Test that a `metal:` kernel unpacking tuples in a `for` loop fails to compile with the frontend refusal "tuple unpacking in a for loop"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/manual_metal_for_tuple_unpack_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var acc: uint32 = 0
    for (x, y) in [(1'u32, 2'u32), (3'u32, 4'u32)]:
      acc = acc + x + y
    output[0] = acc

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel unpacking tuples in a `for` loop fails to compile with the frontend refusal 'tuple unpacking in a for loop'"

when isMainModule:
  runTest()
