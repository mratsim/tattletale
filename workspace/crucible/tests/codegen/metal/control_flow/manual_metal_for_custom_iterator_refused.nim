## Metal: a `for` loop over a user iterator refuses at compile time.
##
## Test that a `metal:` kernel looping over a custom iterator fails to compile
##   with the frontend refusal "for loop over the iterator myIter is outside the supported set"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/manual_metal_for_custom_iterator_refused.nim

import workspace/crucible

iterator myIter(n: int32): int32 =
  var i: int32 = 0
  while i < n:
    yield i
    inc(i)

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var acc: uint32 = 0
    for i in myIter(3):
      acc = acc + uint32(i)
    output[0] = acc

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel looping over a custom iterator fails to compile with the frontend refusal 'for loop over the iterator myIter is outside the supported set'"

when isMainModule:
  runTest()
