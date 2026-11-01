## Metal: a `countup` loop with a negative step refuses at compile time.
##
## Test that a `metal:` kernel with a negative `countup` step fails to compile
##   with the frontend refusal "for loop step -2: only positive steps are supported"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/manual_metal_for_negative_step_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var acc: uint32 = 0
    for i in countup(0, 10, -2):
      acc = acc + uint32(i)
    output[0] = acc

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel with a negative `countup` step fails to compile with the frontend refusal 'for loop step -2: only positive steps are supported'"

when isMainModule:
  runTest()
