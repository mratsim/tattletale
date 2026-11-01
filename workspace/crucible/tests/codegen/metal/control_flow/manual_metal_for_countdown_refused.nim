## Metal: a `countdown` loop refuses at compile time.
##
## Test that a `metal:` kernel looping with `countdown` fails to compile with the frontend
##   refusal "countdown loop: negative steps are outside the supported set"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/manual_metal_for_countdown_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var acc: uint32 = 0
    for i in countdown(3, 0):
      acc = acc + uint32(i)
    output[0] = acc

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel looping with `countdown` fails to compile with the frontend refusal 'countdown loop: negative steps are outside the supported set'"

when isMainModule:
  runTest()
