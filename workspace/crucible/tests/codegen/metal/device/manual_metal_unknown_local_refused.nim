## Metal: reading a module-level variable from a kernel refuses at compile time.
##
## Test that a `metal:` kernel reading a variable declared outside the macro
##   body fails to compile with the frontend refusal "unknown local symbol gv"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/manual_metal_unknown_local_refused.nim

import workspace/crucible

var gv: uint32 = 7

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = gv

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel reading a variable declared outside the macro body fails to compile with the frontend refusal 'unknown local symbol gv'"

when isMainModule:
  runTest()
