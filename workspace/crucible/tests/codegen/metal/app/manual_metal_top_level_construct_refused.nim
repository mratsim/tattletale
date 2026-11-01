## Metal: a non-proc top-level statement refuses at compile time.
##
## Test that a `metal:` block holding a `call` statement beside the kernels
##   fails to compile with the frontend refusal "top-level construct not supported"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/app/manual_metal_top_level_construct_refused.nim

import workspace/crucible

const code = metal:
  echo("hi")
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 1

proc runTest() =
  echo "UNREACHABLE: a `metal:` block holding a `call` statement beside the kernels fails to compile with the frontend refusal 'top-level construct not supported'"

when isMainModule:
  runTest()
