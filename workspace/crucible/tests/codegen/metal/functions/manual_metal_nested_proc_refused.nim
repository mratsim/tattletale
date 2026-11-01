## Metal: a proc definition nested in a kernel refuses at compile time.
##
## Test that a `metal:` kernel defining an inner proc fails to compile with the frontend refusal "nested proc definition"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_nested_proc_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    proc inner(): uint32 = 5
    output[0] = inner()

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel defining an inner proc fails to compile with the frontend refusal 'nested proc definition'"

when isMainModule:
  runTest()
