## Metal: two address-space pragmas on one local refuse at compile time.
##
## Test that a `metal:` kernel local annotated `{.smem, const_mem.}` fails
##   to compile with the frontend refusal "multiple address-space pragmas on one local"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/manual_metal_space_pragma_conflict_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var x {.smem, const_mem.}: uint32 = 3
    output[0] = x

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel local annotated `{.smem, const_mem.}` fails to compile with the frontend refusal 'multiple address-space pragmas on one local'"

when isMainModule:
  runTest()
