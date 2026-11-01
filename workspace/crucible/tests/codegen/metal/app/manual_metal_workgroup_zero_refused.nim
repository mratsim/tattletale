## Metal: a `workgroup` pragma with a zero dimension refuses at compile time.
##
## Test that a `metal:` kernel with `{.workgroup: (0, 1, 1).}` fails to compile
##   with the frontend refusal "workgroup dimension 1 is 0, only positive dimensions are supported"
##   (a negative dimension refuses with the same check)
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/app/manual_metal_workgroup_zero_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global, workgroup: (0, 1, 1).} =
    output[0] = uint32(1)

proc runTest() =
  echo "UNREACHABLE: a zero workgroup dimension fails to compile with the frontend refusal 'workgroup dimension 1 is 0, only positive dimensions are supported'"

when isMainModule:
  runTest()
