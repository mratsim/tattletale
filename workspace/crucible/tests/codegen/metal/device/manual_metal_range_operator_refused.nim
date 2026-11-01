## Metal: the `..` range operator as a value refuses at compile time.
##
## Test that a `metal:` kernel binding `a .. b` to a variable fails to compile
##   with the frontend refusal "operator .. is outside the supported set"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/manual_metal_range_operator_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    let a: uint32 = 1
    let b: uint32 = 3
    let r = a .. b
    output[0] = 0

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel binding `a .. b` to a variable fails to compile with the frontend refusal 'operator .. is outside the supported set'"

when isMainModule:
  runTest()
