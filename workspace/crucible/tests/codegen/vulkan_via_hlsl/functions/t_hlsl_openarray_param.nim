## HLSL: an openArray kernel parameter lowers to a StructuredBuffer read plus
## a push-constant length, compiled through DXC and validated by spirv-val.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/t_hlsl_openarray_param.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const openarrayHlsl = hlsl:
  proc k(output: ptr UncheckedArray[uint32], s: openArray[uint32]) {.global.} =
    output[0] = s[0]

proc runTest() =
  let bindings: seq[BindingExpectation] = @[("output", 0, false), ("s", 1, true)]
  discard compileCheck("openarray_param", openarrayHlsl, bindings)
  echo "  OK - t_hlsl_openarray_param"

when isMainModule:
  runTest()
