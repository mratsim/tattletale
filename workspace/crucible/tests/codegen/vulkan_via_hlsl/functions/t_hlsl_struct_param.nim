## HLSL: a struct-typed kernel parameter lowers to a push-constant member,
## compiled through DXC and validated by spirv-val.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/t_hlsl_struct_param.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

type Pair = object
  a: int32
  b: int32

const structHlsl = hlsl:
  proc k(output: ptr UncheckedArray[uint32], p: Pair) {.global.} =
    output[0] = uint32(p.a + p.b)

proc runTest() =
  let bindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("struct_param", structHlsl, bindings)
  echo "  OK - t_hlsl_struct_param"

when isMainModule:
  runTest()
