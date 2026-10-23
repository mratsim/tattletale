## HLSL: plain struct-typed kernel parameters are rejected, they have no resource binding.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_struct_param_rejected.nim

import workspace/crucible

type
  Pair = object
    a: int32
    b: int32

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32], p: Pair) {.global.} =
    output[0] = uint32(p.a + p.b)

proc runTest() =
  echo code
  echo "UNREACHABLE — struct-typed kernel params must be rejected at compile time"

when isMainModule:
  runTest()
