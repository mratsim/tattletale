## HLSL: void pointer kernel parameters are rejected.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_voidptr_param_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32], p: pointer) {.global.} =
    output[0] = 1'u32

proc runTest() =
  echo code
  echo "UNREACHABLE — void pointer kernel params must be rejected at compile time"

when isMainModule:
  runTest()
