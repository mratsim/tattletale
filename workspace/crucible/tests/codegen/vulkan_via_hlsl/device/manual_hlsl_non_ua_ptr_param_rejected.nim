## HLSL: kernel parameters of `ptr T` other than `ptr UncheckedArray[T]` are rejected.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_non_ua_ptr_param_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32], p: ptr uint32) {.global.} =
    output[0] = 1'u32

proc runTest() =
  echo code
  echo "UNREACHABLE — ptr-to-scalar kernel params must be rejected at compile time"

when isMainModule:
  runTest()
