## HLSL: 64-bit integer kernel parameters are rejected.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_int64_param_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32], z: int64) {.global.} =
    output[0] = uint32(z)

proc runTest() =
  echo code
  echo "UNREACHABLE — int64 kernel params must be rejected at compile time"

when isMainModule:
  runTest()
