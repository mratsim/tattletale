## HLSL: openArray kernel parameters are rejected, they have no HLSL type name.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_openarray_param_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32], s: openArray[uint32]) {.global.} =
    output[0] = s[0]

proc runTest() =
  echo code
  echo "UNREACHABLE — openArray kernel params must be rejected at compile time"

when isMainModule:
  runTest()
