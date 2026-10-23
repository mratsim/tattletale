## HLSL: fixed-array kernel parameters are rejected, buffers pass as ptr UncheckedArray.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_array_param_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32], m: array[4, uint32]) {.global.} =
    output[0] = m[0]

proc runTest() =
  echo code
  echo "UNREACHABLE — fixed-array kernel parameter must be rejected at compile time"

when isMainModule:
  runTest()
