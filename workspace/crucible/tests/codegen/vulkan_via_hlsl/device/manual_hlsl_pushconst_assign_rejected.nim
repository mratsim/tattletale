## HLSL: assignment to a scalar (push-constant) kernel parameter is rejected, push constants are read-only inputs.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_pushconst_assign_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32], n: int32) {.global.} =
    n = n + 1
    output[0] = uint32(n)

proc runTest() =
  echo code
  echo "UNREACHABLE — assignment to push-constant params must be rejected at compile time"

when isMainModule:
  runTest()
