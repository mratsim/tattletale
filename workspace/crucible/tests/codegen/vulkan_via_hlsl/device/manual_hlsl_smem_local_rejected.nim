## HLSL: {.smem.} locals are rejected, shared memory has no binding in the HLSL subset.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_smem_local_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[4, uint32]
    scratch[0] = output[0]
    output[1] = scratch[0]

proc runTest() =
  echo code
  echo "UNREACHABLE — {.smem.} locals must be rejected at compile time"

when isMainModule:
  runTest()
