## HLSL: 64-bit integer locals are rejected, int64_t is a parse error in the HLSL frontend and uint64_t pulls the SPIR-V Int64 capability.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_int64_local_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var t: int64 = 5
    output[0] = uint32(t)

proc runTest() =
  echo code
  echo "UNREACHABLE — int64 locals must be rejected at compile time"

when isMainModule:
  runTest()
