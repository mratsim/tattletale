## HLSL: fp16 conversion builtins are rejected, float16 types are outside the subset.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_fp16_call_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = cast[uint32](toFp16(1.0'f32))

proc runTest() =
  echo code
  echo "UNREACHABLE — fp16 conversion call must be rejected at compile time"

when isMainModule:
  runTest()
