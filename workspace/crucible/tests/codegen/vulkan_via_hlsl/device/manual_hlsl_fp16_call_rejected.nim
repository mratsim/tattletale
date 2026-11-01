## HLSL: the fp16/bf16/fp8 conversion builtins refuse, half-width types are outside the subset.
##
## Test that an `hlsl:` kernel calling `toFp16` fails to compile naming
## the printer refusal "toFp16 has no HLSL form, the fp16/bf16/fp8 conversion builtins are outside the HLSL subset".
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
  echo "UNREACHABLE: the fp16 conversion call must refuse at compile time"

when isMainModule:
  runTest()
