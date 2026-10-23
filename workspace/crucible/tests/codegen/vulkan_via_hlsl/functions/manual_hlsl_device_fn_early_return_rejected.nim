## HLSL: an early return in a device fn is rejected, only a trailing return splices.
##
## A return inside the callee body would exit mid-splice, the statements after it would run unconditionally.
##
## Expected compile error:
##   HLSL: early return in device function `clamp_<mangle suffix>` cannot inline, only a trailing return splices
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/manual_hlsl_device_fn_early_return_rejected.nim

import workspace/crucible

proc clamp(x: uint32): uint32 {.device.} =
  if x > 100u32:
    return 100u32
  result = x

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = clamp(7u32)

proc runTest() =
  echo code
  echo "UNREACHABLE, an early return in a device fn must be rejected at compile time"

when isMainModule:
  runTest()
