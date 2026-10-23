## HLSL: a device fn reading a name the kernel shadows is rejected.
##
## Test that a device fn reading a kernel-shadowed global name fails to compile:
## the name-based splice binds that reference to the kernel's local, not the global.
##
## Expected compile error:
##   HLSL: device function `readG_<mangle suffix>` references `gg` which the kernel declares
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/manual_hlsl_device_fn_shadow_capture_rejected.nim

import workspace/crucible

var gg: uint32 = 42'u32

proc readG(): uint32 {.device.} =
  result = gg

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var gg: uint32 = 0
    output[0] = gg + readG()

proc runTest() =
  echo code
  echo "UNREACHABLE, a device fn referencing a shadowed global must be rejected at compile time"

when isMainModule:
  runTest()
