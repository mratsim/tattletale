## HLSL: a recursive device function refuses at compile time.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/manual_hlsl_device_fn_recursion_refused.nim

import workspace/crucible

const code = hlsl:
  proc down(n: int32): int32 {.device.} =
    if n <= 0:
      0
    else:
      down(n - 1)
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = down(3).uint32

proc runTest() =
  echo "UNREACHABLE: the recursive device function must refuse at compile time"

when isMainModule:
  runTest()
