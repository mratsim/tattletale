## HLSL: component access on an unsupported coordinate builtin is rejected.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_unsupported_builtin_component_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    let n = int(threadgroups_per_grid.y)
    output[0] = uint32(n)

proc runTest() =
  echo code
  echo "UNREACHABLE — component access on unsupported coordinate builtins must be rejected at compile time"

when isMainModule:
  runTest()
