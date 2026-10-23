## HLSL: `var` (pass-by-reference) kernel parameters are rejected.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_kvar_param_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32], z: var uint32) {.global.} =
    output[0] = z

proc runTest() =
  echo code
  echo "UNREACHABLE — var (pass-by-reference) kernel params must be rejected at compile time"

when isMainModule:
  runTest()
