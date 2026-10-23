## HLSL: address-of is rejected, buffers are index-only resources.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/control_flow/manual_hlsl_addr_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    let p = addr output[0]
    output[1] = p[]

proc runTest() =
  echo code
  echo "UNREACHABLE — address-of must be rejected at compile time"

when isMainModule:
  runTest()
