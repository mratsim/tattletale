## HLSL: whole-vector use of a coordinate builtin is rejected, components are read one by one.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/device/manual_hlsl_whole_vector_builtin_rejected.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    let g = thread_position_in_grid
    output[0] = uint32(g.x)

proc runTest() =
  echo code
  echo "UNREACHABLE — whole-vector coordinate builtin use must be rejected at compile time"

when isMainModule:
  runTest()
