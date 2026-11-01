## Builtin tuple-index `[idx]` normalization on `ivec3`
##
## Test that a tuple `[idx]` access normalizes to the field access
##
## - canonical coordinates are `ivec3 = tuple[x, y, z: int]`
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/functions/t_ir_builtin_tuple_index_edge.nim

import std/strutils
import workspace/crucible

block:
  # `[1]` normalizes to the field access and emits `.y`
  const okVulkan = glsl:
    proc ok(C: ptr UncheckedArray[int]) {.global.} =
      C[0] = gl_GlobalInvocationID[1]
  doAssert "gl_GlobalInvocationID.y" in okVulkan,
    "tuple [1] must normalize to the .y field access, got:\n" & okVulkan
  echo "  OK — int-literal tuple index [1] normalizes to .y"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All builtin tuple-index edge tests passed."
