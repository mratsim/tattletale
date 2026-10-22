## Builtin tuple-index `[idx]` normalization on `ivec3`
##
## Test that a tuple `[idx]` access normalizes to the field access and a uint32 index is a compile error
##
## - canonical coordinates are `ivec3 = tuple[x, y, z: int]`
## - the index must be an int literal, `[1]` emits the `.y` field
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/functions/t_ir_builtin_tuple_index_edge.nim

import std/strutils
import workspace/crucible

block:
  # `[1]` normalizes to the field access and emits `.y`
  const okVulkan = vulkan:
    proc ok(C: ptr UncheckedArray[int]) {.global.} =
      C[0] = gl_GlobalInvocationID[1]
  doAssert "gl_GlobalInvocationID.y" in okVulkan,
    "tuple [1] must normalize to the .y field access, got:\n" & okVulkan
  echo "  OK — int-literal tuple index [1] normalizes to .y"

block:
  # a uint32 index is a compile error
  static:
    doAssert not compiles(block:
      const bad = vulkan:
        proc bad(C: ptr UncheckedArray[int]) {.global.} =
          C[0] = gl_GlobalInvocationID[0'u32]
    )
  echo "  OK — [0'u32] tuple index rejected loudly (reject + document)"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All builtin tuple-index edge tests passed."
