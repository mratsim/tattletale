## Builtin tuple-index edge: `uvec3` tuple `[idx]` access.
##
## The canonical coordinate vectors are `uvec3 = tuple[x, y, z: uint32]`,
## so both `.x` and `[idx]` access work. `nim_to_gpu` normalizes a tuple
## `[idx]` to a field access and `doAssert`s that the index is an `nnkIntLit`
## (a plain int literal). A `0'u32` index is rejected even earlier, at Nim
## sem: the tuple `[]` operator takes an int literal, so the uint32-typed
## literal is a loud type mismatch before codegen ever runs. Both rejection
## paths are pinned here: the positive int-literal normalization and the loud
## `[0'u32]` rejection. The rejection is a compile error at Nim sem, never
## a silently wrong emission.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_builtin_tuple_index_edge.nim

import std/strutils
import workspace/crucible

block:
  # Positive: an int-literal index normalizes to the field access.
  # The GLSL-idiom `gl_GlobalInvocationID[1]` therefore emits `.y`.
  const okVulkan = vulkan:
    proc ok(C: ptr UncheckedArray[uint32]) {.global.} =
      C[0] = gl_GlobalInvocationID[1]
  doAssert "gl_GlobalInvocationID.y" in okVulkan,
    "tuple [1] must normalize to the .y field access, got:\n" & okVulkan
  echo "  OK — int-literal tuple index [1] normalizes to .y"

block:
  # Negative: `[0'u32]` must be rejected loudly: the tuple `[]` operator
  # takes an int literal, so a uint32-typed index is a Nim type mismatch.
  # The rejection is a compile error, never a silently wrong emission.
  static:
    doAssert not compiles(block:
      const bad = vulkan:
        proc bad(C: ptr UncheckedArray[uint32]) {.global.} =
          C[0] = gl_GlobalInvocationID[0'u32]
    )
  echo "  OK — [0'u32] tuple index rejected loudly (reject + document)"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All builtin tuple-index edge tests passed."
