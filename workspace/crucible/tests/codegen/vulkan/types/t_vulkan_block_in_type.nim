## Regression: gpuBlock(isExpr) in getType, Vulkan target
##
## Same pattern as CUDA test_cuda_block_in_type.nim: a device function
## using an inline `()` operator with `block:` body, then `.data[0]` on the
## result triggers getType's missing `of gpuBlock:` arm.
##
## Note: Vulkan doesn't support raw pointer casts. The `()` operator uses
## simple integer arithmetic instead of pointer cast to avoid validation.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/vulkan/types/t_vulkan_block_in_type.nim

import workspace/crucible

# ── Types ──────────────────────────────────────────────────────────
type
  MySpan = object
    idx: int32    # index (not a pointer, Vulkan has no raw pointers)
    len: int32

# ── `()` operator: block: + local temps + result → gpuBlock(isExpr) ─
# No raw pointer casts needed for Vulkan compatibility
template `()`(s: MySpan; a, b: int32): auto =
  block:
    let coord = (a, b)
    let offset = coord[0] * s.len + coord[1]
    var result: MySpan
    result.idx = s.idx + int32(offset)
    result.len = s.len
    result

# ── Device function: inline `()` + field access triggers the assertion ──
proc deviceFn(span: MySpan) =
  discard span(0, 0).idx

# ── Kernel ─────────────────────────────────────────────────────────
const kernel = vulkan:
  proc reproKernel(
    output: ptr UncheckedArray[float32],
    input: ptr UncheckedArray[float32],
    M, N: int32,
  ) {.global.} =
    let s = MySpan(idx: M, len: N)
    deviceFn(s)

# ── Harness ────────────────────────────────────────────────────────
when isMainModule:
  echo "Testing block-in-type on Vulkan..."
  echo kernel
  echo "  OK — compiled"
