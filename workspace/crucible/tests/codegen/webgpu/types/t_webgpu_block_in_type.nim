## Regression: gpuBlock(isExpr) in symbol/mutability helpers — WebGPU/WGSL target
##
## WGSL was never affected by the `getType` assertion (it has no `getType`),
## but this test confirms gpuBlock is handled by its helper functions.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/types/t_webgpu_block_in_type.nim

import workspace/crucible

# ── Types ──────────────────────────────────────────────────────────
type
  MySpan = object
    idx: int32
    len: int32

# ── `()` operator: block: + local temps + result → gpuBlock(isExpr) ─
template `()`(s: MySpan; a, b: int32): auto =
  block:
    let coord = (a, b)
    let offset = coord[0] * s.len + coord[1]
    var result: MySpan
    result.idx = s.idx + int32(offset)
    result.len = s.len
    result

# ── Device function: inline `()` + field access ────────────────────
proc deviceFn(span: MySpan) =
  discard span(0, 0).idx

# ── Kernel ─────────────────────────────────────────────────────────
const kernel = webgpu:
  proc reproKernel(
    output: ptr UncheckedArray[float32],
    input: ptr UncheckedArray[float32],
    M, N: int32,
  ) {.global.} =
    let s = MySpan(idx: M, len: N)
    deviceFn(s)

# ── Harness ────────────────────────────────────────────────────────
when isMainModule:
  echo "Testing block-in-type on WebGPU..."
  echo kernel
  echo "  OK — compiled"
