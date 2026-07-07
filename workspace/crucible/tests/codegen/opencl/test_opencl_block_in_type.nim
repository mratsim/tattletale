# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
## Regression: gpuBlock(isExpr) in getType — OpenCL target
##
## Same pattern as CUDA test_nvrtc_block_in_type.nim — a device function
## using an inline `()` operator with `block:` body, then `.data[0]` on the
## result triggers getType's missing `of gpuBlock:` arm.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_block_in_type.nim

import workspace/crucible/src/codegen/cl

# ── Types ──────────────────────────────────────────────────────────
type
  MyLayout = object
    m, n: int32

  MyView = object
    data: ptr UncheckedArray[float32]
    layout: MyLayout

# ── `()` operator: block: + local temps + result → gpuBlock(isExpr) ─
template `()`(v: MyView; a, b: int32): auto =
  block:
    let coord = (a, b)
    let offset = coord[0] * v.layout.n + coord[1]
    var result: MyView
    result.data = cast[ptr UncheckedArray[float32]](
      cast[uint64](v.data) + cast[uint64](offset * 4))
    result.layout = MyLayout(m: v.layout.m - a, n: v.layout.n - b)
    result

# ── Device function: inline `()` + .data[0] triggers the assertion ──
proc deviceFn(view: MyView) =
  discard view(0, 0).data[0]

# ── Kernel ─────────────────────────────────────────────────────────
const kernel = opencl:
  proc reproKernel(
    output: ptr UncheckedArray[float32],
    input: ptr UncheckedArray[float32],
    M, N: int32,
  ) {.global.} =
    let v = MyView(data: input, layout: MyLayout(m: M, n: N))
    deviceFn(v)

# ── Harness ────────────────────────────────────────────────────────
when isMainModule:
  echo "Testing block-in-type on OpenCL..."
  echo kernel
  echo "  OK — compiled"
