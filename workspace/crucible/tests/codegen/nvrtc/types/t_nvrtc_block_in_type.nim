## Minimal reproducer:
##   Error: Not implemented to determine type from node: Block
##
## The `cuda:` macro's `toGpuAst` converts `block:` template bodies with
## local temps to `gpuBlock(isExpr: true)`. When `makeCodeValid` → `getType`
## encounters such a block (via `gpuIndex(gpuDeref(gpuDot(gpuBlock, ...)))`),
## the type resolver hits `else: raiseAssert`.
##
## This matches the exact assertion from ceramic's sgemm_1.nim on NVRTC,
## without requiring ceramic. The trigger is a device function (called from
## the kernel) whose body uses a `()` operator template with `block:` body,
## accessing `.data[0]` on the inline result.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/nvrtc/repro_block_in_type.nim 2>&1
import std/strformat
import workspace/crucible

# ── Types (standalone — no ceramic dependency) ──────────────────────
type
  MyLayout = object
    m, n: int32

  MyView = object
    data: ptr UncheckedArray[float32]
    layout: MyLayout

# ── `()` operator: block: + local temps + result → gpuBlock(isExpr) ─
# Matches ceramic's TensorView.operator() which uses evalOnceAs
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

# ── Kernel calling the device function ──────────────────────────────
const kernel = cuda:
  proc reproKernel(
    output: ptr UncheckedArray[float32],
    input: ptr UncheckedArray[float32],
    M, N: int32,
  ) {.global.} =
    let v = MyView(data: input, layout: MyLayout(m: M, n: N))
    deviceFn(v)

when isMainModule:
  echo "Testing block-in-type on NVRTC..."
  # The assertion fires during the `cuda:` macro (Nim compile time).
  # If we reach runtime, the assertion is fixed.
  echo kernel
  echo "  OK — compiled"
