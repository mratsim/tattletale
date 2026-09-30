## Codesize ledger for the pad family, comparing the hand-emitted macros
## against the concat-composed recursion candidate over the real call sites.
##
## Every kernel compiles one call site to Metal Shading Language, one row per kernel.
## cgsReport renders cost of 1 call plus the marginal over the paired floor:
## - each candidate row pairs its current-form sibling, the Marginal column shows the candidate delta over that form
## - R-suffixed templates = the candidate form, pad to rank through concat plus make_layout recursion, no LayoutCT
## - unsuffixed names = the current src, candidate value parity lives in tests/test_layout_algebra.nim (pad candidate parity section)
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis


func padRightImpl(layout: Layout; n: static int): auto =
  when rank(layout) >= n: layout
  else: padRightImpl(make_layout(concat(layout.shape, 1), concat(layout.stride, 0)), n)

func padLeftImpl(layout: Layout; n: static int): auto =
  when rank(layout) >= n: layout
  else: padLeftImpl(make_layout(concat(1, layout.shape), concat(0, layout.stride)), n)

func padRightR(layout: Layout; rank: static int): auto = padRightImpl(layout, rank)
func padLeftR(layout: Layout; rank: static int): auto = padLeftImpl(layout, rank)

# ── the one-shot candidate, single make_layout with a locally built fill ──
# padRight/padLeft keep their role as the pad-to-rank primitives.
# This test measures whether the fill can be built without the stepwise
# intermediate constructions that pad the recursion route.

template repeatFill(v: Int, n: static int): auto =
  when n <= 0: ()
  else: concat(repeatFill(v, n - 1), v)

func padRightO(layout: Layout; n: static int): auto =
  when rank(layout) >= n: layout
  else: make_layout(concat(layout.shape, repeatFill(Int[1](), n - rank(layout))),
                    concat(layout.stride, repeatFill(Int[0](), n - rank(layout))))

func padLeftO(layout: Layout; n: static int): auto =
  when rank(layout) >= n: layout
  else: make_layout(concat(repeatFill(Int[1](), n - rank(layout)), layout.shape),
                    concat(repeatFill(Int[0](), n - rank(layout)), layout.stride))

# candidate consumers, same bodies as src with the candidate pads
template blockedProductR(blk, tlr): auto =
  const mxR = max(rank(typeof(blk)), rank(typeof(tlr)))
  let lp = logical_product(padRightR(blk, mxR), padRightR(tlr, mxR))
  zipDimensions(dimension(lp, 0), dimension(lp, 1))

template rakedProductR(blk, tlr): auto =
  const mxR = max(rank(typeof(blk)), rank(typeof(tlr)))
  let lp = logical_product(padRightR(blk, mxR), padRightR(tlr, mxR))
  zipDimensions(dimension(lp, 1), dimension(lp, 0))

template tileToShapeR(blk, ts): auto =
  const R = static(rank(ts))
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tss, ts)
    let padded_blk = padRightR(bk, R)
    let blk_shape = product_each(padded_blk.shape)
    let trg_flat = product_each(tss)
    let product_shape = zipDimensionsWith(trg_flat, blk_shape): ceil_div(it_a, it_b)
    let tiler = make_layout(product_shape, LayoutLeft)
    blockedProductR(padded_blk, tiler)

# ── pad alone ──

const padRightCurrentMsl = metal:
  proc padRightCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padRight(L, 4)
    C[0] = float32 toIntVal size(r)

const padRightRecMsl = metal:
  proc padRightRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padRightR(L, 4)
    C[0] = float32 toIntVal size(r)

const padLeftCurrentMsl = metal:
  proc padLeftCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padLeft(L, 4)
    C[0] = float32 toIntVal size(r)

const padLeftRecMsl = metal:
  proc padLeftRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padLeftR(L, 4)
    C[0] = float32 toIntVal size(r)

const padRightOneShotMsl = metal:
  proc padRightOneShotKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padRightO(L, 4)
    C[0] = float32 toIntVal size(r)

const padLeftOneShotMsl = metal:
  proc padLeftOneShotKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 4), (16, 1))
    let r = padLeftO(L, 4)
    C[0] = float32 toIntVal size(r)

# ── the consumers, one rank step of pad ──

const blockedCurrentMsl = metal:
  proc blockedCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = blocked_product(blk, tiler)
    C[0] = float32 toIntVal size(r)

const blockedRecMsl = metal:
  proc blockedRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = blockedProductR(blk, tiler)
    C[0] = float32 toIntVal size(r)

const rakedCurrentMsl = metal:
  proc rakedCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = raked_product(blk, tiler)
    C[0] = float32 toIntVal size(r)

const rakedRecMsl = metal:
  proc rakedRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = rakedProductR(blk, tiler)
    C[0] = float32 toIntVal size(r)

# ── tile_to_shape, the full consumer chain ──

const tileToShapeCurrentMsl = metal:
  proc tileToShapeCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = tile_to_shape(make_layout((2, 3), (1, 2)), (6, 12))
    C[0] = float32 toIntVal size(r)

const tileToShapeRecMsl = metal:
  proc tileToShapeRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = tileToShapeR(make_layout((2, 3), (1, 2)), (6, 12))
    C[0] = float32 toIntVal size(r)

# ── receipts ──

cgsReport("cgs_layout_pad", [
  cgsReceipt("padRightCurrentKernel", padRightCurrentMsl),
  cgsReceipt("padRightRecKernel", padRightRecMsl, padRightCurrentMsl.len),
  cgsReceipt("padRightOneShotKernel", padRightOneShotMsl, padRightCurrentMsl.len),
  cgsReceipt("padLeftCurrentKernel", padLeftCurrentMsl),
  cgsReceipt("padLeftRecKernel", padLeftRecMsl, padLeftCurrentMsl.len),
  cgsReceipt("padLeftOneShotKernel", padLeftOneShotMsl, padLeftCurrentMsl.len),
  cgsReceipt("blockedCurrentKernel", blockedCurrentMsl),
  cgsReceipt("blockedRecKernel", blockedRecMsl, blockedCurrentMsl.len),
  cgsReceipt("rakedCurrentKernel", rakedCurrentMsl),
  cgsReceipt("rakedRecKernel", rakedRecMsl, rakedCurrentMsl.len),
  cgsReceipt("tileToShapeCurrentKernel", tileToShapeCurrentMsl),
  cgsReceipt("tileToShapeRecKernel", tileToShapeRecMsl, tileToShapeCurrentMsl.len)])
