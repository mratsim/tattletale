## Run:
##   nim ceramic_cgs runner=cgs_layout_pad dump=true
##   from the tattletale/ directory.
##

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis


func padRightImpl(layout: Layout; n: static int): auto =
  when rank(layout) >= n: layout
  else: padRightImpl(make_layout(concat(layout.shape, 1), concat(layout.stride, 0)), n)

func padRightR(layout: Layout; rank: static int): auto = padRightImpl(layout, rank)

# candidate consumers, same bodies as src with the candidate pads
template blockedProductR(blk, tlr): auto =
  const mxR = max(blk.rank(), tlr.rank())
  let lp = logical_product(padRightR(blk, mxR), padRightR(tlr, mxR))
  zipDimensions(dimension(lp, 0), dimension(lp, 1))

template rakedProductR(blk, tlr): auto =
  const mxR = max(blk.rank(), tlr.rank())
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
    let r = tile_to_shape(make_layout((2, 3), (1, 2)), (2, 3, 12))
    C[0] = float32 toIntVal size(r)

const tileToShapeRecMsl = metal:
  proc tileToShapeRecKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = tileToShapeR(make_layout((2, 3), (1, 2)), (6, 12))
    C[0] = float32 toIntVal size(r)

# ── receipts ──

cgsReport("cgs_layout_pad", [
  cgsReceipt("blockedCurrentKernel", blockedCurrentMsl),
  cgsReceipt("blockedRecKernel", blockedRecMsl, blockedCurrentMsl.len),
  cgsReceipt("rakedCurrentKernel", rakedCurrentMsl),
  cgsReceipt("rakedRecKernel", rakedRecMsl, rakedCurrentMsl.len),
  cgsReceipt("tileToShapeCurrentKernel", tileToShapeCurrentMsl),
  cgsReceipt("tileToShapeRecKernel", tileToShapeRecMsl, tileToShapeCurrentMsl.len)])
