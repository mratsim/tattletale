# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensor RMSNorm as a device function over row strips.
##
## - `rmsnorm(dst, src, eps, axis)`: x * rsqrt(mean(x²) + eps) over the input axis
## - `rmsnorm(dst, src, weight, eps, axis)`: the same, scaled by the per-axis weight

import workspace/crucible
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/tensors

proc rmsnormStridedImpl(dst, src, weight: distinct TensorView, eps: float32, axis: static int, withWeight: static bool) {.inline.} =
  ## Strided-axis RMSNorm body, the lane mode spans the rows.
  ##
  ## One lane per row over 32-row blocks walked grid-stride:
  ## - lane l owns row l outright, its running sum of squares is the row total
  ## - pass 1 needs no cross-lane combine, pass 2 stores directly through
  ##   the row strips' own layouts
  ## - rows off the tensor's end stay idle
  ##
  ## Loads and stores coalesce when the non-reduced part's compact order is
  ## memory-consecutive (non-reduced stride 1, the column-major case).
  ## Correct but strided otherwise.
  const depthOwn = 8  # sums in flight per lane over the serial row walk
  let
    W = getLaneWidth()
    thrID = getLocalThreadID()
    nonReduced = src.sliceAxes(axis)
    reduced = src.diceAxes(axis)
    gridStride = getGridSize()
    rowSize = src.diceAxes(axis).size()
    rowBlock0 = getLocalTileID(src.shape) * W
  const nr: int = nonReduced.rank()
  when nr == 0:
    # the rank-0 slice's size is a static Int countup rejects, the empty map
    # constructors do not lower through crucible
    let rows = 1
  else:
    let rows = nonReduced.size()
    const identityOrder = block:
      var r: array[nr, int]
      staticFor i, 0, nr:
        r[i] = i
      r
    let
      rowMap = make_ordered_layout(nonReduced.shape, identityOrder)
      dstNonReduced = dst.sliceAxes(axis)
      dstReduced = dst.diceAxes(axis)
  for blk0 in countup(rowBlock0, rows - 1, gridStride * W):
    let
      rowFlat = blk0 + thrID
      rowActive = rowFlat < rows
      fullEnd = rowSize.round_multiple_down(depthOwn)
    when nr == 0:
      # crd2idx has no rank-0 identity (layout_indexing.nim:128), the whole tensor
      # is the row strip
      let
        srcRow = src
        dstRow = dst
    else:
      let
        rowCrd = idx2crd(rowMap, rowFlat)
        srcRow = make_view(src.data +% crd2idx(nonReduced, rowCrd), reduced)
        dstRow = make_view(dst.data +% crd2idx(dstNonReduced, rowCrd), dstReduced)
    var acc = zero(float32)
    if rowActive:
      var accs: array[depthOwn, float32]
      for j in 0 ..< depthOwn:
        accs[j] = zero(float32)
      for c0 in countup(0, fullEnd - 1, depthOwn):
        for j in 0 ..< depthOwn:
          accs[j] += srcRow[c0 + j].square()
      for c in fullEnd ..< rowSize:
        accs[0] += srcRow[c].square()
      for j in 0 ..< depthOwn:
        acc += accs[j]
    let scale = rsqrt(acc / float32(rowSize) + eps)
    when withWeight:
      let weightStrip = atAxisIndex(weight, 0, 0)
      for c0 in countup(0, fullEnd - 1, depthOwn):
        for j in 0 ..< depthOwn:
          if rowActive:
            dstRow[c0 + j] = srcRow[c0 + j] * scale * weightStrip[c0 + j]
      for c in fullEnd ..< rowSize:
        if rowActive:
          dstRow[c] = srcRow[c] * scale * weightStrip[c]
    else:
      for c0 in countup(0, fullEnd - 1, depthOwn):
        for j in 0 ..< depthOwn:
          if rowActive:
            dstRow[c0 + j] = srcRow[c0 + j] * scale
      for c in fullEnd ..< rowSize:
        if rowActive:
          dstRow[c] = srcRow[c] * scale

proc rmsnormImpl(dst, src, weight: distinct TensorView, eps: float32, axis: static int, withWeight: static bool) {.inline.} =
  ## Row-wise RMS normalization shared by the two entry points.
  ##
  ## One warp per row, a grid-stride loop covers the row list from any launch grid size.
  ## Each output row is x * rsqrt(mean(x²) + eps), times the per-axis weight when withWeight. dst is fully overwritten.
  ##
  ## Sum of squares per row, one register tile at a time:
  ## - `atAxisIndex` cuts the source and destination strips per row, the rank-1 weight strip is the full tensor
  ## - tile loads pad past-the-end slots with the `roSum` identity, so short and ragged rows contribute nothing to the sum
  ## - `acc.reduceAcc(reg, roSum)` accumulates each tile's sum of squares onto the accumulator, one subgroup butterfly per tile
  ##
  ## Mean accumulation in fp32. `rsqrt` is the backend native reciprocal square root.
  ##
  ## Preconditions:
  ## - the axis size is at least 1
  ## - dst has the same shape as src
  ## - the launch threadgroup holds exactly one warp's threads, `getLaneWidth()` in total,
  ##   off-contract launches are undefined behavior
  ## - when withWeight, the weight is rank 1 with one slot per axis position,
  ##   sized to the source axis size, see `rmsnorm`
  ##
  ## Args:
  ## - dst: same shape as src, receives the normalized rows
  ## - src: the source tensor
  ## - weight: one slot per axis position, shape (axis size,), any strides,
  ##   unread when withWeight is false
  ##
  ## Scalars:
  ## - eps: the stability constant added to the mean of squares
  ## - axis: the axis normalized over
  ## - withWeight: static switch, false compiles the weight out
  var reg: tile(float32)
  const
    W = getLaneWidth()
    owned = reg.size() div W
  let thrID = getLocalThreadID()
  # The reduce's butterfly is subgroup-scoped: a threadgroup of any other size leaves
  # per-simdgroup partials uncombined or rows partially written.
  # TODO: cross-warp combine via shared memory and barrier.
  when src.rank() == 1:
    # the rank-1 slice is empty, its size is a static Int countup rejects
    let rows = 1
  else:
    let rows = src.sliceAxes(axis).size()
  let gridStride = getGridSize()
  # Map selection from the source layout's strides:
  # - the reduced axis is the lane mode when it is memory-contiguous,
  #   maxCommonVector >= lane width, lanes share each row and a subgroup
  #   butterfly combines the partials
  # - the row mode takes over when it is strided: one lane per row, no butterfly, direct stores, see rmsnormStridedImpl
  # - both maps index rows in the non-reduced part's compact order, every
  #   element addressed through its own layout
  let lanesShareRow = src.diceAxes(axis).maxCommonVector() >= W
  if lanesShareRow:
    for g in countup(getLocalTileID(src.shape), rows - 1, gridStride):
      let
        srcRow = atAxisIndex(src, axis, g)
        dstRow = atAxisIndex(dst, axis, g)
        rowSize = srcRow.size()
        tiled = make_layout(rowSize).logical_divide(reg.getLayout())
        stripTiles = tiled.getTileGrid().size()
      # pass 1: sum of squares, one register tile at a time
      var acc = zero(float32)
      for tileID in 0 ..< stripTiles:
        reg.load(srcRow, thrID, tileID, padWith = reduceIdentity(roSum, float32))
        for i in 0 ..< owned:
          reg.data[i] = reg.data[i].square()
        acc.reduceAcc(reg, roSum)
      let scale = rsqrt(acc / float32(rowSize) + eps)
      # pass 2: scaled apply, the weight broadcast along the axis
      when withWeight:
        let weightStrip = atAxisIndex(weight, 0, 0)
        for r in countup(thrID, rowSize - 1, W):
          dstRow[r] = srcRow[r] * scale * weightStrip[r]
      else:
        for r in countup(thrID, rowSize - 1, W):
          dstRow[r] = srcRow[r] * scale
  else:
    rmsnormStridedImpl(dst, src, weight, eps, axis, withWeight)

proc rmsnorm*(dst, src: distinct TensorView, eps: float32, axis: static int) {.inline.} =
  ## Root-mean-square normalization over the input axis, fully overwriting dst.
  ##
  ## Each row computes x * rsqrt(mean(x²) + eps).
  ## Mean accumulation in fp32, identical to the reference implementations.
  ##
  ## Preconditions:
  ## - the axis size is at least 1
  ## - dst has the same shape as src
  ## - the launch threadgroup holds exactly one warp's threads, `getLaneWidth()` in total,
  ##   off-contract launches are undefined behavior
  ##
  ## Args:
  ## - dst: same shape as src, receives the normalized rows
  ## - src: the source tensor
  ## - eps: the stability constant added to the mean of squares
  ## - axis: the axis normalized over
  rmsnormImpl(dst, src, src, eps, axis, withWeight = false)

proc rmsnorm*(dst, src, weight: distinct TensorView, eps: float32, axis: static int) {.inline.} =
  ## Root-mean-square normalization over the input axis, per-axis weight,
  ## fully overwriting dst.
  ##
  ## Each row computes x * rsqrt(mean(x²) + eps) * weight, the weight broadcast
  ## along the normalized axis. Element r of the row scales by weight[r],
  ## as in `mx.fast.rms_norm(x, weight, eps)` on the last axis.
  ##
  ## Preconditions:
  ## - the axis size is at least 1
  ## - dst has the same shape as src
  ## - the launch threadgroup holds exactly one warp's threads, `getLaneWidth()` in total,
  ##   off-contract launches are undefined behavior
  ## - the weight is rank 1 with one slot per axis position, shape (axis size,)
  ##
  ## The weight size must equal the source axis size: mismatched static shapes
  ## fail to compile, a runtime shape is the caller's responsibility,
  ## a mismatch is undefined behavior.
  ##
  ## Args:
  ## - dst: same shape as src, receives the normalized rows
  ## - src: the source tensor
  ## - weight: one slot per axis position, shape (axis size,), any strides
  ##
  ## Scalars:
  ## - eps: the stability constant added to the mean of squares
  ## - axis: the axis normalized over
  rmsnormImpl(dst, src, weight, eps, axis, withWeight = true)
