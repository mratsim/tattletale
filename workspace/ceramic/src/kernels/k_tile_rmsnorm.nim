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
import workspace/ceramic/src/tile_algebra

proc rmsnormImpl(dst, src, weight: distinct TensorView, eps: float32, axis: static int, withWeight: static bool) {.inline.} =
  ## Row-wise RMS normalization shared by the two entry points.
  ##
  ## One warp per row, a grid-stride loop covers the row list from any launch grid size.
  ## Each output row is x * rsqrt(mean(x²) + eps), times the per-axis weight when withWeight. dst is fully overwritten.
  ##
  ## Sum of squares per row, one register tile at a time:
  ## - `rowView` cuts the source and destination strips per row, the rank-1 weight strip is the full tensor
  ## - tile loads pad past-the-end slots with the `roSum` identity, so short and ragged rows contribute nothing to the sum
  ## - `reduce(acc, reg, roSum, clear = false)` accumulates each tile's sum of squares onto the accumulator, one subgroup butterfly per tile
  ##
  ## Mean accumulation in fp32. `rsqrt` is the backend native reciprocal square root.
  ##
  ## Preconditions:
  ## - the axis size is at least 1
  ## - dst has the same shape as src
  ## - the launch threadgroup holds exactly one warp's threads, `getLaneWidth()` in total,
  ##   other launches are rejected and dst comes back NaN
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
  # NaN from a runtime division: a folded 0/0 constant emits an invalid MSL token
  var poison = zero(float32)
  poison = poison / poison
  # The reduce's butterfly is subgroup-scoped: a threadgroup of any other size leaves
  # per-simdgroup partials uncombined or rows partially written.
  # Deferred: cross-warp combine via shared memory and barrier.
  let
    nThreads = threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z
    offContract = nThreads != W
  let
    rows = src.sliceAxes(axis).size()
    gridStride = getGridSize()
  for g in countup(getLocalTileID(src.shape), rows - 1, gridStride):
    let
      srcRow = rowView(src, axis, g)
      dstRow = rowView(dst, axis, g)
      rowSize = srcRow.size()
      tiled = make_layout(rowSize).logical_divide(reg.getLayout())
      stripTiles = tiled.getTileGrid().size()
    # pass 1: sum of squares, one register tile at a time
    var acc = zero(float32)
    for tileID in 0 ..< stripTiles:
      reg.load(srcRow, thrID, tileID, padWith = reduceIdentity[float32](roSum))
      for i in 0 ..< owned:
        reg.data[i] = reg.data[i] * reg.data[i]
      reduce(acc, reg, roSum, clear = false)
    let scale = rsqrt(acc / float32(rowSize) + eps)
    # pass 2: scaled apply, the weight broadcast along the axis
    when withWeight:
      let weightStrip = rowView(weight, 0, 0)
      for r in countup(thrID, rowSize - 1, W):
        dstRow[r] = srcRow[r] * scale * weightStrip[r]
    else:
      for r in countup(thrID, rowSize - 1, W):
        dstRow[r] = srcRow[r] * scale

proc rmsnorm*(dst, src: distinct TensorView, eps: float32, axis: static int) {.inline.} =
  ## Root-mean-square normalization over the input axis, fully overwriting dst.
  ##
  ## Each row computes x * rsqrt(mean(x²) + eps).
  ## Mean accumulation in fp32, identical to the reference implementations.
  ##
  ## Preconditions: the axis size is at least 1, dst has the same shape as src.
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
  ## Preconditions: the axis size is at least 1, dst has the same shape as src.
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
