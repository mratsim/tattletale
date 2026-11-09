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
##
## Element dtype contract:
##
## - values upcast to float32 on the load, all math runs in float32
## - one rounding at the store through dst's element type,
##   round-to-nearest-even for bfloat16 / float16

import workspace/crucible
import workspace/ceramic/src/tile_algebra

proc rmsnormImpl(dst, src, weight: distinct TensorView, eps: float32, axis: static int, withWeight: static bool) {.inline.} =
  ## Row-wise RMS normalization shared by the two entry points.
  ##
  ## One warp per row, a grid-stride loop covers the row list from any launch grid size.
  ## Each output row is x * rsqrt(mean(x²) + eps), times the per-axis weight when withWeight. dst is fully overwritten.
  ##
  ## Sum of squares per row, one register tile at a time:
  ## - `alongAxisAt` cuts the source and destination strips per row. The rank-1 weight strip is the full tensor
  ## - tile loads pad past-the-end slots with zero, so short and ragged rows contribute nothing to the sum
  ## - `tileSum += x * x` per element, `acc += reduceAdd(tileSum)` folds each tile onto the accumulator, one subgroup butterfly per tile
  ##
  ## Element dtype contract:
  ## - values upcast to float32 on the load, all math runs in float32
  ## - one rounding at the store through dst's element type,
  ##   round-to-nearest-even for bfloat16 / float16
  ## - the loaded tile shares the float32 register tile's thread->value map:
  ##   the default map's chunk width is element-size dependent:
  ##   8 bfloat16 slots vs 4 float32 slots per 16-byte load
  ## - a shared map keeps the two tiles' element positions identical
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
  # reg's thread->value map also maps the loaded tile, see the dtype contract above
  var raw: tile(src.T, reg.getLayout())
  const
    W = getLaneWidth()
    owned = reg.size() div W
  let thrID = getLocalThreadID()
  # The reduce's butterfly is subgroup-scoped: a threadgroup of any other size leaves
  # per-simdgroup partials uncombined or rows partially written.
  # TODO: cross-warp combine via shared memory and barrier.
  # a rank-1 source is a single axis strip: sliceAxes on its only axis
  # yields an empty layout, alongAxisAt special-cases rank 1 the same way
  when src.rank() == 1:
    let rows = 1
  else:
    let rows = src.sliceAxes(axis).size()
  let gridStride = getGridSize()
  for g in countup(getLocalTileID(src.shape), rows - 1, gridStride):
    let
      srcRow = alongAxisAt(src, axis, g)
      dstRow = alongAxisAt(dst, axis, g)
      rowSize = srcRow.size()
      tiled = make_layout(rowSize).logical_divide(reg.getLayout())
      stripTiles = tiled.getTileGrid().size()
    # pass 1: sum of squares, one register tile at a time
    var acc = zero(float32)
    for tileID in 0 ..< stripTiles:
      raw.load(srcRow, thrID, tileID, padWith = zero(src.T))
      var tileSum = zero(float32)
      for i in 0 ..< owned:
        let x = raw.data[i].to(float32)
        tileSum += x * x
      acc += reduceAdd(tileSum)
    let scale = rsqrt(acc / float32(rowSize) + eps)
    # pass 2: scaled apply with per-axis weight broadcast,
    # one rounding at the store through dst's element type
    when withWeight:
      let weightStrip = alongAxisAt(weight, 0, 0)
      for r in countup(thrID, rowSize - 1, W):
        dstRow[r] = (srcRow[r].to(float32) * scale * weightStrip[r].to(float32)).to(dst.T)
    else:
      for r in countup(thrID, rowSize - 1, W):
        dstRow[r] = (srcRow[r].to(float32) * scale).to(dst.T)

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
