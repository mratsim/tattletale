# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensor sums as device functions over the default register tile.
##
## - `sum(dst, src)`: one total for the whole tensor
## - `sum(dst, src, axis)`: sums over the input axis, dst shaped as
##   src with the axis size 1

{.experimental: "codeReordering".}

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra

proc sum*(dst, src: distinct TensorView) {.inline.} =
  ## Computes sum at dst[0]
  ##
  ## Args:
  ## - dst: a one-slot tensor for the total, must be zero-initialized,
  ##   the accumulation is atomic
  ## - src: the source tensor

  let thrID = getLocalThreadID()
  var reg: tile(src.T)
  let tiled = make_layout(src.size()).logical_divide(reg.getLayout())
  let axisTiles = tiled.getTileGrid().size()
  let gridStride = getGridSize()
  const owned = reg.size() div getLaneWidth()
  var acc = zero(src.T)

  # 1. Partial sums
  for tileID in countup(getLocalTileID(src.shape), axisTiles - 1, gridStride):
    reg.load(src, thrID, tileID)
    for i in 0 ..< owned:
      acc += reg.data[i]

  # 2. Epilogue
  let total = reduceAdd(acc)
  if thrID == 0:
    discard atomicFetchAdd(dst.data[0].addr, total)

proc sum*(dst, src: distinct TensorView, axis: static int) {.inline.} =
  ## Sums src over `axis` into dst, one total per output.
  ##
  ## Args:
  ## - dst: shaped like src with the axis size 1, every output written once
  ## - src: the source tensor
  ## - axis: the summed input axis
  const
    depthOwn = 32   # loads in flight per lane over the serial reduced walk
    depthShare = 4  # per-lane unroll, the warp holds laneW * depthShare partials
  let
    laneW = getLaneWidth()
    thrID = getLocalThreadID()
    gridN = getGridSize()
    axisSize = src.shape[axis]
    lanesShareOutput = src.diceAxes(axis).maxCommonVector() >= laneW
    outStep = if lanesShareOutput: gridN else: gridN * laneW
    out0 = if lanesShareOutput: getLocalTileID(dst.shape)
           else: getLocalTileID(dst.shape) * laneW + thrID
  for k in countup(out0, dst.size() - 1, outStep):
    let outCrd = idx2crd(dst.shape, k)
    var crd = outCrd
    # Two configurations:
    #
    #   Lanes share one output element, the reduced axis's elements are consecutive:
    #
    #     reduced i : 0     1    ...  31    32   ...  127  128  ...
    #     lane/acc  : l0a0  l1a0 ...  l31a0 l0a1 ...  l31a1 l0a0 ...
    #
    #     lane l rounds robin its depthShare accumulators over
    #     i = l + j*laneW for j = 0..depthShare-1, then i jumps laneW*depthShare
    #     per-lane sum -- cross-lane reduce --> total -- lane 0 --> dst[out]
    #
    #   Each lane owns one output element, the reduced axis's elements are stride apart:
    #
    #     reduced i : i0   i0+1  i0+2  i0+3   i0+4  ...
    #     accs      : a0   a1    a2    a3     a0    ...  `depthOwn` in flight
    #
    #     lane l owns output k, the lanes take consecutive k, the next
    #     chunk of outputs lands one launch-grid step later
    #     accs -- sum --> total -- lane l --> dst[k]
    if lanesShareOutput:
      var accs: array[depthShare, src.T]
      for j in 0 ..< depthShare:
        accs[j] = zero(src.T)
      let step = laneW * depthShare
      let fullEnd = axisSize.round_multiple_down(step)
      for i in countup(thrID, fullEnd - 1, step):
        for j in 0 ..< depthShare:
          crd[axis] = i + j * laneW
          accs[j] += src[crd]
      for i in countup(fullEnd + thrID, axisSize - 1, laneW):
        crd[axis] = i
        accs[0] += src[crd]
      var laneTotal = zero(src.T)
      for j in 0 ..< depthShare:
        laneTotal += accs[j]

      let total = reduceAdd(laneTotal)
      if thrID == 0:
        dst[outCrd] = total

    else:
      var accs: array[depthOwn, src.T]
      for j in 0 ..< depthOwn:
        accs[j] = zero(src.T)
      let fullEnd = axisSize.round_multiple_down(depthOwn)
      for i0 in countup(0, fullEnd - 1, depthOwn):
        for j in 0 ..< depthOwn:
          crd[axis] = i0 + j
          accs[j] += src[crd]
      for i in fullEnd ..< axisSize:
        crd[axis] = i
        accs[0] += src[crd]
      var total = zero(src.T)
      for j in 0 ..< depthOwn:
        total += accs[j]
      dst[outCrd] = total
