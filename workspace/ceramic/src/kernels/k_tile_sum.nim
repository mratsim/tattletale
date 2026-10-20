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

import std/typetraits
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra

proc sum*(dst, src: distinct TensorView) {.inline.} =
  ## Computes sum at dst[0]
  ##
  ## Args:
  ## - dst: a one-slot tensor for the total
  ## - src: the source tensor
  let thrID = thread_index_in_threadgroup
  var reg: tile(src.T)
  let tiled = make_layout(src.size()).logical_divide(reg.getLayout())
  let grid = tiled.getTileGrid()
  const owned = reg.size() div getThreadCount()
  var acc = zero(src.T)
  # 1. Partial sums
  #    Every threadgroup folds the entire tile grid into its lanes,
  #    replication keeps the single-launch form correct, the two-pass
  #    form at the bottom lifts it at scale.
  for tileID in 0 ..< grid.size():
    reg.loadTile(src, thrID, tileID)
    for i in 0 ..< owned:
      acc += reg.data[i]
  # 2. Epilogue
  #    Butterfly across the lanes, lane 0 banks the total.
  let total = reduceAdd(acc)
  if thrID == 0:
    dst.data[0] = total

proc sum*(dst, src: distinct TensorView, axis: static int) {.inline.} =
  ## Sums over the input axis: dst is shaped as src with the axis size 1,
  ## one partial per kept coordinate.
  ##
  ## Args:
  ## - dst: shaped as src with the axis size 1
  ## - src: the source tensor
  ## - axis: the input axis summed over
  let thrID = thread_index_in_threadgroup
  var reg: tile(src.T)
  const owned = reg.size() div getThreadCount()
  let
    rest = src.getLayout().sliceAxes(axis)
    red = src.getLayout().diceAxes(axis)
    axisTiles = red.size().ceil_div(reg.size())
    gridStride = threadgroups_per_grid.x * threadgroups_per_grid.y * threadgroups_per_grid.z
  for g in countup(getLocalTileID(dst.shape), rest.size() - 1, gridStride):
    let crd = idx2crd(dst.shape, g)
    let strip = make_view(src.displace(crd).data, red)
    var acc = zero(src.T)
    for tileID in 0 ..< axisTiles:
      reg.loadTile(strip, thrID, tileID)
      for i in 0 ..< owned:
        acc += reg.data[i]
    let total = reduceAdd(acc)
    if thrID == 0:
      dst[crd] = total
