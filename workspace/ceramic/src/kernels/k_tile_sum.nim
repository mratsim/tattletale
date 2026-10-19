# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile sums as device functions, one threadgroup per covered tile.
##
## - `sum_full`: one total per tile, dst shaped as the tile grid
## - `sum_axis`: per-row totals over the spread axis, one partial per tile,
##   dst shaped (tileRows..., gridLast), accumulable across chunks
##
## Both loop the tile grid grid-stride and load the lane's block through
## the blocked convention.
##
## Contract:
## - reduction runs through the tile-level `sum`
## - lane 0 stores, every lane holds the totals after the tree

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra

proc sum_full*(blk: static tuple, dst, src: distinct TensorView) {.inline.} =
  ## One threadgroup covers one tile: the lane's block `blk`, its last
  ## axis scaled by the 32 lanes.
  ##
  ## Args:
  ## - blk: the lane's register block, last axis spread over the lanes
  ## - dst: the tile grid, one slot per covered tile
  ## - src: the source tensor
  let thrID = thread_index_in_threadgroup
  let tiled = src.getLayout().zipped_divide(blockMap(blk))
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = int(threadgroups_per_grid.x) * int(threadgroups_per_grid.y) *
        int(threadgroups_per_grid.z)
  for f in countup(grid.getLocalTileID(), cSize - 1, gridStride):
    let tileCrd = idx2crd(grid, f)
    var x = rt(src.T, blk)
    x.loadTile(src, thrID, tileCrd)
    let total = x.sum()
    if thrID == 0:
      dst.data[crd2idx(dst.getLayout(), tileCrd)] = total

proc sum_axis*(blk: static tuple, dst, src: distinct TensorView) {.inline.} =
  ## Per-tile totals over the spread axis, the block's last axis:
  ## per-row totals land in dst at the tile's grid position.
  ##
  ## Args:
  ## - blk: the lane's register block, last axis spread over the lanes
  ## - dst: shaped (tileRows..., gridLast), one partial per grid slot,
  ##   accumulable across chunks
  ## - src: the source tensor
  let thrID = thread_index_in_threadgroup
  let tiled = src.getLayout().zipped_divide(blockMap(blk))
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = int(threadgroups_per_grid.x) * int(threadgroups_per_grid.y) *
        int(threadgroups_per_grid.z)
  for f in countup(grid.getLocalTileID(), cSize - 1, gridStride):
    let tileCrd = idx2crd(grid, f)
    var x = rt(src.T, blk)
    x.loadTile(src, thrID, tileCrd)
    var ss = rt(src.T, reducedShapeOf(blk))
    ss.sum(x, reducedAxisOf(blk))
    if thrID == 0:
      for i in 0 ..< ss.size():
        let c = idx2crd(ss.getLayout(), i)
        dst.guardedStore(reducedStoreCrdFor(blk, tileCrd, c), ss[i])
