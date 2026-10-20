# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile AXPY as a device function, one threadgroup per tile.
##
## - `axpy`: y = alpha*x + y
## - the register tile map (32 lanes x chunk elements) tiles both inputs
## - x and y may carry different layouts of the same shape, elements
##   pair by coordinate

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra

proc axpy*(y, x: distinct TensorView, alpha: float32) {.inline.} =
  ## y = alpha*x + y, one threadgroup per tile, grid-stride over the tile grid.
  ##
  ## Args:
  ## - alpha: the scale on x
  ## - x: the source tensor
  ## - y: the accumulator, updated in place
  let thrID = thread_index_in_threadgroup
  var reg: tile(y.T)
  let tiled = make_layout(y.size()).logical_divide(reg.getLayout())
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = threadgroups_per_grid.x * threadgroups_per_grid.y * threadgroups_per_grid.z
  for tileID in countup(grid.getLocalTileID(), cSize - 1, gridStride):
    var rx: tile(x.T)
    var ry: tile(y.T)
    rx.loadTile(x, thrID, tileID)
    ry.loadTile(y, thrID, tileID)
    for i in 0 ..< rx.size():
      ry[i] = alpha * rx[i] + ry[i]
    y.storeTile(ry, thrID, tileID)
