# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Elementwise add kernel, one threadgroup per tile.
##
## - `add`: z = x + y
## - the register tile map (32 lanes x chunk elements) tiles all inputs
## - x, y and z may carry different layouts of the same shape, elements
##   pair by coordinate

import workspace/ceramic/src/tile_algebra
import workspace/crucible/src/builtins/builtins_gpu_types

proc k_add*(z, x, y: distinct TensorView) =
  let thrID = getLocalThreadID()
  var reg: tile(z.T)
  let tiled = make_layout(z.size()).logical_divide(reg.getLayout())
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = getGridSize()
  for tileID in countup(grid.getLocalTileID(), cSize - 1, gridStride):
    let rx = tile(x.T).load(x, thrID, tileID)
    let ry = tile(y.T).load(y, thrID, tileID)
    var rz: tile(z.T)
    for i in 0 ..< rx.size():
      when typeof(rx[i]) is bfloat16:
        rz[i] = (rx[i].to(float32) + ry[i].to(float32)).to(bfloat16)
      else:
        rz[i] = rx[i] + ry[i]
    z.store(rz, thrID, tileID)
