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
## - inputs promote to float32 and the sum rounds back to z's element type
##   (bfloat16 round back is round-to-nearest-even)

import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/crucible/src/builtins/builtins_gpu_types
import ../device_view

proc k_add*(z, x, y: distinct DeviceView) =
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
    yv = make_view(y.data +% y.offset, y.layout)
  let thrID = getLocalThreadID()
  var reg: tile(zv.T)
  let tiled = make_layout(zv.size()).logical_divide(reg.getLayout())
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = getGridSize()
  for tileID in countup(grid.getLocalTileID(), cSize - 1, gridStride):
    let rx = tile(xv.T).load(xv, thrID, tileID)
    let ry = tile(yv.T).load(yv, thrID, tileID)
    var rz: tile(zv.T)
    for i in 0 ..< rx.size():
      let s = rx[i].to(float32) + ry[i].to(float32)
      when zv.T is bfloat16:
        rz[i] = s.to(bfloat16)
      else:
        rz[i] = s
    zv.store(rz, thrID, tileID)
