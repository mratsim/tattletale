# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Elementwise divmod as a device function, one threadgroup per tile.
##
## - `divmod`: q = floor(x/y), r = x - q*y
## - the register tile map (32 lanes x chunk elements) tiles all inputs
## - x, y, q and r may carry different layouts of the same shape, elements
##   pair by coordinate
## - floor goes through an int32 truncation, |x/y| must stay below 2^31

import workspace/ceramic/src/tile_algebra
import workspace/crucible/src/builtins/builtins_gpu_types

proc k_divmod*(q, r, x, y: distinct TensorView) =
  let thrID = getLocalThreadID()
  var reg: tile(x.T)
  let tiled = make_layout(x.size()).logical_divide(reg.getLayout())
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = getGridSize()
  for tileID in countup(grid.getLocalTileID(), cSize - 1, gridStride):
    let rx = tile(x.T).load(x, thrID, tileID)
    let ry = tile(y.T).load(y, thrID, tileID)
    var rq: tile(q.T)
    var rr: tile(r.T)
    for i in 0 ..< rx.size():
      when typeof(rx[i]) is bfloat16:
        let xf = rx[i].to(float32)
        let yf = ry[i].to(float32)
      else:
        let xf = rx[i]
        let yf = ry[i]
      let d = xf / yf
      var qi = float32(int32(d))
      if qi > d:
        qi -= 1.0
      when typeof(rx[i]) is bfloat16:
        rq[i] = qi.to(bfloat16)
        rr[i] = (xf - qi * yf).to(bfloat16)
      else:
        rq[i] = qi
        rr[i] = xf - qi * yf
    q.store(rq, thrID, tileID)
    r.store(rr, thrID, tileID)
