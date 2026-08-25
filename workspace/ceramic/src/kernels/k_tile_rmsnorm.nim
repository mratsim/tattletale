## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Tile RMSNorm kernel: one 8×128 row tile per threadgroup
#
# ############################################################

import workspace/crucible
import ../int_tuples
import ../layouts
import ../layout_constructors
import ../layout_indexing
import ../tensors
import ../ptr_arithmetic
import ../tile_algebra

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

proc rms_norm*[TIn, TOut](
    Out: ptr UncheckedArray[TOut], X, G: ptr UncheckedArray[TIn],
    C: int32, eps: float32) {.device.} =
  let gid = threadgroup_position_in_grid.y
  let gd_x = gd(X, shape = (-1, -1, -1, -1), stride = ( 8 * 128, 0, 1, 128))
  let gd_g = gd(G, shape = (-1, -1, -1, -1), stride = ( 0, 0, 1, 0))
  let gd_out = gd(Out, shape = (-1, -1, -1, -1), stride = ( 8 * 128, 0, 1, 128))
  var x_rtr: rt_r(float32, 8, 128)
  var gamma_rtr: rt_r(float32, 8, 128)
  var sq_rtr: rt_r(float32, 8, 128)
  var ss: rv(float32, 8, 128)
  loadTile(x_rtr, gd_x, (gid, 0, 0, 0))
  loadTile(gamma_rtr, gd_g, (0, 0, 0, 0))
  sq_rtr.mul(x_rtr, x_rtr)
  ss.row_sum(sq_rtr)
  let invC = 1.0'f32 / float32(C)
  ss.mul(ss, invC)
  ss.add(ss, eps)
  ss.rsqrt(ss)
  x_rtr.mul_row(x_rtr, ss)
  x_rtr.mul(x_rtr, gamma_rtr)
  storeTile(gd_out, x_rtr, (gid, 0, 0, 0))
