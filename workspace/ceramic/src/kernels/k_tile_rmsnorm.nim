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
import ../atoms
import ../tile_algebra/tile_config
import ../tile_algebra/tile_views
import ../tile_algebra/tile_io
import ../tile_algebra/tile_ops

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_config, tile_views, tile_io, tile_ops

proc rms_single_row*[TIn, TOut](
    Out: ptr UncheckedArray[TOut], X, G: ptr UncheckedArray[TIn],
    C: int32, eps: float32) {.device.} =
  ## Computes one 8×128 RMSNorm row tile per threadgroup
  ## (grid.y = row-tile index):
  ## y = x·rsqrt(mean(x²)+ε)·γ, mean over the C runtime columns, C ≤ 128.
  ## Expected input:
  ##   x, out: row-major M×128, data in the first C columns
  ##   γ: 128 slots, the first C real (the host zero-pads beyond C)
  ## The host zero-pads x/γ beyond C and dispatches grid (1, rows/8).
  ## Loads are unconditional.
  ## fp16 storage widens to fp32 in registers.
  ## The output rounds to the storage type at the write.
  let gid = threadgroup_position_in_grid.y
  let gl_x = makeGlStrided(X, 8 * 128, 0, 1, 128)
  let gl_g = makeGlStrided(G, 0, 0, 1, 0)
  let gl_out = makeGlStrided(Out, 8 * 128, 0, 1, 128)
  var x_rtr: rt_r(float32, 8, 128)
  var gamma_rtr: rt_r(float32, 8, 128)
  var sq_rtr: rt_r(float32, 8, 128)
  var ss: ColVecOf(float32, 8, 128)
  x_rtr.load(gl_x, (gid, 0, 0, 0))
  gamma_rtr.load(gl_g, (0, 0, 0, 0))
  sq_rtr.mul(x_rtr, x_rtr)
  ss.row_sum(sq_rtr)
  let invC = 1.0'f32 / float32(C)
  ss.mul(ss, invC)
  ss.add(ss, eps)
  ss.rsqrt(ss)
  x_rtr.mul_row(x_rtr, ss)
  x_rtr.mul(x_rtr, gamma_rtr)
  x_rtr.store(gl_out, (gid, 0, 0, 0))
