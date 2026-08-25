## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Tile attention kernel: one 8×D output tile per threadgroup
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

proc attn_fwd*[TIn, TOut](q, k, v: ptr UncheckedArray[TIn],
                          o: ptr UncheckedArray[TOut],
                          H, N: int32,
                          D: static int) {.device.} =
  ## Computes softmax(Q·Kᵀ / √D) · V

  static: doAssert D mod 8 == 0

  let gd_q = gd(q, 1, H, N, D)
  let gd_k = gd(k, 1, H, N, D)
  let gd_o = gd(o, 1, H, N, D)
  # The V view transposes the (N, D) strides.
  let gd_v = gd(v, shape = (-1, -1, -1, -1), stride = ( H * N * D, N * D, 1, D))

  let batch = threadgroup_position_in_grid.z
  let head = threadgroup_position_in_grid.y
  let q_seq = threadgroup_position_in_grid.x

  let kv_blocks = N div 8

  var q_rtl: rt_l(TIn, 8, D)
  var k_rtr: rt_r(TIn, D, 8)
  var v_rtr: rt_r(TIn, 8, D)
  var att_rtl: rt_l(float32, 8, 8, getTileConfig(float32, TIn))  # Attention full-precision
  var p_rtl: rt_l(TIn, 8, 8)                                     # Attention downcasted
  var o_rtl: rt_l(float32, 8, D, getTileConfig(float32, TIn))
  var max_vec_last: rv(float32, 8, 8)
  var max_vec: rv(float32, 8, 8)
  var norm_vec: rv(float32, 8, 8)

  loadTile(q_rtl, gd_q, (batch, head, q_seq, 0))
  max_vec.neg_infty()
  norm_vec.zero()
  o_rtl.zero()

  let q_mul = rsqrt(float32(D)) * 1.44269504089'f32
  q_rtl.mul(q_rtl, q_mul)
  for kv_idx in 0'i32 ..< kv_blocks:
    loadTile(k_rtr, gd_k, (batch, head, kv_idx, 0))
    att_rtl.zero()
    att_rtl.mma_AB(q_rtl, k_rtr)
    max_vec_last.copy(max_vec)
    max_vec.row_max(att_rtl, max_vec)
    max_vec_last.sub(max_vec_last, max_vec)
    max_vec_last.exp2(max_vec_last)
    att_rtl.sub_row(att_rtl, max_vec)
    att_rtl.exp2(att_rtl)
    norm_vec.mul(norm_vec, max_vec_last)
    norm_vec.row_sum(att_rtl, norm_vec)
    p_rtl.convert(att_rtl)                   # Downcast P to input type (usually bf16/fp16)
    o_rtl.mul_row(o_rtl, max_vec_last)
    loadTile(v_rtr, gd_v, (batch, head, 0, kv_idx))
    o_rtl.mma_AB(p_rtl, v_rtr)
  o_rtl.div_row(o_rtl, norm_vec)
  storeTile(gd_o, o_rtl, (batch, head, q_seq, 0))
