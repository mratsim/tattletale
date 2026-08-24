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
import ../atoms
import ../tile_algebra/tile_config
import ../tile_algebra/tile_views
import ../tile_algebra/tile_io
import ../tile_algebra/tile_ops
import ../tile_algebra/tile_mma

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_config, tile_views, tile_io, tile_ops, tile_mma

proc attn_fwd*[TIn, TOut](q, k, v: ptr UncheckedArray[TIn],
                          o: ptr UncheckedArray[TOut], H, N: int32,
                          D: static int) {.device.} =
  ## Computes one 8×D output tile of non-causal attention per threadgroup:
  ## O = softmax(Q·Kᵀ/√D)·V, the online (running max/sum) softmax over the N/8 KV blocks.
  ## Expected input: q, k, v, o are row-major (B·H·N)×D, grid (q-seq block, head, batch) with 32 threads.
  ## The scores and the output accumulator are fp32.
  ## The store rounds to TOut.
  ## Constraints: N a host-guaranteed multiple of 8, D a static multiple of 8 (asserted).
  ## No host padding needed.
  static: doAssert D mod 8 == 0

  let gl_q = makeGl(q, 1, H, N, D)
  let gl_k = makeGl(k, 1, H, N, D)
  let gl_o = makeGl(o, 1, H, N, D)
  # The V view swaps the (N, D) strides: the load transposes
  # the buffer, so the mma reads V in its natural order.
  # The kv-block index rides the origin's col slot.
  let gl_v = makeGlStrided(v, H * N * D, N * D, 1, D)

  let Block = threadgroup_position_in_grid.z
  let head = threadgroup_position_in_grid.y
  let q_seq = threadgroup_position_in_grid.x

  let kv_blocks = N div 8

  var q_rtl: rt_l(TIn, 8, D)
  var k_rtr: rt_r(TIn, D, 8)
  var v_rtr: rt_r(TIn, 8, D)
  var att_rtl: rt_l(float32, 8, 8)
  var o_rtl: rt_l(float32, 8, D)
  var max_vec_last: ColVecOf(float32, 8, 8)
  var max_vec: ColVecOf(float32, 8, 8)
  var norm_vec: ColVecOf(float32, 8, 8)

  q_rtl.load(gl_q, (Block, head, q_seq, 0))
  max_vec.neg_infty()
  norm_vec.zero()
  o_rtl.zero()
  # q_mul = 1/√D·log2(e): the S = Q·Kᵀ scale folded per-D,
  # so the exp2 online-softmax runs as a max-subtraction.
  # rsqrt avoids the 1/sqrt(D) division (a literal ÷ builtin fold crashes the compiler).
  let q_mul = rsqrt(float32(D)) * 1.44269504089'f32
  q_rtl.mul(q_rtl, q_mul)
  for kv_idx in 0'i32 ..< kv_blocks:
    k_rtr.load(gl_k, (Block, head, kv_idx, 0))
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
    o_rtl.mul_row(o_rtl, max_vec_last)
    v_rtr.load(gl_v, (Block, head, 0, kv_idx))
    # The S tile's C fragment IS the P·V step's A fragment (A and C share
    # the AC layout), so the handoff is a type-level no-op.
    o_rtl.mma_AB(att_rtl, v_rtr)
  o_rtl.div_row(o_rtl, norm_vec)
  o_rtl.store(gl_o, (Block, head, q_seq, 0))
