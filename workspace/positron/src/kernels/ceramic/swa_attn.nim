## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Sliding-window attention forward (Tile API port)
#
# ############################################################

## Sliding-window attention forward on the ceramic Tile API.
## Gemma-4-E2B text sliding-layer attention, one 8×D q tile per threadgroup.
## Experimental: not a production kernel, known gaps below, not fixed.
##
## Dataflow per threadgroup:
##
##   q (8×D) --Q·Kᵀ--> S (8×8) --·log2(e)--> band mask --> softmax --> P (8×8) fp16
##   k: 8-row blocks over the window band --------------------------------+
##   P --P·V mma--> O (8×D) fp32 --÷ row norm--> fp16 store
##   v: 8-row blocks over the window band -----------------+
##
## The softmax is online: each kv block rescales O and the row sum
## by exp2(m_prev − m_cur). The Gemma-4 text scale is 1.0
## (`self.scaling = 1.0`, no 1/sqrt(D) division), so q_mul = log2(e)
## scales the fp32 S tile after the mma and the mma consumes fp16 Q
## unscaled.
##
## Buffers, fp16 first then scalars:
##   - o, q: (num_qo, H, D) row-major
##   - k, v: (num_kv, Nkv, D) row-major
##   - num_qo, num_kv, q_offset, H, Nkv, window: int32
##   - D: static int, 64 or 128 (tile geometry). No generic brackets,
##     no stride or scratch parameters.
##
## Window band (query row i, absolute position p = q_offset + i):
##
##   attended keys j: max(0, p − window + 1) <= j <= min(num_kv − 1, p)
##
##   key rows:  0 ...... p−window+1 ...... p ...... num_kv−1
##              |           |            |          |
##              +-- masked -+--- band ---+-- masked+
##
##   GQA: kv_head = h div (H div Nkv). num_kv >= q_offset + num_qo.
##
## Numerics:
##   - online softmax in the exp2 shape. q_mul = log2(e) applies in fp32
##     after the mma, so exp2(S·q_mul − m) is the exp(S·scale)
##     convention with scale = 1.0
##   - masked S elements are the most-negative finite fp32
##     (−3.402823466e38). The running row max ignores them.
##     exp2(S − m) underflows to exact +0.0
##   - P downcast to fp16 (`convert`) before the P·V mma. The output
##     store quantizes the fp32 O tile to fp16 (RNE) through the `to`
##     chokepoint
##
## Known production gaps (documented, not fixed):
##   - D ∈ {64, 128} only. Gemma-4's real head_dim is 256.
##     The window/scale/GQA semantics are D-independent.
##   - Single sequence: one (num_qo, H, D) q buffer, no batch dim.
##   - k/v not projected in-kernel.

import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ═════════════════════════════════════════════════════════════════════
#  Local device extension: the banded window mask
#  ═════════════════════════════════════════════════════════════════════

proc maskBand[A: static MmaAtom](
    tile: var RtLeft[float32, 8, 8, A],
    limit, window: int32) {.device.} =
  ## Banded sliding-window mask on an 8×8 S tile.
  ## Element (r, c) is attended iff c <= limit + r and c >= limit + r − window + 1.
  ## Masked elements become the most-negative finite fp32
  ## (−3.402823466e38). The online softmax excludes them: the row max
  ## ignores them and exp2(S − m) underflows to exact +0.0. `limit`
  ## is the block's band offset (qAbs − kv_idx·8), signed. A negative limit
  ## masks every column of the row. An unsigned wrap leaves them attended.
  ## The frag walk follows the loadTile lane→element mapping, so the mask
  ## hits exactly the elements that the Q·Kᵀ mma produced.
  const M = A.getM()
  const N = A.getN()
  const rowTiles = 8 div M
  const colTiles = 8 div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let p = limit + int32(row + n * M)
        let c = int32(col + m * N + v)
        if c > p or c < p - window + 1:
          tile.frags[n][m].frag[v] = -3.402823466e38'f32

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc swa_attn_fwd*(
    o: ptr UncheckedArray[float16],     # (num_qo, H, D) fp16 output
    q: ptr UncheckedArray[float16],     # (num_qo, H, D) fp16 queries, already projected
    k: ptr UncheckedArray[float16],     # (num_kv, Nkv, D) fp16 keys, already projected
    v: ptr UncheckedArray[float16],     # (num_kv, Nkv, D) fp16 values, already projected
    num_qo, num_kv, q_offset, H, Nkv, window: int32,
    D: static int) {.device.} =
  ## One grid point = one (head, 8-row q block), following the module
  ## doc's contract: the q tile loads row-bounded by num_qo. The KV
  ## loop covers the window band blocks [kvStart, kvEnd). The fp16
  ## store writes only the q rows below num_qo.
  ##
  ## Grid (ceil(num_qo/8), H, 1), 32 lanes. The KV loop covers the blocks
  ## [kvStart, kvEnd):
  ##   - qAbs = q_offset + qBlock·8
  ##   - kvStart = max(0, qAbs − window + 1) div 8
  ##   - kvEnd = (min(num_kv − 1, qAbs + 7)) div 8 + 1
  ## The last block may extend up to 7 rows past num_kv − 1.
  ## Overhang rows load as zero (Metal bounds-checks buffer reads).
  ## The causal bound excludes them.
  static: doAssert D == 64 or D == 128

  let qBlock = int32(threadgroup_position_in_grid.x)
  let head = int32(threadgroup_position_in_grid.y)

  # The q/o views carry the (q row, head, dim) strides. The K/V views
  # carry the buffer row stride. Their per-block base is the origin's
  # batch component (the kvStart/kvEnd fetch formula).
  let gl_q = q.gd(shape = (-1, -1, -1, -1), stride = (H * D, D, H * D, 1))
  let gl_o = o.gd(shape = (-1, -1, -1, -1), stride = (H * D, D, H * D, 1))
  let gl_k = k.gd(shape = (-1, -1, -1, -1), stride = (1, 1, Nkv * D, 1))
  # The V view is the transposed slab view: the RtRight loadTile
  # hands Vᵀ to the P·V mma.
  let gl_v = v.gd(shape = (-1, -1, -1, -1), stride = (1, 1, 1, Nkv * D))

  let kvHead = head div (H div Nkv)
  let qAbs = q_offset + qBlock * 8
  let lo = qAbs - window + 1
  let kvStart = (if lo > 0: lo else: 0) div 8
  let hi = (if num_kv - 1 < qAbs + 7: num_kv - 1 else: qAbs + 7)
  let kvEnd = hi div 8 + 1
  let rowStride = Nkv * int32(D)

  var q_reg: rt_l(float16, 8, D)
  var k_reg: rt_r(float16, D, 8)
  var v_reg: rt_r(float16, 8, D)
  var att_block: rt_l(float32, 8, 8, getTileConfig(float32, float16))
  var p_reg: rt_l(float16, 8, 8)
  var o_reg: rt_l(float32, 8, D, getTileConfig(float32, float16))
  var max_vec_last: rv(float32, 8, 8)
  var max_vec: rv(float32, 8, 8)
  var norm_vec: rv(float32, 8, 8)

  q_reg.loadTileRows(gl_q, (0, head, qBlock, 0), num_qo)
  max_vec.neg_infty()
  norm_vec.zero()
  o_reg.zero()
  # log2(e), the exp2-form scale. Gemma-4 text attention uses scale 1.0,
  # so q_mul applies in fp32 after the mma, keeping the fp16 Q unscaled.
  let q_mul = 1.44269504089'f32
  for kv_idx in kvStart ..< kvEnd:
    let base = kv_idx * 8 * rowStride + kvHead * int32(D)
    k_reg.loadTile(gl_k, (base, 0, 0, 0))
    att_block.zero()
    att_block.mma_AB(q_reg, k_reg)
    att_block.mul(att_block, q_mul)
    att_block.maskBand(qAbs - kv_idx * 8, window)
    max_vec_last.copy(max_vec)
    max_vec.row_max(att_block, max_vec)
    max_vec_last.sub(max_vec_last, max_vec)
    max_vec_last.exp2(max_vec_last)
    att_block.sub_row(att_block, max_vec)
    att_block.exp2(att_block)
    norm_vec.mul(norm_vec, max_vec_last)
    norm_vec.row_sum(att_block, norm_vec)
    p_reg.convert(att_block)
    o_reg.mul_row(o_reg, max_vec_last)
    v_reg.loadTile(gl_v, (base, 0, 0, 0))
    o_reg.mma_AB(p_reg, v_reg)
  o_reg.div_row(o_reg, norm_vec)
  gl_o.storeTileRows(o_reg, (0, head, qBlock, 0), num_qo)
