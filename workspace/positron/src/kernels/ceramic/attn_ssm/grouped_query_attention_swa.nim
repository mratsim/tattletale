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
##   - o: (num_qo, H, D) row-major, written row-bounded
##   - q: (pad_qo, H, D) row-major, pad_qo = ceil(num_qo/8)·8
##   - k, v: (pad_kv, Nkv, D) row-major, pad_kv = ceil(num_kv/8)·8
##   - num_qo, num_kv, q_offset, H, Nkv, window: int32
##   - D: static int, 64 or 128 (tile geometry). No generic brackets,
##     no stride or scratch parameters.
##
## Padding contract: the q, k and v buffers hold an 8-row multiple,
## the tile loads fetch full 8-row blocks, a partial trailing block
## would read past the buffer end, callers zero-fill the padding rows:
## - the band mask excludes the padding rows (the upper band edge is
##   min(num_kv − 1, p) and num_kv >= q_offset + num_qo keeps every
##   real query row below num_kv)
## - the zero v rows add nothing to the P·V mma
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
import ../tile_io_rows
from ../math_consts import Log2e

export layout_algebra, tensors, tile_algebra, ptr_arithmetic

# ═════════════════════════════════════════════════════════════════════
#  Local device extension: the banded window mask
#  ═════════════════════════════════════════════════════════════════════

proc maskBand[A: static MmaAtom](
    tile: var RtLeft[float32, 8, 8, A],
    limit, window: int32) {.device.} =
  ## Element (r, c) is attended iff
  ##   limit + r − window + 1 <= c <= limit + r.
  ## `limit` is the block's band offset, signed, a negative limit
  ## masks whole rows. Masked elements become −3.402823466e38 (the
  ## most-negative finite fp32), excluded by the online softmax.
  ## For limit = 8, window = 4, X marks masked elements:
  ##
  ##      c:  0 1 2 3 4 5 6 7
  ##  r = 0:  X X X X X . . .
  ##  r = 1:  X X X X X X . .
  ##  r = 2:  X X X X X X X .
  ##  r = 3:  X X X X X X X X
  ##  r = 4:  X X X X X X X X
  ##  r = 5:  X X X X X X X X
  ##  r = 6:  X X X X X X X X
  ##  r = 7:  X X X X X X X X
  const M = A.getM()
  const N = A.getN()
  const rowTiles = 8 div M
  const colTiles = 8 div N
  const vpt = A.getVpt()
  let lane = thread_index_in_threadgroup
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
  ## Launch dims (ceil(num_qo/8), H, 1), 32 lanes, x = the 8-row q block.
  ## One grid point computes 8 q rows against the window-band KV
  ## columns, causal-banded per row:
  ##
  ##            KV columns (blocks [kvStart, kvEnd))
  ##            kvStart   kvStart+1  …   kvEnd−1
  ##   q row 0   ███████   X                X
  ##   q row 1   ███████   ███████          X
  ##   …         ███████   ███████          ██
  ##   q row 7   ███████   ███████          ██
  ##
  ##   █ = attended (limit − window + 1 <= c <= limit + r),
  ##   X = masked, limit = qAbs − kvBlock·8, qAbs = q_offset + qBlock·8.
  ## kvStart = max(0, qAbs − window + 1) div 8,
  ## kvEnd = min(num_kv − 1, qAbs + 7) div 8 + 1.
  ## The last block may extend up to 7 rows past num_kv − 1, the
  ## caller's zero padding, excluded by the band mask.
  ##
  ## Dataflow, one threadgroup (tensors on edges, ops in boxes,
  ## per KV block of the online softmax):
  ##
  ##   q (8, D) ──┐
  ##              ▼
  ##   K (D, 8) ─► [ S = q·Kᵀ ] ─► [ ·log2e ] ─► [ window mask ] ─► [ m ← max(m, S) ]
  ##                                                                │
  ##   P̃ = exp2(S − m) ◄────────────────────────────────────────────┘
  ##        │                                         rescale = exp2(m_prev − m)
  ##        │              ┌────────────────────────────────────┐
  ##        │              │  l ← l·rescale + rowsum(P̃)        │
  ##   V (8, D) ──► [ O ← O·rescale + P̃·V (fp32) ]            │
  ##                └───────────────┬────────────────────────┘
  ##                                 ▼
  ##                 o (8, D) = O / l, fp16 store
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
  let q_mul = Log2e
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
