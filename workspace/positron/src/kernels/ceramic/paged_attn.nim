## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Paged multi-head attention forward (Tile API port)
#
# ############################################################

## Paged multi-head attention forward on the ceramic Tile API: decode
## and prefill in one kernel, K/V fetched per 8-row block through
## a dense `block_table` + `cache_seqlens`
## (no page-to-contiguous gather). The K/V come from the layer-major
## PagePool slab (num_pages, num_layers, page_size, Nkv, D), one layer
## per kernel dispatch via the `layer` scalar.
##
## Dataflow per threadgroup:
##
##   q tile (8×D) --× q_mul--> Q·Kᵀ --> S tile (8×8)
##   k tile: 8-row block <- block_table[seq, t div page_size] <- k_cache slab
##   S tile --banded causal mask--> exp2 --> P (8×8) fp16
##   v tile: 8-row block <- block_table[seq, t div page_size] <- v_cache slab
##   P --P·V mma--> O (8×D) fp32 --÷ row norm--> fp16 store
##
## The softmax is online: each kv block rescales the running O
## and row sum by exp2(m_prev − m_cur).
##
## Modes:
##   - decode (q_len == 1): causal off, each query attends the cached
##     rows [0, cache_seqlen)
##   - prefill (q_len ≥ 2): causal on over [0, cache_seqlen + q_len),
##     query row j attends keys [0, cache_seqlen + j]
##     (flash-attn varlen semantics)
##   - GQA: kv_head = q_head div (H div Nkv)
##
## Buffers, fp16 first then int32 tables then scalars:
##   - o, q: (num_qo_tokens, H, D)
##   - k_cache, v_cache: (num_pages, num_layers, page_size, Nkv, D)
##     layer-major pool (page stride num_layers·page_size·Nkv·D,
##     per-layer row stride Nkv·D, head-dim contiguous)
##   - block_table: (num_seqs, max_pages) dense, -1 padding
##   - cache_seqlens: (num_seqs)
##   - cu_seqlens_q: (num_seqs+1), the prefill query ranges
##   - num_seqs, H, Nkv, max_pages, num_layers, layer: int32
##   - page_size: static int, multiple of 8 (tile geometry)
##   - D: static int, 64 or 128 (tile geometry). No generic brackets,
##     no stride or scratch parameters.
##
## The 16-bit element type is the DSL's `float16` (distinct uint16,
## bit-identical on the host).
##
## Block fetch, page_size a multiple of 8 so each 8-row KV block lies
## inside one page. For block t0:
##   - page_idx = t0 div page_size, in_page = t0 mod page_size,
##     page_id = block_table[seq·max_pages + page_idx]
##   - slab base = page_id·(num_layers·page_size·Nkv·D) +
##     layer·(page_size·Nkv·D) + in_page·(Nkv·D) + kv_head·D
##     (the layer-major pool position), passed as the origin's batch
##     component, the block rows local to it
##   - the K view carries stride (1, 1, Nkv·D, 1). The V view carries
##     the transposed stride (1, 1, 1, Nkv·D), so the P·V mma consumes
##     Vᵀ from the natural slab view
##
## Partial last page:
##   - K/V rows are bounded by cache_seqlens[seq] (decode)
##     or cache_seqlen + q_len (prefill), never by page multiples.
##     The tail page may hold garbage beyond the covered length
##     (bound, don't filter)
##   - the KV loop stays inside the seq's table pages
##     (pageBlocks = ceil(totalK/page_size)·(page_size div 8))
##     and inside the last block's attended range
##     (lastKey = min(cachedLen + q0Local + 7, totalK − 1)). A block
##     beyond the last table page would read a -1 padding slot
##   - garbage rows load as valid slab memory and are excluded
##     by the banded mask. Masked S columns become the most-negative
##     finite fp32 (−3.402823466e38), so the running row max ignores
##     them and exp2(S − m) underflows to exact +0.0. The P, l
##     and O accumulations never see them
##
## Numerics:
##   - online softmax in the exp2 shape, q_mul = scale·log2(e) folded
##     into Q (exp2(S·q_mul − m) is the exp(S·scale) convention,
##     scale = 1/sqrt(D))
##   - P downcast to fp16 (`convert`) before the P·V mma. The output
##     store quantizes the fp32 O tile to fp16 (RNE) through the `to`
##     chokepoint
##
## Local device procs: the shared tile algebra has no banded causal
## mask, so the module carries `maskCausal` locally. The row-bounded
## loads/stores come from `tile_io_rows`.
##
## Production contract:
##   - single addressing path, no toggle: the kernel is the one
##     addressing path, and the Tile API has no `useCeramicLayout`
##     switch
##   - batch metadata (block_table, cache_seqlens, cu_seqlens_q)
##     arrives as raw pointers. A future AttentionBatchInfo struct
##     passes the same raw pointers, so the signature stays unchanged

import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ════════════════════════════════════════
#  Local device extensions: the tile API gaps the paged fetch needs
# ════════════════════════════════════════

proc maskCausal[A: static MmaAtom](
    tile: var RtLeft[float32, 8, 8, A],
    limit: int32) {.device.} =
  ## Banded causal mask on an 8×8 S tile: element (r, c) is attended
  ## iff c <= limit + r. Masked elements become the most-negative
  ## finite fp32 (−3.402823466e38), so the online softmax excludes
  ## them (the row max ignores them, exp2(S − m) underflows to exact
  ## +0.0). `limit` is the block's band offset (cachedLen + q0Local −
  ## kv_idx·8 − decodeAdj), signed: a negative limit masks every
  ## column of the row, where an unsigned wrap would leave them
  ## attended. The frag walk follows the loadTile lane→element mapping,
  ## so the mask hits exactly the elements the Q·Kᵀ mma produced.
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
        let band = limit + int32(row + n * M)
        if int32(col + m * N + v) > band:
          tile.frags[n][m].frag[v] = -3.402823466e38'f32

# ════════════════════════════════════════
#  The kernel
# ════════════════════════════════════════

proc paged_attn_fwd*(
    o: ptr UncheckedArray[float16],            # (num_qo_tokens, H, D)
    q: ptr UncheckedArray[float16],            # (num_qo_tokens, H, D)
    k_cache, v_cache: ptr UncheckedArray[float16],  # (num_pages, num_layers, page_size, Nkv, D) layer-major pool
    block_table: ptr UncheckedArray[int32],    # (num_seqs, max_pages) dense
    cache_seqlens: ptr UncheckedArray[int32],  # (num_seqs)
    cu_seqlens_q: ptr UncheckedArray[int32],   # (num_seqs+1) prefill q ranges
    num_seqs, H, Nkv, max_pages, num_layers, layer: int32,
    page_size: static int, D: static int) {.device.} =
  ## One grid point = (seq, head, 8-row q block), following the module
  ## doc's contract: the q tile loads row-bounded, the KV loop is
  ## bounded by the seq's table pages and the banded causal mask, and
  ## the fp16 store writes only the seq's q rows.
  ## Grid: (q token blocks, H, num_seqs), 32 lanes. x = the seq's
  ## 8-row q block, y = head, z = seq.
  static: doAssert D == 64 or D == 128
  static: doAssert page_size mod 8 == 0

  let seqId = int32(threadgroup_position_in_grid.z)
  let head = int32(threadgroup_position_in_grid.y)
  let qBlock = int32(threadgroup_position_in_grid.x)

  # The q/o views carry the (q row, head, dim) strides. The K/V views
  # carry the slab's row stride and take the per-block base as the
  # origin's batch component (see the module doc's fetch formula).
  let gd_q = q.gd(shape = (-1, -1, -1, -1), stride = (H * D, D, H * D, 1))
  let gd_o = o.gd(shape = (-1, -1, -1, -1), stride = (H * D, D, H * D, 1))
  let gd_k = k_cache.gd(shape = (-1, -1, -1, -1), stride = (1, 1, Nkv * D, 1))
  # The V view is the transposed slab view: the RtRight loadTile
  # hands Vᵀ to the P·V mma.
  let gd_v = v_cache.gd(shape = (-1, -1, -1, -1), stride = (1, 1, 1, Nkv * D))

  let kvHead = head div (H div Nkv)
  let q0 = cu_seqlens_q[seqId]
  let qLen = cu_seqlens_q[seqId + 1] - q0
  let cachedLen = cache_seqlens[seqId]
  # decode (q_len = 1) attends cached rows only. Prefill (q_len ≥ 2)
  # also covers the new tokens' rows (the band carries the one-row
  # difference)
  let decodeAdj = (if qLen == 1: 1'i32 else: 0'i32)
  let totalK = cachedLen + qLen - decodeAdj
  let q0Local = qBlock * 8
  let pageStride = num_layers * page_size * Nkv * int32(D)
  let rowStride = Nkv * int32(D)
  # The KV loop stays inside the seq's table pages and the last
  # attended block.
  let pageBlocks = ((totalK + page_size - 1) div page_size) * (page_size div 8)
  let lastKey = (if cachedLen + q0Local + 7 < totalK - 1:
                   cachedLen + q0Local + 7
                 else:
                   totalK - 1)
  let lastAttBlock = lastKey div 8 + 1
  let kvBlocks = (if pageBlocks < lastAttBlock: pageBlocks else: lastAttBlock)

  var q_reg: rt_l(float16, 8, D)
  var k_reg: rt_r(float16, D, 8)
  var v_reg: rt_r(float16, 8, D)
  var att_block: rt_l(float32, 8, 8, getTileConfig(float32, float16))
  var p_reg: rt_l(float16, 8, 8)
  var o_reg: rt_l(float32, 8, D, getTileConfig(float32, float16))
  var max_vec_last: rv(float32, 8, 8)
  var max_vec: rv(float32, 8, 8)
  var norm_vec: rv(float32, 8, 8)

  q_reg.loadTileRows(gd_q, (q0, head, qBlock, 0), qLen)
  max_vec.neg_infty()
  norm_vec.zero()
  o_reg.zero()
  let q_mul = (if D == 128: 0.08838834764'f32 else: 0.125'f32) *
              1.44269504089'f32
  q_reg.mul(q_reg, q_mul)
  for kv_idx in 0'i32 ..< kvBlocks:
    let pageIdx = (kv_idx * 8) div page_size
    let inPage = kv_idx * 8 - pageIdx * page_size
    let pageId = block_table[seqId * max_pages + pageIdx]
    let base = pageId * pageStride + layer * page_size * Nkv * int32(D) +
               inPage * rowStride + kvHead * int32(D)
    # The block's rows are local to the base: the origin's row-tile
    # component stays 0.
    k_reg.loadTile(gd_k, (base, 0, 0, 0))
    att_block.zero()
    att_block.mma_AB(q_reg, k_reg)
    maskCausal(att_block, cachedLen + q0Local - kv_idx * 8 - decodeAdj)
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
    v_reg.loadTile(gd_v, (base, 0, 0, 0))
    o_reg.mma_AB(p_reg, v_reg)
  o_reg.div_row(o_reg, norm_vec)
  gd_o.storeTileRows(o_reg, (q0, head, qBlock, 0), qLen)
