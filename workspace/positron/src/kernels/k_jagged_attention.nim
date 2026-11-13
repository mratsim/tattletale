# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ═══════════════════════════════════════════════════════════════════════
#  Jagged multi-head attention forward (Tile API)
# ═══════════════════════════════════════════════════════════════════════

## Jagged multi-head attention forward on the ceramic Tile API: decode and prefill
## in one kernel over a ragged page table.
##
## Buffers, element dtype first then int32 tables:
##
## | buffer    | shape                                      |
## | --------- | ------------------------------------------ |
## | o, q      | (num_qo_tokens, H, D)                      |
## | k_pool    | (num_pages, num_layers, page_size, Nkv, D) |
## | v_pool    | (num_pages, num_layers, page_size, Nkv, D) |
## | q_offsets | (num_seqs+1), per-segment q token ranges   |
## | kv        | JaggedKv: offsets, indices, seqlens        |
##
## - each segment's KV tokens live at page ids `indices[offsets[s] ..]`, the layer-major pool slab,
##   one pool layer per dispatch (the `layer` scalar)
## - page_size is a multiple of the KV tile's nt, each KV tile lies inside one page,
##   KV tokens bound at kv_len never at page multiples: the last page's garbage tail
##   zero-pads through the tile loads and never enters the math, a 0·NaN in the P·V
##   mma would poison the output
## - GQA: kv_head = q_head div (H div Nkv), per segment s with q_len = q_offsets[s+1] - q_offsets[s]
##   and kv_len = kv.seqlens[s]: decode (q_len == 1) attends [0, kv_len) with causal off,
##   prefill (q_len >= 2) attends causal over [0, kv_len), flash-attn varlen semantics
## - online softmax in the exp2 shape, q_mul = scale·log2(e), the running max starts
##   at the finite floor so the rescale alpha is finite

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/math
import workspace/positron/src/jagged_kvcache
export jagged_kvcache

export tile_algebra

# ═══════════════════════════════════════════════════════════════════════
#  The kernel
# ═══════════════════════════════════════════════════════════════════════

proc jagged_attn_fwd*(
    o, q, k_pool, v_pool: distinct TensorView,
    q_offsets: distinct TensorView,
    kv: JaggedKv,
    scale: float32, layer, H, Nkv: int,
    page_size, D: static int,
    windowed: static bool = false, windowSize: int = low(int32),
    mmaQK: static TiledMma) =
  ## Online softmax: each KV block of the segment's pages rescales the running
  ## O and token sum by exp2(m_prev - m), a windowed launch masks the KV tokens
  ## more than windowSize - 1 below each query's band and skips the tiles
  ## entirely below the window.
  ##
  ## - launch dims: (q token blocks, H, num_seqs), 32 lanes
  ## - the kernel grid-strides the linearized (q block, head, segment) tiles,
  ##   a wider grid only adds idle threadgroups
  static: doAssert mmaQK.atom.getThreadCount() == getLaneWidth(),
    "jagged attention: one simdgroup per threadgroup, the atom must cover the lanes"
  static: doAssert page_size mod mmaQK.NT == 0,
    "jagged attention: a kv tile's tokens must lie inside one page"

  const
    mt = mmaQK.MT
    nt = mmaQK.NT
    qkMma = tiledMma(float32, q.T, k_pool.T, mt, nt, D)
    pvMma = tiledMma(float32, q.T, v_pool.T, mt, D, nt)
    qMap  = qkMma.getThreadValueMap(opA)       # Q
    kMap  = qkMma.getThreadValueMap(opB)       # K: (S, Dk) storage, (N, K) view
    vMap  = pvMma.getThreadValueMap(opB)       # Vᵀ: (Dv, S) storage, (N, K) view
    oMap  = pvMma.getThreadValueMap(opC)
    qkMap = qkMma.getThreadValueMap(opC)       # scores: band cut
    qkCLayout = qkMma.getMmaLayout(opC)        # S tile, per lane
    pvCLayout = pvMma.getMmaLayout(opC)        # O tile, per lane
    rowStatsL = reducedLayout(qkMap.layout, 1)

  let qMul = scale * Log2e                     # exp2 shape of the softmax

  let thrID = getLocalThreadID()
  # Flat grid-stride over (q slot, head, segment), x-major, the contracted grid's
  # tiles covered once, a wider grid only adds idle threadgroups
  let
    qW = get_num_groups(0)
    hW = get_num_groups(1)
  for tileID in countup(getLocalTileID(getGridSize()), getGridSize() - 1, getGridSize()):
    let
      qSlot = tileID mod qW
      head = (tileID div qW) mod hW
      segId = tileID div (qW * hW)

    let
      q0: int = q_offsets.data[segId]
      qLen = q_offsets.data[segId + 1] - q0
      kvLen: int = kv.seqlens[segId]
      # decode (q_len = 1) attends cached tokens only, prefill (q_len >= 2) also attends the new tokens, one token more per q token
      decodeAdj = (if qLen == 1: 1 else: 0)
      cachedLen = kvLen - qLen + decodeAdj
      totalK = cachedLen + qLen - decodeAdj
      bandBase = cachedLen - decodeAdj
      # KV loop bounds: the segment's pages and the query tiles cap the walk
      pageBlocks = totalK.round_multiple_up(page_size) div nt
      kvHead = head div (H div Nkv)
      qBlocks = qLen.ceil_div(mt)
    # Per-head q/o tokens:
    # - [q0, q0 + qLen) of the packed (num_qo_tokens, H, D) tensors.
    # - Loads/stores past qLen are padded/dropped.
    let
      segQ = q(q0 ..< q0 + qLen, head, _)
      segO = o(q0 ..< q0 + qLen, head, _)

    var
      qReg: tile(q.T, qkMma.getMmaLayout(opA))    # (mt, D)
      qk = zeros(tile(float32, qkCLayout))        # S = Q·Kᵀ scores, (mt × nt)

    # Each workgroup starts at its own query tile and steps by the grid width,
    # so a grid smaller than the tile count still covers every tile
    for qBlock in countup(qSlot, qBlocks - 1, qW):
      # Per-tile softmax state: a workgroup runs several tiles when the grid
      # is smaller than the tile count
      var
        mI = finiteFloor(tile(float32, rowStatsL))  # running max over keys, (mt × 1)
          # k_tile_sdpa's finite-floor init: with bandBase >= 0 the first tile always contains the attended kPos 0
        lI = zeros(tile(float32, rowStatsL))        # running attention-weight sum, (mt × 1)
        acc = zeros(tile(float32, pvCLayout))       # (mt × D) accumulator
      let
        q0Local = qBlock * mt
        lastKey = (if cachedLen + q0Local + (mt - 1) < totalK - 1:
                     cachedLen + q0Local + (mt - 1)
                   else:
                     totalK - 1)
        lastAttBlock = lastKey div nt + 1
        kvBlocks = (if pageBlocks < lastAttBlock: pageBlocks else: lastAttBlock)
        # kvStart skips the tiles entirely below the window's left edge
        kvStart = (when windowed:
                     max(0, (bandBase + q0Local - windowSize + 1) div nt)
                   else:
                     0)
      qReg.load(segQ, qMap, thrID, (qBlock, 0))

      for kvBlk in kvStart ..< kvBlocks:
        let
          kvPos = kvBlk * nt
          pageIdx = kvPos div page_size
          inPage = kvPos - pageIdx * page_size
          pageId: int = kv.indices[kv.offsets[segId] + pageIdx]
          # Bounds at the covered tokens: a 0·NaN from the garbage tail would poison the P·V mma.
          covered = min(nt, kvLen - kvPos)
          # K subview: (covered, D) over the token stride
          # V subview: the transposed slab view (D, covered) that the P·V mma reads as Vᵀ
          vK = k_pool(pageId, layer, inPage ..< inPage + covered, kvHead, _)
          vV = v_pool(pageId, layer, inPage ..< inPage + covered, kvHead, _)
                 .transpose(0, 1)

        var kReg: tile(k_pool.T, qkMma.getMmaLayout(opB))  # (nt, D)
        kReg.load(vK, kMap, thrID, (0, 0))
        qk.fillWith(0'f32)
        qkMma.atom.gemm_tile(qk, qReg, kReg, qk)   # S = Q·Kᵀ   (MMA 1)

        qk *= qMul                                 # scale, before the band cut

        qk.applyAttentionMask(qkCLayout, qkMap, thrID, q0Local, kvPos):
          kPos > bandBase + qPos or
          (when windowed:
             kPos < bandBase + qPos - windowSize + 1
           else:
             false)

        var mIj: tile(float32, rowStatsL)
        mIj.max(qk, axis = 1, qkMap)
        mIj = max(mI, mIj)                         # merge into the running max
        var alpha: tile(float32, rowStatsL)        # rescale factor (mt × 1)
        alpha = mI - mIj
        alpha.exp2()
        mI = mIj
        qk -.= mIj
        qk.exp2()
        lI *= alpha                                # attention-weight-sum rescale
        var lIj: tile(float32, rowStatsL)          # (mt × 1) attention-weight sum
        lIj.sum(qk, axis = 1, qkMap)
        lI += lIj

        let p = qk.downcast(q.T)                   # P.to(q.T), layout preserved
        acc *.= alpha                              # (mt × D) *.= (mt × 1)
        var vReg: tile(v_pool.T, pvMma.getMmaLayout(opB))  # (D, nt), B operand of MMA 2
        vReg.load(vV, vMap, thrID, (0, 0))
        pvMma.atom.gemm_tile(acc, p, vReg, acc)    # O += P·V   (MMA 2)

      acc /.= lI                                   # (mt × D) /.= (mt × 1)
      segO.store(acc.downcast(o.T), oMap, thrID, (qBlock, 0))
