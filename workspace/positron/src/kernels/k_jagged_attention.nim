# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ═══════════════════════════════════════════════════════════════════════
#  Jagged multi-head attention forward (Tile API)
# ═══════════════════════════════════════════════════════════════════════

## Jagged multi-head attention forward on the ceramic Tile API: decode and prefill in one kernel over a ragged page table.
##
## - each segment's KV tokens are stored at the page ids
##   `indices[offsets[s] ..]`, gathered from the layer-major PagePool
##   slab `(num_pages, num_layers, page_size, Nkv, D)`
## - one pool layer per dispatch, selected by the `layer` scalar
## - sequence metadata is passed as DeviceMem-backed bridged arrays
##   `(q_offsets, kv)`
##
## Dataflow per threadgroup:
##
##   q tile (mt×D) --> Q·Kᵀ --× q_mul--> S tile (mt×nt)
##   k tile: nt-token block, one page slice <- indices[offsets[s] + t div page_size] <- k_pool slab
##   S tile --banded causal mask--> exp2 --> P (mt×nt) element dtype
##   v tile: nt-token block, same page walk <- v_pool slab
##   P --P·V mma--> O (mt×D) fp32 --÷ token norm--> element-dtype store
##
## with mt = mmaQK.MT query tokens and nt = mmaQK.NT KV tokens per tile.
##
## Online softmax: each KV block rescales the running O and token sum by exp2(m_prev − m_cur).
##
## Regimes, per segment s with q_len = q_offsets[s+1] − q_offsets[s] and kv_len = kv_seqlens[s]:
##
## - decode (q_len == 1): causal off with the query attending the KV tokens `[0, kv_len)`
## - prefill (q_len >= 2): causal on over `[0, kv_len)`, query token j
##   attends keys `[0, kv_len − q_len + j]`, flash-attn varlen semantics
## - grouped-query attention (GQA): kv_head = q_head div (H div Nkv)
##
## Buffers, element dtype first then int32 tables then scalars:
##
## | buffer    | shape                                          |
## | --------- | ---------------------------------------------- |
## | o, q      | (num_qo_tokens, H, D)                          |
## | k_pool    | (num_pages, num_layers, page_size, Nkv, D)     |
## | v_pool    | (num_pages, num_layers, page_size, Nkv, D)     |
## | q_offsets | (num_seqs+1), per-segment q token ranges       |
## | kv        | JaggedKv: offsets, indices, seqlens, batchSize |
##
## Pool layout, layer-major:
##
## - page stride: num_layers·page_size·Nkv·D
## - per-layer stride: page_size·Nkv·D
## - token stride: Nkv·D, head dim contiguous
## - the 16-bit element type is the DSL's `float16`, a distinct uint16 bit-identical on the host
##
## KV fetch, page_size a multiple of nt so each KV tile lies inside one page,
## for tile t0 of segment s:
##
## - page_idx = t0 div page_size, in_page = t0 mod page_size,
##   page_id = indices[offsets[s] + page_idx] read through the kernel's
##   JaggedTensorView
## - slab base = page_id·stride[0] + layer·stride[1] + in_page·stride[2] +
##   kv_head·stride[3] with the pool view's own strides
## - the natural layer-major slab gives stride[0] = num_layers·page_size·Nkv·D,
##   stride[1] = page_size·Nkv·D, stride[2] = Nkv·D, stride[3] = D
## - the V subview is the transposed (D, nt) view and the K subview is (nt, D)
##   over the token stride, so the P·V mma consumes Vᵀ from the natural slab
##
## Partial last page:
##
## - KV tokens are bounded by kv_len per segment, never by page multiples.
##   The last page may therefore hold garbage beyond the covered length, NaN included
## - the tile loads bound at the segment's covered tokens, so the garbage tail
##   zero-pads and never enters the math, a 0·NaN in the P·V mma would poison the output
## - padded scores sit outside the band and are -inf
## - the KV loop bounds at min(pageBlocks, lastAttBlock) within the segment's pages and the last attended block
##
## Numerics:
##
## - online softmax in the exp2 shape, q_mul = scale·log2(e) folded once into the scores,
##   so exp2(S − m) equals the exp form exp(qk·scale − max) computed by the fixtures'
##   f64 reference, one log2(e) factor apart
## - the running max starts at the finite floor, so the rescale alpha is finite
## - P is downcast to the element dtype (`convert`) before the P·V mma when the elements are fp16
## - the output store quantizes the fp32 O tile to the element dtype (round-to-nearest-even, RNE) through the `to` chokepoint
##
## Production contract:
##
## - single addressing path, no toggle: the kernel is the one addressing path
##   and the Tile API has no `useCeramicLayout` switch
## - the metadata arrays are DeviceMem-backed bridged args and the kernel
##   body builds its JaggedTensorView over them, consuming the KV side
##   strictly through the view's contract
## - scale is a runtime float32 (the host passes 1/sqrt(D)), no head-dim-dependent constant is compiled into the kernel

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
    mmaQK, mmaPV: static TiledMma) =
  ## Launch dims: (q token blocks, H, num_seqs), 32 lanes.
  ##
  ## - x: the segment's mt-token q block
  ## - y: head
  ## - z: segment
  ##
  ## One grid point computes mt query tokens against the segment's KV
  ## tokens (its pages), causal-banded per token:
  ##
  ##             KV tokens (segment pages)
  ##              page 0    page 1    …
  ##   q tok 0    ███████   X         X
  ##   q tok 1    ███████   ██        X
  ##   …          ███████   ███████   ██
  ##   q tok mt−1 ███████   ███████   ██
  ##
  ##   █ = attended (kPos <= bandBase + qPos), X = masked
  ##
  ## - a windowed launch masks the KV tokens more than windowSize - 1 below
  ##   each query token's position bandBase + qPos, and the loop start skips
  ##   the tiles entirely below the window
  ##
  ## Dataflow, one threadgroup, per KV block of the online softmax, tensors on edges, ops in boxes:
  ##
  ##   q (mt, D) ──┐
  ##               ▼
  ##   K (D, nt) ──► [ S = q·Kᵀ (fp32) ]──× q_mul──► [ band mask ]──► [ m ← max(m, S) ]
  ##                                                                      │        │
  ##   P̃ = exp2(S − m) ◄──────────────────────────────────────────────────┘        ▼
  ##         │                                                        rescale = exp2(m_prev − m)
  ##         │            ┌─────────────────────────────────────────────────────────┐
  ##         └───────────►│  l ← l·rescale + tokensum(P̃)                            │
  ##                      │                                                         │
  ##   V (nt, D) ────────►│  O ← O·rescale + P̃·V (fp32)                             │
  ##                      └────────────────────────────┬────────────────────────────┘
  ##                                                   │  per KV block
  ##                                                   ▼
  ##                                o (mt, D) = O / l, element-dtype store
  ##
  ## Output:
  ##
  ## - o is fully written over the segment's q tokens and the banded mask gives masked KV blocks zero attention weight
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

  let
    qBlock0 = threadgroup_position_in_grid.x
    (qStride, _, _) = (threadgroups_per_grid.x, threadgroups_per_grid.y, threadgroups_per_grid.z)
    head = threadgroup_position_in_grid.y
    segId = threadgroup_position_in_grid.z
    thrID = getLocalThreadID()

  let
    q0: int = q_offsets.data[segId]
    qLen = q_offsets.data[segId + 1] - q0
    kvLen: int = kv.seqlens[segId]
    # decode (q_len = 1) attends cached tokens only, prefill (q_len >= 2)
    # also attends the new tokens, one token more per q token
    decodeAdj = (if qLen == 1: 1 else: 0)
    cachedLen = kvLen - qLen + decodeAdj
    totalK = cachedLen + qLen - decodeAdj
    bandBase = cachedLen - decodeAdj
    # KV loop bounds: the segment's pages and the query tiles cap the walk
    pageBlocks = round_multiple_up(totalK, page_size) div nt
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
    mI = finiteFloor(tile(float32, rowStatsL))  # running max over keys, (mt × 1)
      # k_tile_sdpa's finite-floor init: with bandBase >= 0 the first tile always contains the attended kPos 0
    lI = zeros(tile(float32, rowStatsL))        # running attention-weight sum, (mt × 1)
    acc = zeros(tile(float32, pvCLayout))       # (mt × D) accumulator

  let
    thrCrd = idx2crd(qkMap.getThreadPartitioning(), thrID)
    regCrd = qkMap.layout.slice(((thrCrd, _), _))
    crdBase = qkMap.layout.crd2idx(((thrCrd, _), _))

  # Each workgroup starts at its own query tile and steps by the grid width,
  # so a grid smaller than the tile count still covers every tile
  for qBlock in countup(qBlock0, qBlocks - 1, qStride):
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

      var band: tile(bool, qkCLayout)
      forEach idx, var c in band:
        let (qRow, kCol) = crdBase + regCrd.crd2idx(idx)
        let qPos = q0Local + qRow                # query token in the segment
        let kPos = kvPos + kCol                  # KV token in the segment
        c = kPos > bandBase + qPos or
            (when windowed:
               kPos < bandBase + qPos - windowSize + 1
             else:
               false)
      qk.where(band, negInf(float32), qk)        # masked scores are -inf

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
    segO.store(acc.downcast(o.T), oMap, thrID, (qBlock0, 0))