# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Paged SDPA decode: scaled dot product attention over a paged KV pool.
##
## Generalizes `k_tile_sdpa` to paged KV. Each sequence's keys and values
## live in a layer-major page pool, addressed per kv tile through a dense
## `block_table` with -1 padding and bounded by `cache_seqlens`.
##
## - decode regime: one query entry per sequence, causal off, attention
##   range [0, cache_seqlens[seq]), windowed variants attend
##   [max(0, cache_seqlens[seq] - windowSize), cache_seqlens[seq])
## - buffers: query/output (numSeqs, Hq, 1, D)
## - kv pools (numPages, numLayers, PageSize, Nkv, D), head dim contiguous,
##   layer `layer` served per dispatch
## - batch metadata: block_table (numSeqs, maxPages) int32 with -1 padding,
##   cache_seqlens (numSeqs,) int32
##
## Paging contract:
##   - a kv tile spans `nt` consecutive kv entries of one page:
##     PageSize mod nt == 0, the same rule ceramic_v1's
##     grouped_query_attention_paged states as page_size multiple of 8 at nt = 8
##   - the kv loop touches at most ceil(cache_seqlen / PageSize) table entries
##     and never reads a -1 padding slot
##   - windowed decode starts the kv loop at the window's left-edge page
##   - the caller guarantees maxPages * PageSize >= cache_seqlen >= 1
##
## Masking: entries at or beyond cache_seqlens[seq] read -inf on the scores,
## never on the inputs. Loads bound by cache_seqlens[seq] per kv tile zero-pad
## every uncovered entry through tile.load, so page tails never enter the math.

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/math

# ═══════════════════════════════════════════════════════════════════════
#  Launch guard
# ═══════════════════════════════════════════════════════════════════════

proc paged_sdpa_check_window*(windowed: static bool, windowSize: int) =
  ## Host-side launch guard for `paged_sdpa_decode`, call before `launchDeviceFn`.
  ##
  ## Precondition: a windowed decode needs `windowSize > 0`.
  ## A non-positive width empties the attention band and the output stores NaN.
  ## A full-context decode (windowed = false) accepts the `low(int32)` sentinel untouched.
  when windowed:
    doAssert windowSize > 0,
      "paged sdpa decode: windowed=true needs windowSize > 0, got " & $windowSize

# ═══════════════════════════════════════════════════════════════════════
#  The kernel
# ═══════════════════════════════════════════════════════════════════════

proc paged_sdpa_decode*(
      mmaQK, mmaPV: static TiledMma,
      O, Q, kvPoolK, kvPoolV: distinct TensorView,
      blockTable, cacheSeqlens: distinct TensorView,
      scale: float32, Dk, Dv, PageSize: static int,
      windowed: static bool = false, windowSize: int = low(int32),
      layer: int32) =
  ## Paged decode SDPA: `o[s, h] = softmax(scale·q[s,h]·Kᵀ)·V` over the entries [0, cache_seqlens[s]) gathered through `block_table`.
  ##
  ## Schedules:
  ##   - mmaQK: `static TiledMma` for Q·Kᵀ, reduces over Dk
  ##   - mmaPV: `static TiledMma` for attention·V, reduces over S
  ##   - windowed + windowSize: the sliding-window left bound, the band masks
  ##     entries below max(0, cache_seqlens[s] - windowSize)
  ##
  ## Precondition: a windowed decode needs windowSize > 0, a non-positive width
  ## empties the band and the output stores NaN.
  ## `paged_sdpa_check_window` raises on non-positive widths before launch.
  ##
  ## Launch: grid (1, Hq, numSeqs), one simdgroup per threadgroup.
  ## Grid y addresses the head, grid z the sequence, grid x stays 1 (one query tile in decode).
  ##
  ## `layer` selects the pool's layer slice: all sequences of one dispatch
  ## read one layer's keys and values.
  ##
  ## `windowSize`'s default is `low(int32)`: the lowest device int.
  ## Launch arguments narrow to the device's 32-bit int, so no real window
  ## width collides with the sentinel.
  static: doAssert mmaQK.atom.getThreadCount() == getLaneWidth(),
    "paged sdpa: one simdgroup per threadgroup, the atom must cover the lanes"
  static: doAssert PageSize mod mmaQK.NT == 0,
    "paged sdpa: a kv tile's entries must lie inside one page"

  const
    mt = mmaQK.MT
    nt = mmaQK.NT
    qkMma = tiledMma(float32, Q.T, kvPoolK.T, mt, nt, Dk)
    pvMma = tiledMma(float32, Q.T, kvPoolV.T, mt, Dv, nt)
    qMap  = qkMma.getThreadValueMap(opA)       # Q
    kMap  = qkMma.getThreadValueMap(opB)       # K: (S, Dk) storage, (N, K) view
    vMap  = pvMma.getThreadValueMap(opB)       # Vᵀ: (Dv, S) storage, (N, K) view
    oMap  = pvMma.getThreadValueMap(opC)
    qkMap = qkMma.getThreadValueMap(opC)       # scores: band cut
    qkCLayout = qkMma.getMmaLayout(opC)        # S tile, per lane
    pvCLayout = pvMma.getMmaLayout(opC)        # O tile, per lane

  let qMul = scale * Log2e                     # exp2 shape of the softmax

  let
    head  = threadgroup_position_in_grid.y
    seqId = threadgroup_position_in_grid.z
    thrID = getLocalThreadID()
    group = Q.shape[1] div kvPoolK.shape[3]
    kvHead = head div group                    # GQA, direct integer arithmetic
    seqlen = int(cacheSeqlens[seqId])
    windowLo = when windowed: max(0, seqlen - windowSize) else: 0

    vQ = Q(seqId, head, _, _)                  # (1, Dk), A operand of MMA 1
    vO = O(seqId, head, _, _)

  let
    thrCrd = idx2crd(qkMap.getThreadPartitioning(), thrID)
    regCrd = qkMap.layout.slice(((thrCrd, _), _))
    crdBase = qkMap.layout.crd2idx(((thrCrd, _), _))

  var q: tile(Q.T, qkMma.getMmaLayout(opA))    # (MT, Dk), one query tile in decode
  q.load(vQ, qMap, thrID, (0, 0))

  var qk = zeros(tile(float32, qkCLayout))        # S = Q·Kᵀ scores, (MT × NT)
  const rowStatsL = reducedLayout(qkMap.layout, 1)
  var m_i = finiteFloor(tile(float32, rowStatsL)) # running max per query entry
    # init to a finite floor, not -inf: a fully-masked kv tile would NaN alpha through -inf - -inf and poison the entry
  var l_i = zeros(tile(float32, rowStatsL))       # running sum per query entry
  var acc = zeros(tile(float32, pvCLayout))       # (MT × Dv) accumulator

  for kv in windowLo div nt ..< seqlen.ceil_div(nt):
    let
      kvPos = kv * nt
      pageIdx = kvPos div PageSize
      inPage = kvPos mod PageSize
      pageId = int(blockTable[seqId, pageIdx])
      kBase = pageId * kvPoolK.stride[0] + int(layer) * kvPoolK.stride[1] +
              inPage * kvPoolK.stride[2] + kvHead * kvPoolK.stride[3]
      vBase = pageId * kvPoolV.stride[0] + int(layer) * kvPoolV.stride[1] +
              inPage * kvPoolV.stride[2] + kvHead * kvPoolV.stride[3]
      nkvK = kvPoolK.stride[2]           # kv entry stride, above shape[3] * Dk on a padded pool
      nkvV = kvPoolV.stride[2]
      covered = min(nt, seqlen - kvPos)  # bounds the K/V views so tile.load zero-pads every uncovered entry
      kvCols = Dk                        # runtime ints only: mixed static/runtime shape tuples break make_layout (crd2idx)
      kvDv = Dv
      vK = make_view(kvPoolK.data +% kBase,
                     make_layout((covered, kvCols), (nkvK, 1)))
      vV = make_view(kvPoolV.data +% vBase,
                     make_layout((kvDv, covered), (1, nkvV)))

    var k: tile(kvPoolK.T, qkMma.getMmaLayout(opB))  # (NT, Dk)
    k.load(vK, kMap, thrID, (0, 0))
    qk.fillWith(0'f32)
    qkMma.atom.gemm_tile(qk, q, k, qk)         # S = Q·Kᵀ   (MMA 1)

    qk *= qMul                                 # scale, before the band cut

    var band: tile(bool, qkCLayout)
    forEach idx, var c in band:
      let (_, kCol) = crdBase + regCrd.crd2idx(idx)
      let kPos = kvPos + kCol                  # kv tile's column base
      c = kPos < windowLo or kPos >= seqlen
    qk.where(band, negInf(float32), qk)        # masked keys read -inf

    var m_ij: tile(float32, rowStatsL)
    m_ij.max(qk, axis = 1, qkMap)
    m_ij = max(m_i, m_ij)                      # running max per query entry
    var alpha: tile(float32, rowStatsL)        # alpha = exp2(m_prev - m_cur), the online-softmax rescale
    alpha = m_i - m_ij
    alpha.exp2()
    m_i = m_ij
    qk -.= m_ij
    qk.exp2()
    l_i *= alpha                               # rescale the running sum
    var l_ij: tile(float32, rowStatsL)
    l_ij.sum(qk, axis = 1, qkMap)
    l_i += l_ij

    let p = qk.downcast(Q.T)                   # P.to(Q.T), layout preserved
    acc *.= alpha
    var v: tile(kvPoolV.T, pvMma.getMmaLayout(opB))  # (Dv, NT), B operand of MMA 2
    v.load(vV, vMap, thrID, (0, 0))
    pvMma.atom.gemm_tile(acc, p, v, acc)       # O += P·V   (MMA 2)

  acc /.= l_i                                  # (MT × Dv) /.= (MT × 1)
  vO.store(acc.downcast(O.T), oMap, thrID, (0, 0))
