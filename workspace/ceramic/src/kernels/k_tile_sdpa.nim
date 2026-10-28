# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SDPA: scaled dot product attention
##
## `o = softmax(scale·Q·Kᵀ)·V` over the last two dims (L, S),
## production shapes only: causal, sliding window, ragged lengths.
##
## Inputs (B, H, seq, D), output (B, Hq, L, Dv):
## - query: (B, Hq, L, Dk), key: (B, Hkv, S, Dk), value: (B, Hkv, S, Dv)
##
## Masking:
## - GQA: kv head = query head h div (Hq div Hkv)
## - Causal: lower-triangular band, SWA: lookback band composed with causal
## - Masked keys read -inf on the scores, never on the inputs
## - Band bottom-right aligned, offset = S - L, per query row q:
##
##   lo(q) = max(0, q + offset - windowSize)   when windowed
##   hi(q) = min(S, q + offset + 1)            when isCausal

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/math_consts
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import workspace/ceramic/src/kernels/k_layout_gemm

# ═══════════════════════════════════════════════════════════════════════
#  The kernel
# ═══════════════════════════════════════════════════════════════════════

proc scaled_dot_product_attention*(
      mmaQK, mmaPV: static TiledMma,
      O, Q, K, V: distinct TensorView,
      scale: float32, Dk, Dv: static int,
      isCausal: static bool = false,
      windowed: static bool = false,
      windowSize: int = -2147483648) =
  ## SDPA: Transformers' attention, `o = softmax(scale·Q·Kᵀ)·V`.
  ##
  ## Supports GQA, causal masking, sliding-window attention, ragged lengths.
  ##
  ## Schedules:
  ##   - mmaQK: `static TiledMma` for Q·Kᵀ, reduces over Dk
  ##   - mmaPV: `static TiledMma` for attention·V, reduces over S
  ##
  ## Parameters:
  ##   - scale: the Q·Kᵀ scale, 1/sqrt(head_dim) at the call site
  ##   - isCausal: the lower-triangular right bound
  ##   - windowed + windowSize: the sliding-window left bound
  ##
  ## `windowSize`'s default is `low(int32)`: the lowest device int.
  ## Launch arguments narrow to the device's 32-bit int, so no real window
  ## size can equal the sentinel.
  ##
  ## Equivalent python, bottom-right aligned (offset = S - L):
  ##
  ## def scaled_dot_product_attention(query, key, value, scale, is_causal, window_size):
  ##     attn_weight = query @ key.transpose(-2, -1) * scale
  ##     for q in range(query.shape[-2]):
  ##         lo = max(0, q + offset - window_size) if window_size else 0
  ##         hi = min(key.shape[-2], q + offset + 1) if is_causal else key.shape[-2]
  ##         attn_weight[..., q, :lo] = float("-inf")
  ##         attn_weight[..., q, hi:] = float("-inf")
  ##     # GQA: query head h reads kv head h // (Hq // Hkv)
  ##     attn_weight = softmax(attn_weight, dim=-1)
  ##     return attn_weight @ value
  static: doAssert mmaQK.atom.getThreadCount() == getLaneWidth(),
    "sdpa: one simdgroup per threadgroup, the atom must cover the lanes"

  const
    mt = mmaQK.MT
    nt = mmaQK.NT
    qkMma = tiledMma(float32, Q.T, K.T, mt, nt, Dk)
    pvMma = tiledMma(float32, Q.T, V.T, mt, Dv, nt)
    qMap  = qkMma.getThreadValueMap(opA)       # Q
    kMap  = qkMma.getThreadValueMap(opB)       # K: (S, Dk) storage, (N, K) view
    vMap  = pvMma.getThreadValueMap(opB)       # Vᵀ: (Dv, S) storage, (N, K) view
    oMap  = pvMma.getThreadValueMap(opC)
    qkMap = qkMma.getThreadValueMap(opC)       # scores: band cut
    qkCLayout = qkMma.getMmaLayout(opC)        # S tile, per lane
    pvCLayout = pvMma.getMmaLayout(opC)        # O tile, per lane

  let qMul = scale * Log2e                     # exp2 shape of the softmax

  let
    qTile = threadgroup_position_in_grid.x
    head  = threadgroup_position_in_grid.y
    batch = threadgroup_position_in_grid.z
    thrID = getLocalThreadID()
    group = Q.shape[1] div K.shape[1]
    kvHead = head div group                    # GQA, plain integer arithmetic
    offset = K.shape[2] - Q.shape[2]

    vQ = Q(batch, head, _, _)                  # (L, Dk), A operand of MMA 1
    vO = O(batch, head, _, _)
    vK = K(batch, kvHead, _, _)                # (S, Dk), (N, K) view of K
    vV0 = V(batch, kvHead, _, _)               # (S, Dv), transposed below
    vV = vV0.transpose(1, 0)                   # (Dv, S), B operand of MMA 2

  let (kvTiles, _) = vK.tileWith((nt, Dk)).getTileGrid()
  let
    thrCrd = idx2crd(qkMap.getThreadPartitioning(), thrID)
    regCrd = qkMap.layout.slice(((thrCrd, _), _))
    crdBase = qkMap.layout.crd2idx(((thrCrd, _), _))

  # Each workgroup starts at its own query tile and steps by the grid width,
  # so a grid smaller than the tile count still covers every tile
  let
    qGrid = vQ.tileWith((mt, Dk)).getTileGrid()
    (qStride, _, _) = (threadgroups_per_grid.x, threadgroups_per_grid.y, threadgroups_per_grid.z)
  for qTile in countup(threadgroup_position_in_grid.x, qGrid.size() - 1, qStride):
    var q: tile(Q.T, qkMma.getMmaLayout(opA))    # (MT, Dk)
    q.load(vQ, qMap, thrID, (qTile, 0))

    var qk = zeros(tile(float32, qkCLayout))      # S = Q·Kᵀ scores, (MT × NT)
    const rowStatsL = reducedLayout(qkMap.layout, 1)
    var m_i = finiteFloor(tile(float32, rowStatsL)) # running row max, (MT × 1)
      # init to a finite floor, not -inf: a fully-masked kv tile would NaN alpha through -inf - -inf and poison the row
    var l_i = zeros(tile(float32, rowStatsL))     # running row sum, (MT × 1)
    var acc = zeros(tile(float32, pvCLayout))     # (MT × Dv) accumulator

    let
      qBase = qTile * mt + offset                # this tile's first aligned query position
      kvStart = when windowed: max(0, qBase - windowSize) div nt else: 0
      kvEnd = when isCausal: min(kvTiles, (qBase + mt).ceil_div(nt)) else: kvTiles

    for kv in kvStart ..< kvEnd:
      var k: tile(K.T, qkMma.getMmaLayout(opB))  # (NT, Dk)
      k.load(vK, kMap, thrID, (kv, 0))
      qk.fillWith(0'f32)
      qkMma.atom.gemm_tile(qk, q, k, qk)         # S = Q·Kᵀ   (MMA 1)

      qk *= qMul                                 # scale, before the band cut

      var band: tile(bool, qkCLayout)
      forEach idx, var c in band:
        let (qRow, kCol) = crdBase + regCrd.crd2idx(idx)
        let qPos = qBase + qRow
        let kPos = kv * nt + kCol                # kv tile's column base
        c = kPos < (when windowed: max(0, qPos - windowSize) else: 0) or
            kPos >= (when isCausal: min(K.shape[2], qPos + 1) else: K.shape[2])
      qk.where(band, negInf(float32), qk)        # masked keys read -inf

      var m_ij: tile(float32, rowStatsL)         # (MT × 1) row max
      m_ij.max(qk, axis = 1, qkMap)
      m_ij = max(m_i, m_ij)                      # running row max
      var alpha: tile(float32, rowStatsL)        # rescale factor (MT × 1)
      alpha = m_i - m_ij
      alpha.exp2()
      m_i = m_ij
      qk -.= m_ij                                # (MT × NT) -.= (MT × 1)
      qk.exp2()
      l_i *= alpha                               # row-sum rescale
      var l_ij: tile(float32, rowStatsL)         # (MT × 1) row sum
      l_ij.sum(qk, axis = 1, qkMap)
      l_i += l_ij

      let p = qk.downcast(Q.T)                   # P.to(Q.T), (MT × NT), layout preserved
      acc *.= alpha                              # (MT × Dv) *.= (MT × 1)
      var v: tile(V.T, pvMma.getMmaLayout(opB))  # (Dv, NT), B operand of MMA 2
      v.load(vV, vMap, thrID, (0, kv))
      pvMma.atom.gemm_tile(acc, p, v, acc)       # O += P·V   (MMA 2)

    acc /.= l_i                                  # (MT × Dv) /.= (MT × 1)
    vO.store(acc.downcast(O.T), oMap, thrID, (qTile, 0))
