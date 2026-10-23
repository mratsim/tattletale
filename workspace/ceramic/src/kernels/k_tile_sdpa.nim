# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Scaled dot product attention, `softmax(scale · Q·Kᵀ + mask) · V`,
## tiled flash-style over the tile algebra.
##
## One simdgroup per (batch, query head, 32-row query tile) keeps
## the online softmax entirely in fragment registers. Both matmuls are
## TiledMma schedules over the registry atom of the element type.
##
## - grouped-query attention needs no flag: the kernel recomposes key,
##   value and mask views so the flat query-head index reads kv head
##   `h div (Hq div Hkv)` through the layout, never materializing repeats
## - the additive mask is a (B, Hm, S, S) float32 view broadcast over
##   batch and heads by zero strides, added before the softmax
## - causal masking skips fully-masked key tiles, blocks the diagonal
##   band per element
## - a fully-masked row produces NaN, matching torch
##
## Schedules are `static TiledMma` params, schedule-first like `gemm`:
## `tiledMma(float32, T, T, 32, 32, 32)` for both products. `Dk`/`Dv`
## are compile-time head dims with statically sized register tiles.
## L, S, B and every stride live in the views' layouts.

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties

{.experimental: "callOperator".}

const NegBigF32* = -3.0e38'f32
  ## Finite blocked-score sentinel: `exp2(x − m)` underflows to zero on blocked entries.
  ## The running max never leaves the finite range.

func narrowTo*[T](x: float32): T {.inline.} =
  ## f32 accumulator value narrowed to the mma operand dtype
  when T is float32:
    x
  elif T is float16:
    toFp16(x)
  elif T is bfloat16:
    bfloat16(x)
  else:
    {.error: "no mma operand dtype for the requested element type".}

template sdpaImpl(mmaQK, mmaPV: static TiledMma,
                  o, q, k, v, mask: distinct TensorView,
                  scale: float32, Dk, Dv: static int,
                  Causal, Masked: static bool) =
  ## One threadgroup computes one (batch, head, 32-row query tile) output,
  ## `o = softmax(qkScale·Q·Kᵀ + mask)·V` with f32 accumulation.
  ##
  ## Score and output fragments live per lane, k/v streams over key tiles.
  ## Unmasked entry points pass `q` for `mask`. The `Masked = false` instantiation compiles the mask path out.
  static:
    doAssert mmaQK.atom == mmaPV.atom, "sdpa: both schedules share one atom"
    doAssert mmaQK.NT == mmaPV.KT, "sdpa: P reuses the S fragment, mmaQK.NT must equal mmaPV.KT"
    doAssert mmaQK.MT == mmaPV.MT, "sdpa: both schedules share the query tile height"
    doAssert not Masked or mask.T is float32, "sdpa: the additive mask is float32"

  const
    MT = mmaQK.MT
    NT = mmaQK.NT
    atomM = mmaQK.atom.getM()
    Mo = MT div atomM
    DkChunks = Dk.ceil_div(mmaQK.KT)
    DvChunks = Dv.ceil_div(mmaPV.NT)
    tvmA = mmaQK.getThreadValueMap(opA)
    tvmK = mmaQK.getThreadValueMap(opB)
    tvmV = mmaPV.getThreadValueMap(opB)
    tvmS = mmaQK.getThreadValueMap(opC)
    tvmO = mmaPV.getThreadValueMap(opC)
    sLayout = mmaQK.getMmaLayout(opC)
    oLayout = mmaPV.getMmaLayout(opC)

  let
    thrID = getLocalThreadID()
    qTile = threadgroup_position_in_grid.x
    head = threadgroup_position_in_grid.y
    batch = threadgroup_position_in_grid.z

  let qkScale = scale * 1.4426950408889634'f32

  if thrID < getLaneWidth():
    # per-(batch, head) views. The GQA head map is layout composition.
    # The flat query-head index decomposes over the (group, Hkv) mode whose
    # strides are (0, kv-head stride), so the kv head of query head h is h div group.
    let
      Hq = q.shape[1]
      Hkv = k.shape[1]
      group = Hq div Hkv
      qSlice = q(batch, head, _, _)
      oSlice = o(batch, head, _, _)
      kvLayout = make_layout((q.shape[0], (group, Hkv), k.shape[2], k.shape[3]),
          (k.stride[0], (0, k.stride[1]), k.stride[2], k.stride[3]))
      kSlice = make_view(k.data, kvLayout)(batch, head, _, _)
      vTLayout = make_layout((q.shape[0], (group, Hkv), v.shape[3], v.shape[2]),
          (v.stride[0], (0, v.stride[1]), v.stride[3], v.stride[2]))
      vTSlice = make_view(v.data, vTLayout)(batch, head, _, _)
      # the mask broadcasts over batch and heads through zero strides
      maskHeadStride = if mask.shape[1] == 1: 0 else: mask.stride[1]
      maskBatchStride = if mask.shape[0] == 1: 0 else: mask.stride[0]
      maskLayout = make_layout((q.shape[0], (group, Hq), mask.shape[2], mask.shape[3]),
          (maskBatchStride, (0, maskHeadStride), mask.stride[2], mask.stride[3]))
      maskSlice = make_view(mask.data, maskLayout)(batch, head, _, _)
      kvTiles = k.shape[2].ceil_div(NT)
      diagTile = (qTile * MT + MT - 1) div NT
      kvEnd = if Causal: min(diagTile + 1, kvTiles) else: kvTiles

    # the lane's fragment slot -> tile-cell map, from the same layout
    # machinery the tile loads use. The slice gives the slot offsets
    # relative to the lane's atom cell, crd2idx of the tiled identity
    # layout gives that cell. Rows atomM*ro + baseM are the lane's rows.
    let
      thrCrd = idx2crd(tvmS.getThreadPartitioning(), thrID)
      sIdentity = make_identity_layout(tvmS.tileShape)
      sTiled = sIdentity.logical_divide(tvmS.layout)
      sel = (((thrCrd, _), _), (0, 0))
      sCrdMap = sTiled.slice(sel)
      (baseM, baseN) = sTiled.crd2idx(sel)
      mLane = baseM mod atomM

    var rQ: array[DkChunks, tile(q.T, mmaQK.getMmaLayout(opA))]
    for c in 0 ..< DkChunks:
      rQ[c].load(qSlice, tvmA, thrID, (qTile, c))

    var
      mState: array[Mo, float32]
      lPart: array[Mo, float32]
    for ro in 0 ..< Mo:
      mState[ro] = NegBigF32
      lPart[ro] = 0.0'f32
    var rOut: array[DvChunks, tile(float32, oLayout)]
    for c in 0 ..< DvChunks:
      rOut[c].fillWith(zero(float32))

    var
      rS: tile(float32, sLayout)
      rK: tile(k.T, mmaQK.getMmaLayout(opB))
      rV: tile(v.T, mmaPV.getMmaLayout(opB))

    for j in 0 ..< kvEnd:
      # S = Q·Kᵀ over the head dimension, f32 accumulation
      rS.fillWith(zero(float32))
      for c in 0 ..< DkChunks:
        rK.load(kSlice, tvmK, thrID, (j, c))
        mmaQK.atom.gemm_tile(rS, rQ[c], rK, rS)
      for i in 0 ..< sLayout.size():
        rS[i] = rS[i] * qkScale

      # boundary pass: ragged key tails and the causal diagonal block
      # per element. An additive mask makes every tile a boundary tile.
      let
        raggedTile = (j + 1) * NT > k.shape[2]
        boundary = raggedTile or (Causal and j == diagTile) or Masked
      if boundary:
        for i in 0 ..< sLayout.size():
          let (relM, relN) = crd2idx(sCrdMap, i)
          let
            qAbs = qTile * MT + baseM + relM
            kAbs = j * NT + baseN + relN
          if kAbs >= k.shape[2] or (Causal and kAbs > qAbs):
            rS[i] = NegBigF32
        when Masked:
          var rMask: tile(float32, sLayout)
          rMask.load(maskSlice, tvmS, thrID, (qTile, j), padWith = NegBigF32)
          for i in 0 ..< sLayout.size():
            rS[i] = rS[i] + rMask[i] * 1.4426950408889634'f32

      # online softmax: one predicated full-subgroup combine per
      # (row repetition, atom row) reduces the fragment's running row max.
      # Row sums defer. Rescale factors are lane-local.
      for ro in 0 ..< Mo:
        let row = atomM * ro + mLane
        var partial = NegBigF32
        for i in 0 ..< sLayout.size():
          let (relM, relN) = crd2idx(sCrdMap, i)
          if baseM + relM == row:
            partial = max(partial, rS[i])
        for r0 in 0 ..< atomM:
          let contrib = if mLane == r0: partial else: NegBigF32
          let rowMax = reduceMax(contrib)
          if mLane == r0:
            let
              mNew = max(mState[ro], rowMax)
              corr = exp2(mState[ro] - mNew)
            mState[ro] = mNew
            lPart[ro] = lPart[ro] * corr
            for c in 0 ..< DvChunks:
              for i in 0 ..< oLayout.size():
                let (relM, relN) = crd2idx(sCrdMap, i)
                if baseM + relM == row:
                  rOut[c][i] = rOut[c][i] * corr
            for i in 0 ..< sLayout.size():
              let (relM, relN) = crd2idx(sCrdMap, i)
              if baseM + relM == row:
                rS[i] = exp2(rS[i] - mNew)
            var rowSum = 0.0'f32
            for i in 0 ..< sLayout.size():
              let (relM, relN) = crd2idx(sCrdMap, i)
              if baseM + relM == row:
                rowSum = rowSum + rS[i]
            lPart[ro] = lPart[ro] + rowSum

      # O += P·V, P is the exp'd S fragment narrowed to the operand dtype
      when q.T is float32:
        for c in 0 ..< DvChunks:
          rV.load(vTSlice, tvmV, thrID, (c, j))
          mmaPV.atom.gemm_tile(rOut[c], rS, rV, rOut[c])
      else:
        var rP: tile(q.T, sLayout)
        for i in 0 ..< sLayout.size():
          rP[i] = narrowTo[q.T](rS[i])
        for c in 0 ..< DvChunks:
          rV.load(vTSlice, tvmV, thrID, (c, j))
          mmaPV.atom.gemm_tile(rOut[c], rP, rV, rOut[c])

    # deferred row sums, one predicated full-subgroup combine per
    # (row repetition, atom row).
    var lTot: array[Mo, float32]
    for ro in 0 ..< Mo:
      for r0 in 0 ..< atomM:
        let contrib = if mLane == r0: lPart[ro] else: 0.0'f32
        let total = reduceAdd(contrib)
        if mLane == r0:
          lTot[ro] = total

    # normalize and store. A row whose max never left the sentinel
    # stayed fully masked, its 0/0 writes the torch-parity NaN.
    for c in 0 ..< DvChunks:
      for i in 0 ..< oLayout.size():
        let (relM, relN) = crd2idx(sCrdMap, i)
        let
          ro = (baseM + relM) div atomM
          denom = if mState[ro] == NegBigF32: 0.0'f32 / 0.0'f32 else: lTot[ro]
        rOut[c][i] = rOut[c][i] / denom
      when o.T is float32:
        oSlice.store(rOut[c], tvmO, thrID, (qTile, c))
      else:
        var rOutT: tile(o.T, oLayout)
        for i in 0 ..< oLayout.size():
          rOutT[i] = narrowTo[o.T](rOut[c][i])
        oSlice.store(rOutT, tvmO, thrID, (qTile, c))

proc scaled_dot_product_attention*(mmaQK, mmaPV: static TiledMma, o, q, k, v: distinct TensorView, scale: float32, Dk, Dv: static int) =
  ## Attention with no mask, `o = softmax(scale·Q·Kᵀ)·V`.
  sdpaImpl(mmaQK, mmaPV, o, q, k, v, q, scale, Dk, Dv, Causal = false, Masked = false)

proc scaled_dot_product_attention*(mmaQK, mmaPV: static TiledMma, o, q, k, v: distinct TensorView,
    scale: float32, Dk, Dv: static int, Causal: static bool) =
  ## Attention with causal masking, key > query blocked before the softmax.
  ##
  ## Args:
  ##   - Causal: pass `true` for the lower-triangular mask
  sdpaImpl(mmaQK, mmaPV, o, q, k, v, q, scale, Dk, Dv, Causal, Masked = false)

proc scaled_dot_product_attention*(mmaQK, mmaPV: static TiledMma, o, q, k, v, mask: distinct TensorView, scale: float32, Dk, Dv: static int, Causal: static bool) =
  ## Attention with an additive mask, `softmax(scale·Q·Kᵀ + mask)·V`.
  ##
  ## The mask is a float32 view `(B, Hm, S, S)` with `Hm` in `{1, Hkv}`,
  ## head count 1 broadcasts over the kv heads.
  ##
  ## Args:
  ##   - Causal: `true` applies the lower-triangular mask on top
  sdpaImpl(mmaQK, mmaPV, o, q, k, v, mask, scale, Dk, Dv, Causal, Masked = true)
