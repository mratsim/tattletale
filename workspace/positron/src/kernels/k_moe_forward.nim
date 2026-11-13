# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Token dispatch of the MoE layer: the router's decision moves the tokens
## into per-expert slot blocks and the layer output sums them back out.
## - moe_dispatch_metadata: histogram, padded prefix, slot scatter, inverse map
## - moe_dispatch_gather: activation gather, per-slot expert id resolution
## - moe_combine: weighted un-permute sum into the layer output
##
## The slot capacity: moeCapacityBound(t, topk, e), the padded blocks tile-align
## with the expert contraction's m tiles.

import workspace/crucible/src/crucible_gpu_builtins
import workspace/ceramic/src/tile_algebra
import workspace/positron/src/kernels/k_moe_router {.all.}

const MoEDispatchPadM = 32
  ## Slot-block multiple per expert: expert slot blocks start tile-aligned with grouped_gemm's m tiles.

const MoEMaxExperts = 1024
  ## Expert count bound of the metadata kernel's histogram: 4 KB of threadgroup
  ## memory at this bound. Covers the largest deployed expert count, Kimi K3's 1024.

func moeCapacityBound(t, topk, e: int): int =
  result = t * topk + (MoEDispatchPadM - 1) * e

const MoECombineOutputs = 4

proc moe_dispatch_metadata(packedOffsets, sortedPairs, slotOfPair, ids: distinct TensorView, topk: int) =
  ## Histogram, padded prefix, slot scatter and inverse map, one launch, one token per lane.
  let
    totalT = ids.shape[0]
    e = packedOffsets.shape[0]
    slots = totalT.ceil_div(getLaneWidth())
  for slotID in countup(getLocalTileID(slots), slots - 1, getGridSize()):
    let t = slotID * getLaneWidth() + getLocalThreadID()
    if t < totalT:
      var counts: array[MoEMaxExperts, int]
      for g in 0 ..< e:
        counts[g] = 0
      var seen: array[MoERouterMaxTopk, int]
      # One combined scan over the (token, slot) pairs in ascending order:
      # every pair counts into the histogram, `seen` records this token's slot
      # ranks ahead of the expert's counter increment
      for tp in 0 ..< totalT:
        for ip in 0 ..< topk:
          let g: int = ids[tp, ip]
          if g >= 0:
            if tp == t:
              seen[ip] = counts[g]
            counts[g] += 1
      var acc = 0
      for g in 0 ..< e:
        acc += counts[g].round_multiple_up(MoEDispatchPadM)
        packedOffsets[g] = int32(acc)
      # Pair slots for this token: the slot is the expert's block start plus the rank
      for i in 0 ..< topk:
        let g: int = ids[t, i]
        if g >= 0:
          let
            c = counts[g]
            offG: int = packedOffsets[g]
            start = offG - c.round_multiple_up(MoEDispatchPadM)
            slot = start + seen[i]
          sortedPairs[slot] = int32(t)
          slotOfPair[t, i] = int32(slot)
        else:
          slotOfPair[t, i] = int32(-1)
      # Padding tails of the experts this token selects: all-(-1) writes,
      # redundant across the token's slots and across tokens, never over a real slot
      for i in 0 ..< topk:
        let g: int = ids[t, i]
        if g >= 0:
          let
            c = counts[g]
            padded = c.round_multiple_up(MoEDispatchPadM)
            offG: int = packedOffsets[g]
            start = offG - padded
          for s in start + c ..< start + padded:
            sortedPairs[s] = int32(-1)

proc moe_dispatch_gather(packedIds, packedX, x, sortedPairs, packedOffsets: distinct TensorView) =
  ## Activation gather and per-slot expert id resolution, one launch:
  ## Scatter and gather the padded slots, one launch, one slot per lane.
  let
    k = x.shape[1]
    cap = sortedPairs.shape[0]
    e = packedOffsets.shape[0]
    slots = (cap * k).ceil_div(getLaneWidth())
  for slotID in countup(getLocalTileID(slots), slots - 1, getGridSize()):
    let p = slotID * getLaneWidth() + getLocalThreadID()
    if p < cap * k:
      let
        slot = p div k
        kk = p - slot * k
        src: int = sortedPairs[slot]
      if src >= 0:
        packedX[slot, kk] = x[src, kk]
        if kk == 0:
          var
            lo = 0
            hi = e - 1
          while lo < hi:
            let
              mid = (lo + hi) div 2
              offMid: int = packedOffsets[mid]
            if offMid > slot:
              hi = mid
            else:
              lo = mid + 1
          packedIds[slot] = int32(lo)
      else:
        if kk == 0:
          packedIds[slot] = int32(-1)

proc moe_combine(dst, dPacked, slotOfPair, weights: distinct TensorView, topk: int) =
  ## Weighted un-permute sum, one launch, one threadgroup per (token, n-block),
  ## dst(t, n) = sum over ascending i of weights(t, i) · dPacked[slotOfPair(t, i), n].
  let
    tokens = dst.shape[0]
    n = dst.shape[1]
    nBlocks = n.ceil_div(getLaneWidth() * MoECombineOutputs)
    totalTiles = tokens * nBlocks
  for tileID in countup(getLocalTileID(totalTiles), totalTiles - 1, getGridSize()):
    let
      t = tileID div nBlocks
      n0 = (tileID mod nBlocks) * (getLaneWidth() * MoECombineOutputs) +
        getLocalThreadID() * MoECombineOutputs
    if n0 < n:
      var sums: array[MoECombineOutputs, float32]
      for e in 0 ..< MoECombineOutputs:
        sums[e] = 0
      for i in 0 ..< topk:
        let s: int = slotOfPair[t, i]
        if s >= 0:
          # Scale-at-load: the routing weight multiplies each packed slot value,
          # per-output sums accumulate in fp32 over ascending i
          let w = to(weights[t, i], float32)
          for e in 0 ..< MoECombineOutputs:
            # Ragged N edge: in-range elements load unguarded
            if n0 + e < n:
              sums[e] += w * to(dPacked[s, n0 + e], float32)
      for e in 0 ..< MoECombineOutputs:
        if n0 + e < n:
          dst[t, n0 + e] = to(sums[e], dst.T)
