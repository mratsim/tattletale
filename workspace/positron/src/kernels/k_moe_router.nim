# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## MoE router top-k on Metal: per-token expert selection and routing weights.
##
## - selection scores are scores + bias, bias affects selection only
## - equal scores select the lower expert id, one fixed serial scan, no atomics
## - weights come from the UNBIASED scores at the selected ids: softmax over all E
##   (exp2 form) or sigmoid, renormed over the top-k with an epsilon floor,
##   scaled by `scalingFactor`
## - topk > E: excess slots carry id -1 and weight 0

import workspace/crucible/src/crucible_gpu_builtins
import workspace/ceramic/src/tile_algebra

const MoERouterMaxTopk = 8
  ## Slot count of the router's top-k insertion buffers, `topk` bounded by this constant.
  ## Covers the recorded fleet, 4 to 8 slots, host-asserted, bump alongside a new model.
  ## GLM and gpt-oss run 4, Moonlight 6, DeepSeek, Qwen3.6, North, gemma-4 and Laguna 8.

type MoERoute* = enum
  ## Weights source of the router: the softmax over all E unbiased scores,
  ## or the sigmoid of the unbiased scores.
  moeSoftmax
  moeSigmoid

const MoERenormEps = 1e-20'f32
  ## Renorm denominator floor: w = w / max(sum w, MoERenormEps).

const MoERouterLog2e = 1.4426950408889634'f32
  ## 1/ln(2), folds exp(v - m) into exp2((v - m)·MoERouterLog2e).

proc moe_router_topk*(ids, weights, scores, bias: distinct TensorView, topk, variant: int, scalingFactor: float32) =
  ## Per-token top-k routing, one launch: T threadgroups, (32, 1) lanes, lane 0 runs the serial scan.
  let
    tCount = scores.shape[0]
    e = scores.shape[1]
  for slotID in countup(getLocalTileID(tCount), tCount - 1, getGridSize()):
    let t = slotID
    if getLocalThreadID() == 0:
      var
        selVal: array[MoERouterMaxTopk, float32]
        selIdx: array[MoERouterMaxTopk, int]
        nSel = 0
      # Pass 1, softmax only: unbiased max over E, then a max-shifted exp sum
      var m: float32
      var z: float32
      if variant == ord(moeSoftmax):
        m = to(scores[t, 0], float32)
        for i in 1 ..< e:
          let v = to(scores[t, i], float32)
          if v > m:
            m = v
        for i in 0 ..< e:
          z += exp2((to(scores[t, i], float32) - m) * MoERouterLog2e)
      # Pass 2: top-k insertion over the biased scores. Strict comparisons keep
      # it sorted descending, equal scores select the LOWER id
      for i in 0 ..< e:
        let s = to(scores[t, i], float32) + to(bias[i], float32)
        var insert = nSel < topk
        if not insert and s > selVal[topk - 1]:
          insert = true
        if insert:
          var pos = nSel
          if pos > topk - 1:
            pos = topk - 1
          while pos > 0 and selVal[pos - 1] < s:
            selVal[pos] = selVal[pos - 1]
            selIdx[pos] = selIdx[pos - 1]
            pos -= 1
          selVal[pos] = s
          selIdx[pos] = i
          if nSel < topk:
            nSel += 1
      # Pass 3: unbiased weights at the selected experts, gathered at selIdx.
      # Renorm over the selected set, scale, store. selVal is the weight buffer
      var wSum = 0'f32
      for i in 0 ..< nSel:
        let g = selIdx[i]
        var w: float32
        if variant == ord(moeSoftmax):
          w = exp2((to(scores[t, g], float32) - m) * MoERouterLog2e) / z
        else:
          let v = to(scores[t, g], float32)
          w = 1'f32 / (1'f32 + exp2(-v * MoERouterLog2e))
        selVal[i] = w
        wSum += w
      let denom = if wSum > MoERenormEps: wSum else: MoERenormEps
      for i in 0 ..< nSel:
        ids[t, i] = int32(selIdx[i])
        weights[t, i] = to(selVal[i] / denom * scalingFactor, float32)
      for i in nSel ..< topk:
        ids[t, i] = int32(-1)
        weights[t, i] = 0'f32
