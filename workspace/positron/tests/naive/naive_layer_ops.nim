# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Naive stage references for the fused GDN decoder layer composition
## (the qwen35_moe mega kernel's reference side), the stage ops the naive tier did not yet carry:
##
## the bias-one RMSNorm with residual add, the q/k l2 normalization, the causal conv + silu step,
## the recurrence values, the dense GEMV, the softmax top-K router and the MoE activation chain.
##
## Storage contract, every proc here:
##
## | rule         | behavior                                                                                                                                                                                             |
## | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | storage      | family-dtype (bf16 or fp16) operands and results are stored as their uint16 bit patterns (the naive tier's convention, `naive_tensors`), widened exactly to fp32 for arithmetic                      |
## | accumulation | fp32 sequential over the row, one RNE bf16 round at each storage handoff the kernel chain rounds at                                                                                                  |
## | rsqrt        | the fp32 rsqrt is the correctly-rounded `1.0 / sqrt(x)`, the Metal approximate `rsqrt` builtin a different rounding class, that difference is a composition-band item, judged by the comparison tier |

import std/math
import naive_tensors
from naive_grouped_mm import GmmFamily, gmmBf16, gmmF16, gmmWiden, gmmRoundEl

# Host libm log1p under its C float32 spelling, std/math spells no log1p.
proc log1p(x: float32): float32 {.importc: "log1pf", header: "<math.h>", noSideEffect.}

const Log2E* = 1.4426950408889634'f32

func bf16Round*(x: float32): uint16 =
  ## Returns the bf16 bit pattern of an fp32 value, round-to-nearest-even.
  f32ToBf16(x)

func softplus*(x: float32): float32 =
  ## Softplus in the ATen `softplus(x, 1, 20)` shape, linear past the threshold, `log(1 + exp(x))` under it.
  if x > 20.0'f32: x else: log1p(exp(x))

func sigmoid*(x: float32): float32 =
  ## Returns `1 / (1 + exp(-x))` in fp32.
  1.0'f32 / (1.0'f32 + exp(-x))

func famWiden*(fam: GmmFamily, h: uint16): float32 =
  ## Returns the exact fp32 widening of a family-dtype bit pattern.
  gmmWiden(fam, h)

func famRound*(fam: GmmFamily, x: float32): uint16 =
  ## Returns the family-dtype round-to-nearest-even bit pattern of an fp32 value.
  gmmRoundEl(fam, x)

# ─── Norm stages ─────────────────────────────────────────────────────

proc naiveRmsNormRes*(xPrev, rPrev, w: seq[uint16]; H: int; eps: float32;
    fam: GmmFamily = gmmBf16): tuple[stream, normed: seq[uint16]] =
  ## One bias-one RMSNorm pass over the residual add, both the decoder layer's
  ## stage 1 and stage 11 (the same proc for each norm site).
  ##
  ## Returns:
  ##
## | output    | value                                             |
## | --------- | ------------------------------------------------- |
## | stream    | s[e] = bf16(x[e] + r[e]), the new residual stream |
## | acc       | sum_e widen(s[e])², fp32 serial over the row      |
## | rstd      | 1/sqrt(acc/H + eps), fp32, no round               |
## | normed[e] | bf16(widen(s[e])·rstd·(widen(w[e]) + 1))          |
  ##
  ## Example (H = 2, exact small values)
  ## x = [1.0, 0.0], r = [0.0, 0.0], w = [0.0, 0.0], eps = 0
  ## gives s = [1.0, 0.0], acc = 1.0, rstd = sqrt(2), normed = [sqrt(2), 0.0] up to the store's bf16 round.
  doAssert xPrev.len == H and rPrev.len == H and w.len == H
  result.stream = newSeq[uint16](H)
  result.normed = newSeq[uint16](H)
  var acc = 0.0'f32
  for e in 0 ..< H:
    let s = famRound(fam, famWiden(fam, xPrev[e]) + famWiden(fam, rPrev[e]))
    result.stream[e] = s
    acc += famWiden(fam, s) * famWiden(fam, s)
  let rstd = 1.0'f32 / sqrt(acc / float32(H) + eps)
  for e in 0 ..< H:
    result.normed[e] = famRound(fam, 
      famWiden(fam, result.stream[e]) * rstd * (famWiden(fam, w[e]) + 1.0'f32))

proc naiveL2NormRow*(x: seq[uint16]; cols: int; fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One l2-normalized row, the q/k normalization's rounding pipeline.
  ##
  ## Returns:
  ##
## | part   | value                                                                |
## | ------ | -------------------------------------------------------------------- |
## | acc    | sum_c El(widen(x[c])²), each square rounds to the family dtype first |
## | sumFam | El(acc), the sum's own round at the model's `.sum` output dtype      |
## | inv    | El(1/sqrt(El(widen(sumFam) + 1e-6)))                                 |
## | out    | out[c] = El(widen(x[c])·widen(inv))                                  |
  ##
  ## Recorded-chain rounding, none of it belongs to the mathematical l2 norm:
  ## - each square rounds to the family dtype elementwise
  ## - the fp32 sum rounds to the family dtype at the model's `.sum` output dtype
  ## - the eps add rounds again before the reciprocal
  doAssert x.len == cols
  var acc = 0.0'f32
  for c in 0 ..< cols:
    let xi = famWiden(fam, x[c])
    acc += famWiden(fam, famRound(fam, xi * xi))
  let sumBf = famRound(fam, acc)
  let inv = famRound(fam, 1.0'f32 /
    sqrt(famWiden(fam, famRound(fam, famWiden(fam, sumBf) + 1.0e-6'f32))))
  result = newSeq[uint16](cols)
  for c in 0 ..< cols:
    result[c] = famRound(fam, famWiden(fam, x[c]) * famWiden(fam, inv))

# ─── Convolutions and projections ────────────────────────────────────

proc naiveDenseLinear*(x: seq[uint16]; w: seq[uint16]; N, K: int;
    fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One row's dense projection, the GEMV form of an (N, K) row-major weight.
  ##
  ## Returns:
  ## - out[n] = bf16(sum_k widen(x[k])·widen(w[n·K + k])), fp32 sequential
  ##
  ## The kernel's 16-wide mma chunk chain reassociates this sum, the reassociation
  ## budget belongs to the comparison tier.
  result = newSeq[uint16](N)
  for n in 0 ..< N:
    var acc = 0.0'f32
    for k in 0 ..< K:
      acc += famWiden(fam, x[k]) * famWiden(fam, w[n * K + k])
    result[n] = famRound(fam, acc)

proc naiveCausalConvSiluStep*(convW: seq[uint16]; ring: var seq[uint16];
    xCol: seq[uint16]; ConvDim, kernel: int; fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One decode conv step over `ConvDim` channels at kernel width `kernel`,
  ## the ring carrying the `kernel - 1` history taps.
  ##
  ## Returns:
  ## - acc[c] = widen(convW[c·kernel + j])·widen(ring[c·(kernel-1) + j]) for j < kernel-1
  ##   plus widen(convW[c·kernel + kernel-1])·widen(xCol[c])
  ## - out[c] = El(silu(El(acc[c])))
  ## - ring[c] is shifted down one slot in place, xCol[c] the newest tap
  ##
  ## The tapped dot rounds once at the family dtype, the silu runs in fp32 over
  ## the widened tapped value, one family-dtype round at the output, per channel
  ## independent of its neighbors.
  let taps = kernel - 1
  doAssert convW.len == ConvDim * kernel
  doAssert ring.len == ConvDim * taps and xCol.len == ConvDim
  result = newSeq[uint16](ConvDim)
  for c in 0 ..< ConvDim:
    var acc = 0.0'f32
    for j in 0 ..< taps:
      acc += famWiden(fam, convW[c * kernel + j]) *
        famWiden(fam, ring[c * taps + j])
    acc += famWiden(fam, convW[c * kernel + taps]) * famWiden(fam, xCol[c])
    let tapped = famRound(fam, acc)
    result[c] = famRound(fam, 
      famWiden(fam, tapped) / (1.0'f32 + exp(-famWiden(fam, tapped))))
  for c in 0 ..< ConvDim:
    for j in 0 ..< taps - 1:
      ring[c * taps + j] = ring[c * taps + j + 1]
    ring[c * taps + taps - 1] = xCol[c]

# ─── Recurrence gates ────────────────────────────────────────────────

proc naiveGdnGates*(aRow, bRow, dtBias: seq[uint16]; aLog: seq[float32]; H: int;
    fam: GmmFamily = gmmBf16):
    tuple[g: seq[float32], beta: seq[uint16]] =
  ## Recurrence values over the H value heads, one head per index.
  ##
  ## Returns:
  ## - g[h]    = -exp(A_log[h])·softplus(widen(a[h]) + widen(dtBias[h]))
  ##   fp32 end to end, no round
  ## - beta[h] = El(sigmoid(widen(b[h])))
  doAssert aRow.len == H and bRow.len == H and dtBias.len == H and aLog.len == H
  result.g = newSeq[float32](H)
  result.beta = newSeq[uint16](H)
  for h in 0 ..< H:
    let x = famWiden(fam, aRow[h]) + famWiden(fam, dtBias[h])
    result.g[h] = -exp(aLog[h]) * softplus(x)
    result.beta[h] = famRound(fam, sigmoid(famWiden(fam, bRow[h])))

# ─── The MoE router and expert activation ────────────────────────────

proc naiveSoftmaxTopKRouter*(x: seq[uint16]; routerW: seq[uint16];
    E, H, K: int; scale: float32; fam: GmmFamily = gmmBf16):
    tuple[ids: seq[int32], w: seq[float32]] =
  ## One token's top-K expert ids and routing weights, the softmax form.
  ##
  ## Returns:
  ##
## | step   | value                                                                 |
## | ------ | --------------------------------------------------------------------- |
## | logits | logits[e] = bf16(sum_k widen(x[k])·widen(routerW[e·H + k])), fp32 dot |
## | p      | softmax over the widened logits, fp32                                 |
## | top-K  | by score, the lowest index on a tie                                   |
## | w      | w[slot] = El(p[id]/sum(top-K p)·scale), renormalized over             |
## |        | the selected set only, the model's routeToExperts contract            |
  ##
  ## Example (E = 4, K = 2, exact small values, all logits distinct):
  ##   logits [3.0, 1.0, 2.0, 0.0] → p ∝ [e³, e¹, e², 1] → ids [0, 2],
  ##   w = [e³/(e³+e²), e²/(e³+e²)], each rounded to the family dtype.
  doAssert routerW.len == E * H
  var logits = newSeq[float32](E)
  for e in 0 ..< E:
    var acc = 0.0'f32
    for k in 0 ..< H:
      acc += famWiden(fam, x[k]) * famWiden(fam, routerW[e * H + k])
    logits[e] = famWiden(fam, famRound(fam, acc))
  var p = newSeq[float32](E)
  for e in 0 ..< E:
    p[e] = exp(logits[e] - max(logits))
  result.ids = newSeq[int32](K)
  result.w = newSeq[float32](K)
  var used = newSeq[bool](E)
  var topSum = 0.0'f32
  var picked = newSeq[int](K)
  for slot in 0 ..< K:
    var best = -1
    var bestP = -1.0'f32
    for e in 0 ..< E:
      if not used[e] and (best < 0 or p[e] > bestP):
        bestP = p[e]
        best = e
    used[best] = true
    picked[slot] = best
    result.ids[slot] = int32(best)
    topSum += p[best]
  # renormalize over the selected set only, then one bf16 round per weight
  for slot in 0 ..< K:
    result.w[slot] = famWiden(fam, famRound(fam, p[picked[slot]] / topSum * scale))

proc naiveSiluMulEl*(g, u: float32; fam: GmmFamily = gmmBf16): uint16 =
  ## MoE expert activation element, the mega chain's rounding form over the fp32 g/up accumulator operands.
  ##
  ## Returns:
  ## - El(El(silu(g))·u)
  ##
  ## The silu result rounds at the family dtype first, then the product with the fp32 up
  ## operand rounds once at the store.
  let s = g / (1.0'f32 + exp(-g))
  famRound(fam, famWiden(fam, famRound(fam, s)) * u)

proc naiveSharedGate*(x, sharedGateVecW: seq[uint16]; H: int;
    fam: GmmFamily = gmmBf16): float32 =
  ## One token's shared-expert scalar, the sigmoid of the raw fp32 scalar logit,
  ## one bf16 round, returned widened.
  ##
  ## Returns:
  ## - l32 = sum_k widen(x[k])·widen(sharedGateVecW[k]), fp32 sequential
  ## - the returned value = El(sigmoid(l32))
  doAssert sharedGateVecW.len == H
  var l32 = 0.0'f32
  for k in 0 ..< H:
    l32 += famWiden(fam, x[k]) * famWiden(fam, sharedGateVecW[k])
  famWiden(fam, famRound(fam, sigmoid(l32)))

# ─── The gated RMSNorm and the MoE merge ─────────────────────────────

proc naiveRmsNormGated*(y, z, w: seq[uint16]; Dv: int; eps: float32;
    fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One gated RMSNorm row, the o_norm epilogue's rounding chain.
  ##
  ## Returns:
  ##
## | step     | value                                                                 |
## | -------- | --------------------------------------------------------------------- |
## | rstd     | 1/sqrt(sum_d widen(y[d])²/Dv + eps), fp32 squares, fp32 sum, no round |
## | normed   | normed[d] = El(widen(y[d])·rstd)                                      |
## | weighted | weighted[d] = El(widen(w[d])·widen(normed[d]))                        |
## | out[d]   | El(widen(weighted[d])·silu32(widen(z[d])))                            |
  ##
  ## The squares stay fp32 here (unlike the l2-normalized row's family-dtype squares),
  ## matching the o_norm tile core's fp32 square-and-reduce.
  doAssert y.len == Dv and z.len == Dv and w.len == Dv
  var sumSq = 0.0'f32
  for d in 0 ..< Dv:
    let yv = famWiden(fam, y[d])
    sumSq += yv * yv
  let rstd = 1.0'f32 / sqrt(sumSq / float32(Dv) + eps)
  result = newSeq[uint16](Dv)
  for d in 0 ..< Dv:
    let normed = famRound(fam, famWiden(fam, y[d]) * rstd)
    let weighted = famRound(fam, famWiden(fam, w[d]) * famWiden(fam, normed))
    let g = famWiden(fam, z[d])
    let silu = g / (1.0'f32 + exp(-g))
    result[d] = famRound(fam, famWiden(fam, weighted) * silu)

proc naiveMoeMerge*(partial: seq[float32]; K, H: int; fam: GmmFamily = gmmBf16): seq[uint16] =
  ## One token's MoE merge, the fp32 partial rows summed in slot order
  ## (the shared contribution last), one family-dtype round at the store.
  ##
  ## Returns:
  ## - out[e] = bf16(sum_slot widen(partial[slot·H + e])), slot order
  doAssert partial.len == (K + 1) * H
  result = newSeq[uint16](H)
  for e in 0 ..< H:
    var acc = 0.0'f32
    for slot in 0 .. K:
      acc += partial[slot * H + e]
    result[e] = famRound(fam, acc)
