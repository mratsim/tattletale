# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Host f64 references and fp32 error bars for the Metal MoE suites.
##
## Every reference runs in float64 over the exact element-dtype widenings,
## with the exact El-round sequence of the form under check:
##
## - Qwen softmax form: f64 dot → El round per logit → f64 exp2 softmax
##   (max-subtract) → lowest-index top-K → p/Σp(selected) · Scale → El round.
## - GLM sigmoid form: f64 sigmoid of the RAW f64 logits, with no El round,
##   then lowest-index top-K and s/(Σs + RouteSumEps) · scale.
## - the merge: the host fp32 sequential slot-order sum with one El
##   RNE round, judged bit-exact, plus the shared-expert gate weight
##   logit judged as the raw f64 dot.
## - the fused forward: the f64 activation members (silu, gelu-tanh)
##   with the exact double El round and the slope bound each member needs.
##
## The bars bound the kernel's fp32 side against each exact f64
## reference:
##
## - the mma chunk walk reassociation errors.
## - the 1-ulp-class fp32 exp2, the divisions, and the El store rounds.

import std/math
import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_router
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_fwd

const U32* = 5.9604644775390625e-8      # 2^-24, the fp32 unit roundoff
const FloorSub* = 2.9802322387695312e-8 # 2^-25, half the fp16 subnormal ulp
                                        # and the rounding floor at tiny outputs
const Log2e64* = 1.4426950408889634     # log2(e) in f64, matching the reference exp2 form

proc uStepF64*(El: typedesc[float16]): float64 = 0.00048828125 # 2^-11, the fp16 grid step
proc uStepF64*(El: typedesc[bfloat16]): float64 = 0.00390625   # 2^-8, the bf16 grid step

proc widenF64*[El](v: El): float64 =
  ## Exact widening of an element-dtype value into f64.
  v.to(float32).float64

func f16BitsRNE(v: float32): uint16 =
  ## Round-to-nearest-even bits of an fp32 value as fp16.
  ## Covers subnormal targets and the overflow-to-inf boundary.
  let u = cast[uint32](v)
  let sign = uint16(u shr 16) and 0x8000'u16
  let exp32 = int((u shr 23) and 0xFF'u32)
  let man32 = u and 0x007FFFFF'u32
  if exp32 == 255:
    if man32 != 0:
      return sign or 0x7E00'u16
    return sign or 0x7C00'u16
  # fp16 subnormals: value = q · 2^-24 for a whole q, where q comes
  # from the significand S = 0x800000 | man32 as q = S >> (14 - e)
  # with e = exp32 - 112. A q of 2^10 is the min normal value 2^-14,
  # encoded directly by the subnormal bit pattern 0x0400.
  let e = exp32 - 112
  if e >= 31:
    return sign or 0x7C00'u16
  if e <= 0:
    if e < -10:
      # q < 0.5: rounds to zero.
      return sign
    let s = 0x00800000'u32 or man32
    var q = s shr (14 - e)
    let roundBit = (s shr (13 - e)) and 1
    let sticky = (s and ((1'u32 shl (13 - e)) - 1)) != 0
    if roundBit == 1 and (sticky or (q and 1) == 1):
      q += 1
    return sign or uint16(q)
  var man = man32 shr 13
  let roundBit = (man32 shr 12) and 1
  let sticky = (man32 and 0x00000FFF'u32) != 0
  if roundBit == 1 and (sticky or (man and 1) == 1):
    man += 1
    if man == 0x0400:
      man = 0
      return sign or uint16((e + 1) shl 10)
  return sign or uint16(e shl 10) or uint16(man)

proc narrowEl*[El](v: float64): El =
  ## One round-to-nearest-even round of an f64 value into the element dtype,
  ## the reference's El store round.
  ## Returns the El-typed rounded value.
  ##
  ## fp16 rounds bit-level: the host `toFp16` fallback returns garbage
  ## for subnormal-range fp32 inputs, with a shift count that wraps
  ## beyond the 32-bit word.
  ##
  ## The bf16 fallback rounds raw bits at a fixed shift and is correct
  ## over the whole range, f32 subnormals included.
  when El is float16:
    float16(f16BitsRNE(v.float32))
  else:
    v.float32.to(El)

proc elRoundF64*[El](l: float64): float64 =
  ## Returns one El round of an f64 logit, modeled the way the kernel
  ## rounds its fp32 accumulator value: f64 → fp32 → El → fp32,
  ## carried back in f64.
  narrowEl[El](l).to(float32).float64

proc rawLogitsF64*[El](x, w: ptr UncheckedArray[El], T, H, E: int): seq[float64] =
  ## Returns: per token per expert f64 dot products over the exact
  ## element-dtype widenings, in the exact accumulation order.
  result = newSeq[float64](T * E)
  for t in 0 ..< T:
    for e in 0 ..< E:
      var acc = 0.0'f64
      for k in 0 ..< H:
        acc += x[t * H + k].widenF64() * w[e * H + k].widenF64()
      result[t * E + e] = acc

proc gateDotF64*[El](x, g: ptr UncheckedArray[El], H: int, tok: int): float64 =
  ## Returns: the exact f64 dot of token `tok`'s activation values
  ## against the shared-gate weight vector, the shared-gate weight logit reference.
  for k in 0 ..< H:
    result += x[tok * H + k].widenF64() * g[k].widenF64()

proc sumAbsF64*[El](x, w: ptr UncheckedArray[El], H: int, xOff, wOff: int): float64 =
  ## Returns: Σ|x_k · w_k| over one dot product's terms in f64,
  ## the reassociation bar's magnitude input.
  for k in 0 ..< H:
    result += abs(x[xOff + k].widenF64() * w[wOff + k].widenF64())

proc reassocDelta*(H: int, sumAbs: float64): float64 =
  ## Returns: a sound fp32 reassociation bound between the kernel's
  ## 16-wide-chunk mma walk and the exact f64 value at one logit.
  ## Covers the chunk walk's 15 intra-chunk terms, the H div 16 − 1
  ## cross-chunk terms, and the mma intra-chunk order errors.
  (float64(H + H div 16 + 28) * U32) * sumAbs

proc elSlack*(uStep: float64, l: float64): float64 =
  ## Returns: one El grid step's rounding slack at logit l,
  ## counting each side's El value within half an ulp of its fp32 operand,
  ## giving two per pair comparison.
  2.0 * uStep * abs(l)

proc pairTieBound*(uStep: float64, H: int, sumAbsA, sumAbsB: float64, lA, lB: float64): float64 =
  ## Returns: the tie-region bound for one pair of El logits.
  ## Computed as the sum of both reassociation deltas and their El
  ## grid slack. No legitimate swap exceeds it.
  reassocDelta(H, sumAbsA) + reassocDelta(H, sumAbsB) +
    elSlack(uStep, lA) + elSlack(uStep, lB)

proc softmaxF64*(logits: seq[float64], E: int): seq[float64] =
  ## Returns: exp2-form softmax numerators over one token's
  ## El-rounded logits, with the staged max subtracted and left unnormalized.
  ## The callers renormalize over their own sets.
  result = newSeq[float64](E)
  var lm = float64(finiteFloor(float32))
  for e in 0 ..< E:
    lm = max(lm, logits[e])
  for e in 0 ..< E:
    result[e] = exp2((logits[e] - lm) * Log2e64)

proc topKF64*(p: seq[float64], E, K: int): seq[int32] =
  ## Returns: top-K expert ids by descending p, with the lowest expert
  ## id on equal scores. Selection sort runs over the full id order,
  ## with ids distinct by construction.
  var order = newSeq[int32](E)
  for e in 0 ..< E:
    order[e] = int32(e)
  result = newSeq[int32](K)
  for slot in 0 ..< K:
    var best = slot
    for i in (slot + 1) ..< E:
      if p[order[i]] > p[order[best]] or
          (p[order[i]] == p[order[best]] and order[i] < order[best]):
        best = i
    swap(order[slot], order[best])
    result[slot] = order[slot]

proc renormWeightsF64*(p: seq[float64], sel: seq[int32], K: int, Scale: float32): seq[float64] =
  ## Renormalize the selected probs over the selected set, scale, in f64.
  var sumSel = 0.0'f64
  for i in 0 ..< K:
    sumSel += p[sel[i]]
  result = newSeq[float64](K)
  for slot in 0 ..< K:
    result[slot] = p[sel[slot]] / sumSel * Scale.float64

proc weightBar*(uStep: float64, w: float64, E, K: int): float64 =
  ## Returns: the bar for one El-rounded routing weight,
  ## covering the score-chain rounding sites
  ## (softmax denominator, 1-ulp-class exp2, division, K-term renorm sum),
  ## one RNE per side, and the subnormal floor.
  uStep * abs(w) + float64(E + K + 4) * U32 * abs(w) + FloorSub

proc sigmoidScoresF64*(rawLogits: seq[float64], eCount: int): seq[float64] =
  ## Returns: the sigmoid of the RAW f64 logits with no El round,
  ## where experts at or beyond `eCount` take the fp32 finite floor,
  ## the value the kernel stages for its ragged tail.
  result = newSeq[float64](rawLogits.len)
  for e in 0 ..< rawLogits.len:
    if e < eCount:
      result[e] = 1.0 / (1.0 + exp2(-rawLogits[e] * Log2e64))
    else:
      result[e] = float64(finiteFloor(float32))

proc sigmoidWeightsF64*(s: seq[float64], sel: seq[int32], K: int, scale: float32): seq[float64] =
  ## Returns: the sigmoid form's weights over the selected set in f64,
  ## with s / (Σs + RouteSumEps) · scale and no El round.
  var sumSel = 0.0'f64
  for i in 0 ..< K:
    sumSel += s[sel[i]]
  sumSel += RouteSumEps.float64
  result = newSeq[float64](K)
  for slot in 0 ..< K:
    result[slot] = s[sel[slot]] / sumSel * scale.float64

proc sigmoidWeightBar*(w: float64, K: int, slopeDeltaSum, slopeDeltaE: float64): float64 =
  ## Returns: the bar for one fp32 sigmoid routing weight against
  ## the f64 reference, covering:
  ##
  ## - the fp32 score chain of the sigmoid: the K-term sum, the division,
  ##   and the scaling.
  ## - the logit reassociation under a sigmoid slope of at most 1/4,
  ##   and the normalization.
  ##
  ## `slopeDeltaE` is the slot's own logit delta and `slopeDeltaSum`
  ## the selected set's logit errors summed.
  (2.0 * float64(K + 4) * U32) * abs(w) +
    0.25 * (slopeDeltaE + slopeDeltaSum) * abs(w) + FloorSub

proc sigmoidScoreTieBound*(reassocA, reassocB: float64): float64 =
  ## Returns: the tie-region bound for one pair of fp32 sigmoid scores.
  ## The 24 ulps per side bound the fp32 score chain:
  ##
  ## - the Log2e multiply's rounding, amplified inside exp2 by the exponent magnitude.
  ## - exp2's own 1-ulp-class rounding and the divide.
  ##
  ## Both experts' logit reassociation propagates through the sigmoid
  ## slope of at most 1/4. Two scores closer than this can tie or invert
  ## in fp32, so the selection order may swap, with saturated scores
  ## collapsing onto 1.0 as the wide end of this region.
  2.0 * 24.0 * U32 + 0.25 * (reassocA + reassocB)

proc sharedGateBar*(H: int, sumAbs: float64): float64 =
  ## Returns: the bar for the raw fp32 shared-gate weight logit against
  ## the exact f64 dot. The 2·H factor counts one fp32 unit roundoff
  ## per multiply and per add over the H-term 16-wide-chunk walk,
  ## scaled by the sum of the absolute term values. The store is raw.
  2.0 * float64(H) * U32 * sumAbs + FloorSub

# ─── Decode reference chain (routed + shared slots) ───────────────────

const SiluSlopeMax* = 1.125 # 9/8, an upper bound on silu's first derivative (true max ≈ 1.0998)

proc dotOffF64*[El](a, b: ptr UncheckedArray[El], n, aOff, bOff: int): float64 =
  ## Returns: the exact f64 dot of `n` element-dtype products at the given
  ## offsets over the exact widenings, in the exact accumulation order.
  for k in 0 ..< n:
    result += a[aOff + k].widenF64() * b[bOff + k].widenF64()

proc siluF64*(g: float64): float64 =
  ## Returns: silu in f64 as g / (1 + exp2(−g·log2e)), matching the kernel's fp32 silu.
  g / (1.0 + exp2(-g * Log2e64))

proc partialBarF64*(N: int, sumAbsDown, w, wantP: float64): float64 =
  ## Returns: the bar for one fp32 partial value against w times the f64
  ## down dot over the kernel's own El h values.
  ## The bar covers the down walk's mma reassociation
  ## scaled by the routing weight, plus the rounding of the weight multiply.
  abs(w) * reassocDelta(N, sumAbsDown) + U32 * abs(wantP) + FloorSub

proc gateValBarF64*(H: int, uStep, sumAbsGate, gateVal64: float64): float64 =
  ## Returns: the error bound of the El-rounded sigmoid gate value against
  ## the f64 sigmoid of the exact logit: the logit reassociation through
  ## the sigmoid slope of at most 1/4 plus one fp32 sigmoid evaluation
  ## and one El round.
  0.25 * reassocDelta(H, sumAbsGate) +
    4.0 * U32 * abs(gateVal64) + uStep * abs(gateVal64)

proc sharedPartialBarF64*(I, H: int, uStep, sumAbsDown, sumAbsGate, gateVal64, d64, wantP: float64): float64 =
  ## Returns: the bar for one fp32 shared partial value. The bar covers
  ## the sigmoid gate value's error (see `gateValBarF64`) as it propagates
  ## through the down dot plus the down walk's mma reassociation
  ## scaled by the sigmoid gate value, and the rounding of the weight multiply.
  abs(d64) * gateValBarF64(H, uStep, sumAbsGate, gateVal64) +
    abs(gateVal64) * reassocDelta(I, sumAbsDown) + U32 * abs(wantP) + FloorSub

# ─── Fused forward reference chain (activation members) ───────────────

const GeluSlopeMax* = 1.1291
  ## An upper bound on the gelu-tanh form's first derivative
  ## (numeric scan: true max ≈ 1.1290 at x ≈ 1.4185).

proc geluTanhF64*(g: float64): float64 =
  ## Returns: the gelu-tanh form in f64 through the kernel's exp2 tanh identity,
  ## with s = InvSqrt2Pi·(g + GeluCoef·g³), th = 1 − 2/(exp2(2s·log2e) + 1),
  ## a = 0.5·g·(1 + th).
  let s = InvSqrt2Pi.float64 * (g + GeluCoef.float64 * g * g * g)
  let th = 1.0 - 2.0 / (exp2(2.0 * s * Log2e64) + 1.0)
  0.5 * g * (1.0 + th)

proc hValueF64*[El](g64, u64: float64, act: MoeAct = maSilu): El =
  ## Returns: one h value of the eager activation-mul chain in the exact
  ## El-round sequence, with the activation member `act` selected:
  ##
  ## - the activation in f64
  ## - one El round
  ## - the fp32-carried activation value times the f64 up value
  ## - one El round at the store
  let a64 = case act
    of maSilu: siluF64(g64)
    of maGeluTanh: geluTanhF64(g64)
  let aEl64 = narrowEl[El](a64).to(float32).float64
  narrowEl[El](aEl64 * u64)

proc hBarF64*(H: int, uStep, sumAbsG, sumAbsU, a64, u64, h64: float64, actSlope: float64 = SiluSlopeMax): float64 =
  ## Returns: the bar for one El h value against the f64 activation chain,
  ## covering:
  ##
  ## - the g/u mma reassociation, through the activation's slope of at most
  ##   `actSlope` for g and multiplicatively for u.
  ## - the fp32 activation evaluation: the Log2e multiply, one 1-ulp-class exp2 and the divide
  ## - both El rounds: the activation round propagated through the up operand,
  ##   and the product round at the store.
  abs(u64) * (actSlope * reassocDelta(H, sumAbsG) +
      4.0 * U32 * abs(a64) + uStep * abs(a64)) +
    abs(a64) * reassocDelta(H, sumAbsU) +
    U32 * abs(h64) + uStep * abs(h64) + FloorSub
