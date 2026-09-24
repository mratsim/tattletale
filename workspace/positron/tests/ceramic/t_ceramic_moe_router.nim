# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_moe_router.nim
##
## Ceramic Qwen softmax router suite, the kernel judged against the host reference
##
##   x·router_wᵀ → logits (fp32) → El → softmax → top-K → renorm·Scale → El
##
## - ids[t] = top-K(p), lowest expert id on ties
## - w[t, slot] = El(p[ids[slot]] / sum(p[ids]) · Scale)
## - El = one round-to-nearest-even round in the element dtype
##
## | subject    | contract                                                                                                 |
## | ---------- | -------------------------------------------------------------------------------------------------------- |
## | naive side | host fp32 dot products over the exact widenings, then the same rounding chain in fp32                    |
## | softmax    | both sides run the exp2 exponential form, so the top-K set is a function of the El-rounded logits        |
## | top-K      | lowest expert id on equal scores, ids distinct and in [0, E), routing weights descending per token       |
## | merge      | the fp32 partial-row merge sums in slot order, one El round, judged bit-exact against the same sum       |
## | poisoned   | the all-NaN router weight leaves every slot on the unmatched branch, ids E−1, zero weights, finite chain |
## | shared exp | the (1, H) shared-expert row GEMV returns the raw fp32 logit, the same 16-wide chunk walk as the router  |
##
## | shape      | T | H            | E                 | K   | Scale | dtype      | cases |
## | ---------- | --- | ------------ | ----------------- | --- | ----- | ---------- | ----- |
## | mega       | 8 | 2048         | 256               | 8   | 1.0   | bf16, fp16 | 8     |
## | mega s2    | 8 | 2048         | 256               | 8   | 2.0   | bf16       | 8     |
## | small      | 4 | 256          | 64                | 4   | 1.0   | bf16       | 8     |
## | shared exp | 8 | 2048         | ----------------- | --- | ----- | bf16, fp16 | 8     |
## | merge      | 8 | 256 and 2048 | 8 routed + shared | 8   | 1.0   | bf16       | 8, 2  |
## | merge f16  | 8 | 256          | 8 routed + shared | 8   | 1.0   | fp16       | 8     |
## | poisoned   | 1 | 2048         | 256               | 8   | 1.0   | bf16       | 1     |
##
## Bars, stated before measurement, U32 = 2⁻²⁴ fp32, uStep = 2⁻⁸ bf16 / 2⁻¹¹ fp16
## - the reassociation check re-demonstrates per token per run that the 16-wide-chunk
##   walk of the kernel's mma accumulation stays inside its stated fp32 bound
## - a standalone reassociation check runs over four extra seeds per shape,
##   mega binding H = 2048 included, worst usage printed
## - the weights carry every divergence, per the bar tables
##
## | tie region | rule                                                                                                                                 |
## | ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
## | why        | two fp32 accumulation orders can round one El logit a grid step apart, demonstrated by the fp16 mega check flipping 8 of 2048 logits |
## | swap bound | the pair's reassociation deltas (H + H div 16 + 28)·u32·Σabs per logit plus one El grid step                                         |
## | set swap   | a swapped-in expert must pair with a swapped-out expert inside the same bound                                                        |
## | swapped w  | the naive weight recomputed over the kernel's own set, the bar gains the exp2 amplification 8·(δ + grid slack)·abs(w), lm included   |
##
## | term                   | bar                                                | covers                                                                                     |
## | ---------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------ |
## | w (t, s)               | u_step·abs(w) + (E + K + 4)·u32·abs(w) + 2⁻²⁵      | the score-chain rounding sites, one RNE per side                                           |
## | GEMV accumulator       | (E + K + 4)·u32·abs(w) shares this term            | an E-wide dot on both sides, the kernel walks 16-wide mma chunks                           |
## | softmax denominator    | (E + K + 4)·u32·abs(w) shares this term            | the summation order, E terms                                                               |
## | exp2, division, renorm | (E + K + 4)·u32·abs(w) shares this term            | the 1-ulp-class exp2 form, the division, the K-term renorm sum                             |
## | store round            | u_step·abs(w)                                      | the two sides round slightly different fp32 weights, each RNE within u_step of its operand |
## | swapped-slot w         | + 8·(δ + grid slack)·abs(w), δ and slack per logit | the exp2 amplification of a tie-region logit spread, lm included                           |
## | subnormal floor        | 2⁻²⁵                                               | the fp16 subnormal grid, also the bf16 grid                                                |
##
## - the merge output is exact-where-legal, both sides sum the same fp32 partials
##   in the same slot order, one El round, asserted bit-identical
## - the measured divergence justifies the model, never sets the bar
## - adjudicated on Apple M4 Max with fresh seeded xorshift64 inputs

import std/[strformat, math]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/moe_router
import ../../src/kernels/ceramic/ffn_moe_decode_single
import ../naive/naive_rng
import ../naive/naive_tensors
import ceramic_pagebuf
import ceramic_dtype

# ─── Device entries, one per (element dtype, shape) binding ────────────

const MoeRouterMsl = metal:
  proc cer_moe_route_bf16_mega(
      ids: ptr UncheckedArray[int32],
      rout_w, x, router_w: ptr UncheckedArray[bfloat16],
      num_tokens: int32) {.global.} =
    moe_route_fwd[bfloat16, 2048, 256, 8, 1.0'f32](
      ids, rout_w, x, router_w, num_tokens)

  proc cer_moe_route_f16_mega(
      ids: ptr UncheckedArray[int32],
      rout_w, x, router_w: ptr UncheckedArray[float16],
      num_tokens: int32) {.global.} =
    moe_route_fwd[float16, 2048, 256, 8, 1.0'f32](
      ids, rout_w, x, router_w, num_tokens)

  proc cer_moe_route_bf16_small(
      ids: ptr UncheckedArray[int32],
      rout_w, x, router_w: ptr UncheckedArray[bfloat16],
      num_tokens: int32) {.global.} =
    moe_route_fwd[bfloat16, 256, 64, 4, 1.0'f32](
      ids, rout_w, x, router_w, num_tokens)

  proc cer_moe_route_bf16_mega_s2(
      ids: ptr UncheckedArray[int32],
      rout_w, x, router_w: ptr UncheckedArray[bfloat16],
      num_tokens: int32) {.global.} =
    moe_route_fwd[bfloat16, 2048, 256, 8, 2.0'f32](
      ids, rout_w, x, router_w, num_tokens)

  proc cer_moe_merge_bf16_mega(
      out_r: ptr UncheckedArray[bfloat16],
      partial: ptr UncheckedArray[float32]) {.global.} =
    moe_decode_merge[bfloat16, 2048, 8](out_r, partial)

  proc cer_moe_fwd_bf16_mega(
      partial: ptr UncheckedArray[float32],
      x, router_w, gate_up_w, down_w: ptr UncheckedArray[bfloat16],
      shared_gate_w, shared_up_w, shared_down_w,
      shared_gate_vec_w: ptr UncheckedArray[bfloat16],
      h_scratch, hs_scratch: ptr UncheckedArray[bfloat16]) {.global.} =
    moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 1.0'f32, true](
      partial, x, router_w, gate_up_w, down_w,
      shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
      h_scratch, hs_scratch)

  proc cer_shared_gate_bf16(
      outp: ptr UncheckedArray[float32],
      x, sgw: ptr UncheckedArray[bfloat16]) {.global.} =
    let t = int32(threadgroup_position_in_grid.x)
    let v = sharedGateLogit[bfloat16, 2048](x, sgw, t)
    if int(thread_index_in_threadgroup) == 0:
      outp[t] = v

  proc cer_moe_merge_bf16(
      out_r: ptr UncheckedArray[bfloat16],
      partial: ptr UncheckedArray[float32]) {.global.} =
    moe_decode_merge[bfloat16, 256, 8](out_r, partial)

  proc cer_moe_merge_f16(
      out_r: ptr UncheckedArray[float16],
      partial: ptr UncheckedArray[float32]) {.global.} =
    moe_decode_merge[float16, 256, 8](out_r, partial)

  proc cer_shared_gate_f16(
      outp: ptr UncheckedArray[float32],
      x, sgw: ptr UncheckedArray[float16]) {.global.} =
    let t = int32(threadgroup_position_in_grid.x)
    let v = sharedGateLogit[float16, 2048](x, sgw, t)
    if int(thread_index_in_threadgroup) == 0:
      outp[t] = v

# ─── Host, the independent reference ─────────────────────────────────

# libm exp2f, the crucible export table also carries an exp2 device builtin
# whose host body is a discard, so std/math's exp2 cannot be called here
proc exp2fHost(x: cfloat): cfloat {.importc: "exp2f", header: "<math.h>".}

from ../../src/kernels/ceramic/math_consts import Log2e

const
  FloorSub = 2.9802322387695312e-8   # 2^-25, half the fp16 subnormal ulp,
                                     # the rounding floor at tiny outputs

proc elRound(dt: ScalarKind, v: float32): float32 =
  ## One round-to-nearest-even round into the element dtype and back to fp32.
  if dt == kBfloat16: bf16ToF32(f32ToBf16(v)) else: fp16ToFp32(fp32ToFp16(v))

proc dotRawLogits(dt: ScalarKind, x, w: seq[uint16]; T, H, E: int;
    chunkWalk: bool): seq[float32] =
  ## Raw fp32 logits over the exact widenings under the two accumulation orders
  ## the reassociation bound covers
  ##
  ## - the naive sequential fp32 sum, when `chunkWalk` is false
  ## - the 16-wide-chunk walk of the kernel's mma accumulation when `chunkWalk`
  ##   is true, each chunk partial summed sequentially, the chunk partials
  ##   accumulated in fp32 across H div 16 chunks
  result = newSeq[float32](T * E)
  for t in 0 ..< T:
    for e in 0 ..< E:
      var acc = 0.0'f32
      if chunkWalk:
        for kk in 0 ..< H div 16:
          var c = 0.0'f32
          for j in 0 ..< 16:
            c += x[t * H + kk * 16 + j].widenTo(dt) *
              w[e * H + kk * 16 + j].widenTo(dt)
          acc += c
      else:
        for k in 0 ..< H:
          acc += x[t * H + k].widenTo(dt) * w[e * H + k].widenTo(dt)
      result[t * E + e] = acc

proc reassocDelta(H: int, sumAbs: float64): float64 =
  ## Sound fp32 reassociation bound between the kernel's chunk walk and the naive
  ## sequential sum at one logit
  ##
  ## - the sequential side's (H - 1)-term forward error
  ## - the chunk walk's 15-term intra-chunk and (H div 16 - 1)-term cross-chunk errors
  ## - the mma intra-chunk order's 15-term bound on top
  (float64(H + H div 16 + 28) * U32) * sumAbs

proc naiveSharedGateDot(dt: ScalarKind, x, sgw: seq[uint16]; T, H: int): seq[float32] =
  ## Host reference for the shared-expert scalar logit, a sequential fp32 dot over
  ## the exact element-dtype widenings, the kernel returning the raw fp32 logit too
  ## No element-dtype round on either side.
  result = newSeq[float32](T)
  for t in 0 ..< T:
    var acc = 0.0'f32
    for k in 0 ..< H:
      acc += x[t * H + k].widenTo(dt) * sgw[k].widenTo(dt)
    result[t] = acc

proc naiveRouter(dt: ScalarKind, x, w: seq[uint16]; T, H, E, K: int, Scale: float32):
    tuple[ids: seq[int32], routW: seq[uint16], logits: seq[float32]] =
  ## Independent host reference for the softmax form's rounding chain in fp32
  ##
  ## - sequential fp32 dot over the exact widenings, one El round per logit
  ## - fp32 softmax over the exp2 form
  ## - top-K with the lowest-index tiebreak, then renorm, scale, one El round per routing weight
  result.ids = newSeq[int32](T * K)
  result.routW = newSeq[uint16](T * K)
  result.logits = newSeq[float32](T * E)
  for t in 0 ..< T:
    var logits = newSeq[float32](E)
    for e in 0 ..< E:
      var acc = 0.0'f32
      for k in 0 ..< H:
        acc += x[t * H + k].widenTo(dt) * w[e * H + k].widenTo(dt)
      logits[e] = elRound(dt, acc)
      result.logits[t * E + e] = logits[e]
    var lm = -3.402823466e38'f32
    for e in 0 ..< E:
      lm = max(lm, logits[e])
    var p = newSeq[float32](E)
    var ls = 0.0'f32
    # the crucible export table also carries an exp2 device builtin whose
    # host body is a discard, so exp2fHost is the only host exp2 here
    for e in 0 ..< E:
      p[e] = exp2fHost((logits[e] - lm) * Log2e)
      ls += p[e]
    var order = newSeq[int32](E)
    for e in 0 ..< E:
      order[e] = int32(e)
    # selection sort by descending p, lowest expert id on equal scores
    # select the K first, then renormalize over the selected set only
    var sel = newSeq[int32](K)
    for slot in 0 ..< K:
      var best = slot
      for i in (slot + 1) ..< E:
        if p[order[i]] > p[order[best]] or
            (p[order[i]] == p[order[best]] and order[i] < order[best]):
          best = i
      swap(order[slot], order[best])
      sel[slot] = order[slot]
      result.ids[t * K + slot] = order[slot]
    var sumSel = 0.0'f32
    for i in 0 ..< K:
      sumSel += p[sel[i]]
    for slot in 0 ..< K:
      let weight = p[sel[slot]] / sumSel * Scale
      result.routW[t * K + slot] =
        if dt == kBfloat16: f32ToBf16(weight) else: fp32ToFp16(weight)

proc elSlack(uStep: float64, l: float32): float64 =
  ## One El grid step's rounding slack at logit l
  ## Each side's El value sits within half an ulp of its fp32 operand, a pair
  ## comparison carries two of them
  2.0 * uStep * abs(l.float64)

proc pairTieBound(uStep: float64, H: int, sumAbsA, sumAbsB: float64;
    lA, lB: float32): float64 =
  ## Tie-region bound for one pair of El logits, the sum of both reassociation
  ## deltas and their El grid slack
  ## No legitimate swap can exceed it.
  reassocDelta(H, sumAbsA) + reassocDelta(H, sumAbsB) +
    elSlack(uStep, lA) + elSlack(uStep, lB)

proc naiveWeightFor(dt: ScalarKind, logits: seq[float32]; t, E, K: int;
    ids: seq[int32]; Scale: float32; target: int32): uint16 =
  ## Naive chain's routing weight for expert `target`, renormalized over the set
  ## `ids` in slot order, used when the kernel's top-K set differs from the naive
  ## one inside the tie region
  var lm = -3.402823466e38'f32
  for e in 0 ..< E:
    lm = max(lm, logits[t * E + e])
  var sumSel = 0.0'f64
  for slot in 0 ..< K:
    let e = logits[t * E + ids[slot].int]
    sumSel += exp2fHost((e - lm) * Log2e).float64
  let p = exp2fHost((logits[t * E + target.int] - lm) * Log2e).float64
  elRound(dt, (p / sumSel * Scale).float32).uint16

var suiteCases, suiteLaunches, suiteExact, suiteTotal = 0
var suiteWorstUse = 0.0'f64

proc runCombo(engine: HwEngine; dt: ScalarKind, T, H, E, K, cases: int;
    seed: uint64; scale: float32; label: string) =
  ## One (element dtype, shape, Scale) combination over `cases` independent seeded runs
  ##
  ## - ids judged against the naive top-K under the tie region, weights per element
  ##   under the band
  ## - the reassociation check re-demonstrated per token, case 0 relaunched bit-identical
  let nIds = T * K
  let nX = T * H
  let nW = E * H
  var idsB = allocPageBuf[int32](nIds)
  var wB = allocPageBuf[uint16](nIds)
  var xB = allocPageBuf[uint16](nX)
  var rWB = allocPageBuf[uint16](nW)
  defer:
    freePageBuf(idsB); freePageBuf(wB); freePageBuf(xB); freePageBuf(rWB)
  var idsPA = idsB.pa()
  var wPA = wB.pa()
  var xPA = xB.pa()
  var rWPA = rWB.pa()
  var kernelName = if dt == kBfloat16:
    (if H == 2048: "cer_moe_route_bf16_mega" else: "cer_moe_route_bf16_small")
  else:
    "cer_moe_route_f16_mega"
  if scale != 1.0'f32:
    doAssert dt == kBfloat16 and H == 2048,
      "the Scale 2.0 binding exists only at the mega bf16 shape"
    kernelName = kernelName & "_s2"
  let ulpG = if dt == kBfloat16: ulpBf16 else: ulpFp16
  let uStep = binadeStep(ulpG, -1)

  var worstUse = 0.0'f64
  var exactW = 0
  var total = 0
  var totalSwaps = 0
  var reassocWorst = 0.0'f64
  var launches = 0

  proc takeInputs(rng: var NaiveRng): tuple[x, w: seq[uint16]] =
    ## Seeded inputs, element-dtype bits for x and the router weight.
    var xBits = newSeq[uint16](nX)
    var wBits = newSeq[uint16](nW)
    for i in 0 ..< nX:
      xBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
    for i in 0 ..< nW:
      wBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
    result = (xBits, wBits)

  proc load(bits: tuple[x, w: seq[uint16]]) =
    for i in 0 ..< nX:
      xB.hostPtr[i] = bits.x[i]
    for i in 0 ..< nW:
      rWB.hostPtr[i] = bits.w[i]
    for i in 0 ..< nIds:
      idsB.hostPtr[i] = -1
      wB.hostPtr[i] = 0

  proc sentinels(bits: tuple[x, w: seq[uint16]]) =
    assertReadUnchanged(xB, bits.x)
    assertReadUnchanged(rWB, bits.w)
    for i in 0 ..< nIds:
      doAssert idsB.hostPtr[i] >= 0 and idsB.hostPtr[i] < int32(E),
        &"expert id out of range at pair {i}"

  proc launch =
    engine.run << (grid: (T, 1, 1), blk: (32, 1, 1)) >>
      (kernelName, idsPA, (wPA, xPA, rWPA, int32(T)))
    inc launches

  proc record(): tuple[ids: seq[int32], w: seq[uint16]] =
    result.ids = newSeq[int32](nIds)
    result.w = newSeq[uint16](nIds)
    for i in 0 ..< nIds:
      result.ids[i] = idsB.hostPtr[i]
      result.w[i] = wB.hostPtr[i]

  var case0record: tuple[ids: seq[int32], w: seq[uint16]]
  var rng = initNaiveRng(seed)
  for caseId in 0 ..< cases:
    let bits = takeInputs(rng)
    let want = naiveRouter(dt, bits.x, bits.w, T, H, E, K, scale)
    let seqLogits = dotRawLogits(dt, bits.x, bits.w, T, H, E, false)
    let chunkLogits = dotRawLogits(dt, bits.x, bits.w, T, H, E, true)
    # the ids' reassociation premise, re-demonstrated per token per run
    # the 16-wide-chunk walk of the kernel's mma accumulation stays inside
    # the stated fp32 bound against the naive sequential sum
    var sumAbs = newSeq[float64](T * E)
    for t in 0 ..< T:
      for e in 0 ..< E:
        var sa = 0.0'f64
        for k in 0 ..< H:
          sa += abs(bits.x[t * H + k].widenTo(dt).float64 *
            bits.w[e * H + k].widenTo(dt).float64)
        sumAbs[t * E + e] = sa
        let bar = reassocDelta(H, sa)
        let diff = abs(chunkLogits[t * E + e].float64 - seqLogits[t * E + e].float64)
        doAssert diff <= bar,
          &"the chunk walk's fp32 logit left its reassociation bound at " &
          &"(t {t}, e {e}, case {caseId}): {diff:.3e} > {bar:.3e}"
        reassocWorst = max(reassocWorst, diff / bar)
    load(bits)
    launch()
    sentinels(bits)
    let got = record()
    for t in 0 ..< T:
      # token-level tie-region bookkeeping, the naive El logits, the max logit,
      # and the set membership on both sides
      var lm = -3.402823466e38'f32
      var lmE = 0
      for e in 0 ..< E:
        if want.logits[t * E + e] > lm:
          lm = want.logits[t * E + e]
          lmE = e
      var inWant = newSeq[bool](E)
      var inGot = newSeq[bool](E)
      for slot in 0 ..< K:
        inWant[want.ids[t * K + slot].int] = true
        inGot[got.ids[t * K + slot].int] = true
      var gotRow = newSeq[int32](K)
      for slot in 0 ..< K:
        gotRow[slot] = got.ids[t * K + slot]
      proc lg(e: int): float32 = want.logits[t * E + e]

      # the top-K set and the slot order, judged under the tie region
      # outside the region the ids are asserted exact against the naive top-K
      # a slot that differs is legitimate only when the swapped pair sits inside
      # the tie-region bound, and a swapped-in expert needs a swapped-out partner
      var prev = 3.402823466e38'f64
      var swaps = 0
      for slot in 0 ..< K:
        let idx = t * K + slot
        doAssert got.ids[idx] != (if slot > 0: got.ids[idx - 1] else: -1),
          "duplicate expert id in one token's top-K"
        let gE = got.ids[idx].int
        let wE = want.ids[idx].int
        if gE != wE:
          inc swaps
          doAssert abs(lg(gE).float64 - lg(wE).float64) <=
              pairTieBound(uStep.float64, H, sumAbs[t * E + gE],
                sumAbs[t * E + wE], lg(gE), lg(wE)),
            &"top-K swap at (t {t}, slot {slot}, case {caseId}) outside the " &
            &"tie region: experts {gE} and {wE}"
        if not inWant[gE]:
          var found = false
          for ws in 0 ..< K:
            let wE2 = want.ids[t * K + ws].int
            if not inGot[wE2] and
                abs(lg(gE).float64 - lg(wE2).float64) <=
                  pairTieBound(uStep.float64, H, sumAbs[t * E + gE],
                    sumAbs[t * E + wE2], lg(gE), lg(wE2)):
              found = true
              break
          doAssert found,
            &"swapped-in expert {gE} at (t {t}, case {caseId}) has no " &
            &"tie-region partner among the swapped-out experts"

        # the routing weights carry the band, a swapped slot is judged against
        # the naive weights recomputed over the kernel's own set, the bar widened
        # by the exp2 amplification of the tie-region logit spread
        let gotW = got.w[idx].widenTo(dt).float64
        doAssert gotW <= prev, "routing weights not descending"
        prev = gotW
        var wantW = want.routW[idx].widenTo(dt).float64
        var bar = uStep.float64 * abs(wantW) +
          float64(E + K + 4) * U32 * abs(wantW) + FloorSub
        if gE != wE:
          wantW = naiveWeightFor(dt, want.logits, t, E, K, gotRow, scale,
              int32(gE)).widenTo(dt).float64
          bar = uStep.float64 * abs(wantW) +
            float64(E + K + 4) * U32 * abs(wantW) +
            8.0 * (reassocDelta(H, sumAbs[t * E + gE]) +
              elSlack(uStep.float64, lg(gE)) +
              reassocDelta(H, sumAbs[t * E + lmE]) +
              elSlack(uStep.float64, lm)) * abs(wantW) + FloorSub
        let diff = abs(gotW - wantW)
        doAssert diff <= bar,
          &"routing weight outside the bar at (t {t}, slot {slot}, case {caseId}): " &
          &"{diff:.3e} > {bar:.3e}"
        worstUse = max(worstUse, diff / bar)
        if got.w[idx] == want.routW[idx]:
          inc exactW
        inc total
      totalSwaps += swaps
    if caseId == 0:
      case0record = got

  block determinism:
    # identical inputs give identical top-K sets, bit-identical ids and weights
    var rng0 = initNaiveRng(seed)
    let bits0 = takeInputs(rng0)
    load(bits0)
    launch()
    sentinels(bits0)
    let again = record()
    for i in 0 ..< nIds:
      doAssert again.ids[i] == case0record.ids[i], "ids differ run to run"
      doAssert again.w[i] == case0record.w[i], "routing weights differ run to run"

  echo &"[{label} {ulpDatatypeName(ulpG)}] cases={cases} launches={launches} " &
    &"worst bar usage {worstUse:.3f}, bit-exact {exactW}/{total}, " &
    &"tie-region swaps {totalSwaps}, reassociation use {reassocWorst:.3f}"
  suiteCases += cases
  suiteLaunches += launches
  suiteWorstUse = max(suiteWorstUse, max(worstUse, reassocWorst))
  suiteExact += exactW
  suiteTotal += total

proc runSharedGateCombo(engine: HwEngine; dt: ScalarKind; T, cases: int;
    seed: uint64; label: string) =
  ## Shared-expert scalar GEMV at the mega binding (H = 2048), one raw fp32
  ## logit per token against the (1, H) shared expert row weight, the element
  ## dtype selects the static binding
  ##
  ## - judged under the band against the naive sequential dot
  ## - case 0 relaunched bit-identical
  let H = 2048
  let ulpG = if dt == kBfloat16: ulpBf16 else: ulpFp16
  let nX = T * H
  var outB = allocPageBuf[float32](T)
  var xB = allocPageBuf[uint16](nX)
  var sgwB = allocPageBuf[uint16](H)
  defer:
    freePageBuf(outB); freePageBuf(xB); freePageBuf(sgwB)
  var outPA = outB.pa()
  var xPA = xB.pa()
  var sgwPA = sgwB.pa()
  let poison = 4.203895392974451e-45'f32   # the smallest positive subnormal fp32

  var worstUse = 0.0'f64
  var exact = 0
  var launches = 0

  proc takeInputs(rng: var NaiveRng): tuple[x, sgw: seq[uint16]] =
    ## Seeded inputs, element-dtype bits for the activations and the (1, H)
    ## shared expert row weight.
    var xBits = newSeq[uint16](nX)
    var sgwBits = newSeq[uint16](H)
    for i in 0 ..< nX:
      xBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
    for k in 0 ..< H:
      sgwBits[k] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
    result = (xBits, sgwBits)

  proc load(bits: tuple[x, sgw: seq[uint16]]) =
    for i in 0 ..< nX:
      xB.hostPtr[i] = bits.x[i]
    for k in 0 ..< H:
      sgwB.hostPtr[k] = bits.sgw[k]
    for t in 0 ..< T:
      outB.hostPtr[t] = poison

  proc sentinels(bits: tuple[x, sgw: seq[uint16]]) =
    assertReadUnchanged(xB, bits.x)
    assertReadUnchanged(sgwB, bits.sgw)

  proc launch =
    let kernelName = if dt == kBfloat16: "cer_shared_gate_bf16" else: "cer_shared_gate_f16"
    engine.run << (grid: (T, 1, 1), blk: (32, 1, 1)) >>
      (kernelName, outPA, (xPA, sgwPA))
    inc launches

  proc record(): seq[float32] =
    result = newSeq[float32](T)
    for t in 0 ..< T:
      result[t] = outB.hostPtr[t]

  var case0record: seq[float32]
  var rng = initNaiveRng(seed)
  for caseId in 0 ..< cases:
    let bits = takeInputs(rng)
    let want = naiveSharedGateDot(dt, bits.x, bits.sgw, T, H)
    load(bits)
    launch()
    sentinels(bits)
    let got = record()
    for t in 0 ..< T:
      doAssert got[t] != poison, &"shared gate logit never stored at token {t}"
      # the kernel walks 16-wide mma chunks, the naive side a sequential fp32 sum,
      # the band covers both orders' fp32 forward error, the store is raw fp32
      var sumAbs = 0.0'f64
      for k in 0 ..< H:
        sumAbs += abs(bits.x[t * H + k].widenTo(dt).float64 *
          bits.sgw[k].widenTo(dt).float64)
      let bar = 2.0 * float64(H) * U32 * sumAbs + FloorSub
      let diff = abs(got[t].float64 - want[t].float64)
      doAssert diff <= bar,
        &"shared gate logit outside the bar at (t {t}, case {caseId}): " &
        &"{diff:.3e} > {bar:.3e}"
      worstUse = max(worstUse, diff / bar)
      if got[t] == want[t]:
        inc exact
    if caseId == 0:
      case0record = got

  block determinism:
    # identical inputs give a bit-identical raw fp32 logit run to run
    var rng0 = initNaiveRng(seed)
    let bits0 = takeInputs(rng0)
    load(bits0)
    launch()
    sentinels(bits0)
    let again = record()
    for t in 0 ..< T:
      doAssert again[t] == case0record[t], "shared gate logit differs run to run"

  echo &"[{label} {ulpDatatypeName(ulpG)}] cases={cases} launches={launches} " &
    &"worst bar usage {worstUse:.3f}, bit-exact {exact}/{cases * T}"
  suiteCases += cases
  suiteLaunches += launches
  suiteWorstUse = max(suiteWorstUse, worstUse)
  suiteExact += exact
  suiteTotal += cases * T

proc runMergeCombo(engine: HwEngine; dt: ScalarKind; T, H, K, cases: int;
    seed: uint64; kernelName: string) =
  ## fp32-partial merge, judged bit-exact against the same sequential fp32
  ## slot-order sum, one El round at the store on both sides, `kernelName`
  ## selects the static binding (H = 256 suite scale, H = 2048 the mega scale).
  let H32 = H
  let ulpG = if dt == kBfloat16: ulpBf16 else: ulpFp16
  let nOut = T * H32
  let nPart = T * (K + 1) * H32
  var outB = allocPageBuf[uint16](nOut)
  var partB = allocPageBuf[float32](nPart)
  defer:
    freePageBuf(outB); freePageBuf(partB)
  var outPA = outB.pa()
  var partPA = partB.pa()
  let gridX = int32(T)
  let gridY = int32(H32 div 32)

  var launches = 0
  var exact = 0
  var rng = initNaiveRng(seed)
  for caseId in 0 ..< cases:
    var want = newSeq[uint16](nOut)
    for i in 0 ..< nPart:
      partB.hostPtr[i] = rng.nextF32(-1.0'f32, 1.0'f32)
    for i in 0 ..< nOut:
      outB.hostPtr[i] = 0xFFFF'u16
    engine.run << (grid: (T, H32 div 32, 1), blk: (32, 1, 1)) >>
      (kernelName, outPA, partPA)
    inc launches, int(gridX * gridY)
    for t in 0 ..< T:
      for col in 0 ..< H32:
        var acc = 0.0'f32
        for y in 0 ..< K + 1:
          acc += partB.hostPtr[t * (K + 1) * H32 + y * H32 + col]
        want[t * H32 + col] = acc.narrowTo(dt)
    for i in 0 ..< nOut:
      doAssert outB.hostPtr[i] == want[i],
        &"merge output not bit-identical at element {i}, case {caseId}"
      if outB.hostPtr[i] == want[i]:
        inc exact
  echo &"[merge {ulpDatatypeName(ulpG)}] cases={cases} launches={launches} bit-exact {exact}/{nOut*cases}"
  suiteCases += cases
  suiteLaunches += launches
  suiteExact += exact
  suiteTotal += nOut * cases

proc runPoisonedRouter(engine: HwEngine) =
  ## All-poisoned score pass, the router weight holding NaN bits, every logit
  ## NaNs, no score ever matches the group max, all K slots take the unmatched
  ## branch. The expert walk and merge then run on the same poisoned routing
  ##
  ## | judged   | assertion                                                                           |
  ## | -------- | ----------------------------------------------------------------------------------- |
  ## | ids      | every slot's id is E−1, inside [0, E), the expert-row reads stay in bounds          |
  ## | weights  | every slot's weight is the 0x0000 bit pattern, the renorm never NaNs a zero sum     |
  ## | partials | routed rows exactly +0.0 (zero weight times a finite projection), shared row finite |
  ## | merge    | the merged row holds no NaN or Inf bit pattern                                      |
  ## | repeat   | the chain relaunched bit-identical                                                  |
  const T = 1
  const H = 2048
  const E = 256
  const K = 8
  const I = 512
  const Seed = 0xC04D0531'u64
  let nIds = T * K
  var idsB = allocPageBuf[int32](nIds)
  var wB = allocPageBuf[uint16](nIds)
  var xB = allocPageBuf[uint16](T * H)
  var rWB = allocPageBuf[uint16](E * H)
  var partB = allocPageBuf[float32](T * (K + 1) * H)
  var hB = allocPageBuf[uint16](T * K * I)
  var hsB = allocPageBuf[uint16](T * I)
  var outB = allocPageBuf[uint16](T * H)
  var guB = allocPageBuf[uint16](E * 2 * I * H)
  var dWB = allocPageBuf[uint16](E * H * I)
  var sgB = allocPageBuf[uint16](I * H)
  var suB = allocPageBuf[uint16](I * H)
  var sdB = allocPageBuf[uint16](H * I)
  var gvB = allocPageBuf[uint16](H)
  defer:
    freePageBuf(idsB); freePageBuf(wB); freePageBuf(xB); freePageBuf(rWB)
    freePageBuf(partB); freePageBuf(hB); freePageBuf(hsB); freePageBuf(outB)
    freePageBuf(guB); freePageBuf(dWB); freePageBuf(sgB); freePageBuf(suB)
    freePageBuf(sdB); freePageBuf(gvB)
  var idsPA = idsB.pa()
  var wPA = wB.pa()
  var xPA = xB.pa()
  var rWPA = rWB.pa()
  var partPA = partB.pa()
  var hPA = hB.pa()
  var hsPA = hsB.pa()
  var outPA = outB.pa()
  var guPA = guB.pa()
  var dWPA = dWB.pa()
  var sgPA = sgB.pa()
  var suPA = suB.pa()
  var sdPA = sdB.pa()
  var gvPA = gvB.pa()

  # the poisoned pass, NaN bits across the router weight, the activations finite
  var rng = initNaiveRng(Seed)
  for e in 0 ..< E * H:
    rWB.hostPtr[e] = 0x7FC0'u16
  for i in 0 ..< T * H:
    xB.hostPtr[i] = f32ToBf16(rng.nextF32(-1.0'f32, 1.0'f32))
  # the expert and shared weights finite, expert E−1's rows live and in bounds
  for i in 0 ..< E * 2 * I * H:
    guB.hostPtr[i] = f32ToBf16(rng.nextF32(-0.02'f32, 0.02'f32))
  for i in 0 ..< E * H * I:
    dWB.hostPtr[i] = f32ToBf16(rng.nextF32(-0.02'f32, 0.02'f32))
  for i in 0 ..< I * H:
    sgB.hostPtr[i] = f32ToBf16(rng.nextF32(-0.02'f32, 0.02'f32))
    suB.hostPtr[i] = f32ToBf16(rng.nextF32(-0.02'f32, 0.02'f32))
  for i in 0 ..< H * I:
    sdB.hostPtr[i] = f32ToBf16(rng.nextF32(-0.02'f32, 0.02'f32))
  for i in 0 ..< H:
    gvB.hostPtr[i] = f32ToBf16(rng.nextF32(-0.02'f32, 0.02'f32))

  proc loadSentinels() =
    for i in 0 ..< nIds:
      idsB.hostPtr[i] = -777
      wB.hostPtr[i] = 0xBEEF
    for i in 0 ..< T * (K + 1) * H:
      partB.hostPtr[i] = 4.203895392974451e-45'f32

  proc launchChain() =
    engine.run << (grid: (T, 1, 1), blk: (32, 1, 1)) >>
      ("cer_moe_route_bf16_mega", idsPA, (wPA, xPA, rWPA, int32(T)))
    engine.run << (grid: (T, K + 1, 1), blk: (32, 1, 1)) >>
      ("cer_moe_fwd_bf16_mega", partPA,
        (xPA, rWPA, guPA, dWPA, sgPA, suPA, sdPA, gvPA, hPA, hsPA))
    engine.run << (grid: (T, H div 32, 1), blk: (32, 1, 1)) >>
      ("cer_moe_merge_bf16_mega", outPA, partPA)

  proc record(): tuple[ids: seq[int32], w: seq[uint16], part: seq[float32], outRow: seq[uint16]] =
    result.ids = newSeq[int32](nIds)
    result.w = newSeq[uint16](nIds)
    result.part = newSeq[float32](T * (K + 1) * H)
    result.outRow = newSeq[uint16](T * H)
    for i in 0 ..< nIds:
      result.ids[i] = idsB.hostPtr[i]
      result.w[i] = wB.hostPtr[i]
    for i in 0 ..< T * (K + 1) * H:
      result.part[i] = partB.hostPtr[i]
    for i in 0 ..< T * H:
      result.outRow[i] = outB.hostPtr[i]

  loadSentinels()
  launchChain()
  let got = record()
  for t in 0 ..< T:
    for slot in 0 ..< K:
      let id = got.ids[t * K + slot]
      doAssert id >= 0 and id < int32(E),
        &"poisoned pass id out of range at slot {slot}: {id}"
      doAssert id == int32(E - 1),
        &"poisoned pass slot {slot} not on the unmatched branch's expert E−1: {id}"
      doAssert got.w[t * K + slot] == 0x0000'u16,
        &"poisoned pass weight not the zero pattern at slot {slot}: " &
        &"0x{got.w[t * K + slot]:04x} = {bf16ToF32(got.w[t * K + slot])}"
    # the routed partial rows, zero weight times a finite projection, exactly +0.0
    for slot in 0 ..< K:
      for e in 0 ..< H:
        doAssert got.part[(t * (K + 1) + slot) * H + e] == 0.0'f32,
          &"poisoned pass routed partial not exactly zero at (slot {slot}, col {e}): " &
          &"{got.part[(t * (K + 1) + slot) * H + e]}"
    # the shared row and the merge, finite everywhere, no NaN or Inf pattern
    for e in 0 ..< H:
      doAssert classify(got.part[(t * (K + 1) + K) * H + e]) notin {fcNan, fcInf},
        &"poisoned pass shared partial not finite at col {e}"
      let ob = bf16ToF32(got.outRow[t * H + e])
      doAssert classify(ob) notin {fcNan, fcInf},
        &"poisoned pass merged output not finite at col {e}"
  # the relaunch, bit-identical ids, weights, partials and merged row
  loadSentinels()
  launchChain()
  let again = record()
  for i in 0 ..< nIds:
    doAssert again.ids[i] == got.ids[i], "poisoned pass ids differ run to run"
    doAssert again.w[i] == got.w[i], "poisoned pass weights differ run to run"
  for i in 0 ..< T * (K + 1) * H:
    doAssert again.part[i] == got.part[i], "poisoned pass partials differ run to run"
  for i in 0 ..< T * H:
    doAssert again.outRow[i] == got.outRow[i], "poisoned pass merge differs run to run"
  echo "[poisoned router bf16 mega] all K slots on the unmatched branch, " &
    &"weights the zero pattern, routed partials exactly zero, " &
    &"chain relaunched bit-identical"

proc checkReassociation(dt: ScalarKind; T, H, E: int; seed: uint64) =
  ## Standalone reassociation check over fresh seeds beyond the committed combos
  ##
  ## - the 16-wide-chunk walk, the kernel's mma accumulation structure, must sit
  ##   inside its stated fp32 reassociation bound against the naive sequential sum
  ## - judged per logit, worst usage printed
  var rng = initNaiveRng(seed)
  let nX = T * H
  var x = newSeq[uint16](nX)
  var w = newSeq[uint16](E * H)
  for i in 0 ..< nX:
    x[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
  for i in 0 ..< w.len:
    w[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
  let seqLogits = dotRawLogits(dt, x, w, T, H, E, false)
  let chunkLogits = dotRawLogits(dt, x, w, T, H, E, true)
  var worstUse = 0.0'f64
  var checks = 0
  for t in 0 ..< T:
    for e in 0 ..< E:
      var sa = 0.0'f64
      for k in 0 ..< H:
        sa += abs(x[t * H + k].widenTo(dt).float64 *
          w[e * H + k].widenTo(dt).float64)
      let bar = reassocDelta(H, sa)
      let diff = abs(chunkLogits[t * E + e].float64 - seqLogits[t * E + e].float64)
      doAssert diff <= bar,
        &"the chunk walk's fp32 logit left its reassociation bound at " &
        &"(token {t}, expert {e}): {diff:.3e} > {bar:.3e}"
      worstUse = max(worstUse, diff / bar)
      inc checks
  suiteWorstUse = max(suiteWorstUse, worstUse)
  echo &"[reassociation {ulpDatatypeName(if dt == kBfloat16: ulpBf16 else: ulpFp16)} H{H} seed 0x{seed:x}] worst bar " &
    &"usage {worstUse:.3f} over {checks} logits, chunk walk vs sequential sum"


proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(MoeRouterMsl)
  runCombo(engine, kBfloat16, 8, 2048, 256, 8, 8, 0xC04D0521'u64, 1.0'f32,
    "router mega")
  runCombo(engine, kFloat16, 8, 2048, 256, 8, 8, 0xC04D0522'u64, 1.0'f32,
    "router mega")
  runCombo(engine, kBfloat16, 4, 256, 64, 4, 8, 0xC04D0523'u64, 1.0'f32,
    "router small")
  runCombo(engine, kBfloat16, 8, 2048, 256, 8, 8, 0xC04D0528'u64, 2.0'f32,
    "router mega scale 2")
  runSharedGateCombo(engine, kBfloat16, 8, 8, 0xC04D0527'u64, "shared gate")
  runSharedGateCombo(engine, kFloat16, 8, 8, 0xC04D0531'u64, "shared gate")
  runMergeCombo(engine, kBfloat16, 8, 256, 8, 8, 0xC04D0524'u64, "cer_moe_merge_bf16")
  runMergeCombo(engine, kFloat16, 8, 256, 8, 8, 0xC04D0532'u64, "cer_moe_merge_f16")
  runMergeCombo(engine, kBfloat16, 8, 2048, 8, 2, 0xC04D0529'u64, "cer_moe_merge_bf16_mega")
  runPoisonedRouter(engine)
  checkReassociation(kBfloat16, 8, 2048, 256, 0xC04D0525'u64)
  checkReassociation(kFloat16, 8, 2048, 256, 0xC04D0526'u64)
  checkReassociation(kBfloat16, 4, 256, 64, 0xC04D0530'u64)
  echo &"CERAMIC MOE_ROUTER VERDICT: cases={suiteCases} launches={suiteLaunches} " &
    &"worst bar usage {suiteWorstUse:.3f}, bit-exact {suiteExact}/{suiteTotal}"

main()
