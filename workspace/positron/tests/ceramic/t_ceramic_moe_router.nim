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
##   ids[t] = top-K(p), lowest expert id on ties
##   w[t, slot] = El(p[ids[slot]] / sum(p[ids]) · Scale)
##   El = one round-to-nearest-even round in the family dtype
##
## | subject    | contract                                                                                           |
## | ---------- | -------------------------------------------------------------------------------------------------- |
## | naive side | host fp32 dot products over the exact widenings, then the same rounding chain in fp32              |
## | softmax    | both sides run the exp2 exponential form, so the top-K set is a function of the El-rounded logits  |
## | top-K      | lowest expert id on equal scores, ids distinct and in [0, E), routing weights descending per token |
## | merge      | the fp32 partial-row merge sums in slot order, one El round, judged bit-exact against the same sum |
##
## | shape | T | H    | E   | K | Scale | family     | cases |
## | ----- | --- | ---- | --- | --- | ----- | ---------- | ----- |
## | mega  | 8 | 2048 | 256 | 8 | 1.0   | bf16, fp16 | 8     |
## | small | 4 | 256  | 64  | 4 | 1.0   | bf16       | 8     |
##
## Band model, stated before measurement, u32 = 2⁻²⁴ fp32, u_fam = 2⁻⁸ bf16 / 2⁻¹¹ fp16
## - the top-K set follows from the El-rounded logits alone, softmax is strictly monotone in the logits and the exp2
##   form is applied identically on both sides, so the ids are asserted equal to the naive top-K, not banded
## - the weights carry every divergence, per the bar table
##
## | term                   | bar                                          | covers                                                                                    |
## | ---------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------- |
## | w (t, s)               | u_fam·abs(w) + (E + K + 4)·u32·abs(w) + 2⁻²⁵ | the score-chain rounding sites, one RNE per side                                          |
## | GEMV accumulator       | (E + K + 4)·u32·abs(w) shares this term      | an E-wide dot on both sides, the kernel walks 16-wide mma chunks                          |
## | softmax denominator    | (E + K + 4)·u32·abs(w) shares this term      | the summation order, E terms                                                              |
## | exp2, division, renorm | (E + K + 4)·u32·abs(w) shares this term      | the 1-ulp-class exp2 form, the division, the K-term renorm sum                            |
## | store round            | u_fam·abs(w)                                 | the two sides round slightly different fp32 weights, each RNE within u_fam of its operand |
## | subnormal floor        | 2⁻²⁵                                         | the fp16 subnormal grid, also the bf16 grid                                               |
##
## - the merge output is exact-where-legal, both sides sum the same fp32 partials
##   in the same slot order, one El round, asserted bit-identical
## - the measured divergence justifies the model, never sets the bar
## - adjudicated on Apple M4 Max with fresh seeded xorshift64 inputs

import std/[strformat, math]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/moe_router
import ../naive/naive_rng
import ../naive/naive_tensors
import ceramic_pagebuf

# ─── Device entries, one per (family dtype, shape) binding ────────────

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

  proc cer_moe_merge_bf16(
      out_r: ptr UncheckedArray[bfloat16],
      partial: ptr UncheckedArray[float32]) {.global.} =
    moe_decode_merge[bfloat16, 256, 8](out_r, partial)

# ─── Host, family dtype helpers, the independent reference ────────────

# libm exp2f, the crucible export table also carries an exp2 device builtin
# whose host body is a discard, so std/math's exp2 cannot be called here
proc exp2fHost(x: cfloat): cfloat {.importc: "exp2f", header: "<math.h>".}

const
  U32 = 5.9604644775390625e-8        # 2^-24, the fp32 unit roundoff
  FloorSub = 2.9802322387695312e-8   # 2^-25, half the fp16 subnormal ulp,
                                     # the rounding floor at tiny outputs
  Log2E = 1.4426950408889634'f32

type Family = enum
  famBf16, famF16

proc toFamBits(fam: Family, x: float32): uint16 =
  ## Returns the family-dtype round-to-nearest-even bit pattern of an fp32 value.
  if fam == famBf16: f32ToBf16(x) else: fp32ToFp16(x)

proc famWiden(fam: Family, h: uint16): float32 =
  ## Returns the exact fp32 widening of a family dtype bit pattern.
  if fam == famBf16: bf16ToF32(h) else: fp16ToFp32(h)

proc famName(fam: Family): string =
  if fam == famBf16: "bf16" else: "fp16"

proc naiveRouter(fam: Family, x, w: seq[uint16]; T, H, E, K: int, Scale: float32):
    tuple[ids: seq[int32], routW: seq[uint16]] =
  ## Independent host reference for the softmax form's rounding chain in fp32
  ##
  ## - sequential fp32 dot over the exact widenings, one El round per logit
  ## - fp32 softmax over the exp2 form
  ## - top-K with the lowest-index tiebreak, then renorm, scale, one El round per routing weight
  result.ids = newSeq[int32](T * K)
  result.routW = newSeq[uint16](T * K)
  for t in 0 ..< T:
    var logits = newSeq[float32](E)
    for e in 0 ..< E:
      var acc = 0.0'f32
      for k in 0 ..< H:
        acc += famWiden(fam, x[t * H + k]) * famWiden(fam, w[e * H + k])
      logits[e] = if fam == famBf16: bf16ToF32(f32ToBf16(acc))
                  else: fp16ToFp32(fp32ToFp16(acc))
    var lm = -3.402823466e38'f32
    for e in 0 ..< E:
      lm = max(lm, logits[e])
    var p = newSeq[float32](E)
    var ls = 0.0'f32
    # the crucible export table also carries an exp2 device builtin whose
    # host body is a discard, so exp2fHost is the only host exp2 here
    for e in 0 ..< E:
      p[e] = exp2fHost((logits[e] - lm) * Log2E)
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
        if fam == famBf16: f32ToBf16(weight) else: fp32ToFp16(weight)

proc runCombo(engine: HwEngine; fam: Family, T, H, E, K, cases: int;
    seed: uint64; label: string) =
  ## One (family dtype, shape) combination over `cases` independent seeded runs,
  ## ids judged exact against the naive top-K, weights judged per element under
  ## the band, case 0 relaunched bit-identical.
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
  let kernelName = if fam == famBf16:
    (if H == 2048: "cer_moe_route_bf16_mega" else: "cer_moe_route_bf16_small")
  else:
    "cer_moe_route_f16_mega"
  let uFam = if fam == famBf16: 3.90625e-3 else: 4.8828125e-4

  var worstUse = 0.0'f64
  var exactW = 0
  var total = 0
  var launches = 0

  proc takeInputs(rng: var NaiveRng): tuple[x, w: seq[uint16]] =
    ## Seeded inputs, family-dtype bits for x and the router weight.
    var xBits = newSeq[uint16](nX)
    var wBits = newSeq[uint16](nW)
    for i in 0 ..< nX:
      xBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
    for i in 0 ..< nW:
      wBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
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

  proc snap(): tuple[ids: seq[int32], w: seq[uint16]] =
    result.ids = newSeq[int32](nIds)
    result.w = newSeq[uint16](nIds)
    for i in 0 ..< nIds:
      result.ids[i] = idsB.hostPtr[i]
      result.w[i] = wB.hostPtr[i]

  var case0Snap: tuple[ids: seq[int32], w: seq[uint16]]
  var rng = initNaiveRng(seed)
  for caseId in 0 ..< cases:
    let bits = takeInputs(rng)
    let want = naiveRouter(fam, bits.x, bits.w, T, H, E, K, 1.0'f32)
    load(bits)
    launch()
    sentinels(bits)
    let got = snap()
    for t in 0 ..< T:
      # the ids are exact, the top-K set follows from the El-rounded logits alone
      # both sides apply the same monotone exp2 form
      for slot in 0 ..< K:
        let idx = t * K + slot
        doAssert got.ids[idx] == want.ids[idx],
          &"top-K id mismatch at (t {t}, slot {slot}, case {caseId}): " &
          &"got {got.ids[idx]}, want {want.ids[idx]}"
      # the routing weights carry the band
      var prev = 3.402823466e38'f64
      for slot in 0 ..< K:
        let idx = t * K + slot
        doAssert got.ids[idx] != (if slot > 0: got.ids[idx - 1] else: -1),
          "duplicate expert id in one token's top-K"
        let gotW = famWiden(fam, got.w[idx]).float64
        doAssert gotW <= prev, "routing weights not descending"
        prev = gotW
        let wantW = famWiden(fam, want.routW[idx]).float64
        let bar = uFam.float64 * abs(wantW) +
          float64(E + K + 4) * U32 * abs(wantW) + FloorSub
        let diff = abs(gotW - wantW)
        doAssert diff <= bar,
          &"routing weight outside the bar at (t {t}, slot {slot}, case {caseId}): " &
          &"{diff:.3e} > {bar:.3e}"
        worstUse = max(worstUse, diff / bar)
        if got.w[idx] == want.routW[idx]:
          inc exactW
        inc total
    if caseId == 0:
      case0Snap = got

  block determinism:
    # identical inputs give identical top-K sets, bit-identical ids and weights
    var rng0 = initNaiveRng(seed)
    let bits0 = takeInputs(rng0)
    load(bits0)
    launch()
    sentinels(bits0)
    let again = snap()
    for i in 0 ..< nIds:
      doAssert again.ids[i] == case0Snap.ids[i], "ids differ run to run"
      doAssert again.w[i] == case0Snap.w[i], "routing weights differ run to run"

  echo &"[{label} {famName(fam)}] cases={cases} launches={launches} " &
    &"worst bar usage {worstUse:.3f}, bit-exact {exactW}/{total}"

proc runMergeCombo(engine: HwEngine; T, H, K, cases: int; seed: uint64) =
  ## fp32-partial merge, judged bit-exact against the same sequential fp32
  ## slot-order sum, one El round at the store on both sides.
  let H32 = H
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
      ("cer_moe_merge_bf16", outPA, partPA)
    inc launches, int(gridX * gridY)
    for t in 0 ..< T:
      for col in 0 ..< H32:
        var acc = 0.0'f32
        for y in 0 ..< K + 1:
          acc += partB.hostPtr[t * (K + 1) * H32 + y * H32 + col]
        want[t * H32 + col] = f32ToBf16(acc)
    for i in 0 ..< nOut:
      doAssert outB.hostPtr[i] == want[i],
        &"merge output not bit-identical at element {i}, case {caseId}"
      if outB.hostPtr[i] == want[i]:
        inc exact
  echo &"[merge bf16] cases={cases} launches={launches} bit-exact {exact}/{nOut*cases}"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(MoeRouterMsl)
  runCombo(engine, famBf16, 8, 2048, 256, 8, 8, 0xC04D0521'u64, "router mega")
  runCombo(engine, famF16, 8, 2048, 256, 8, 8, 0xC04D0522'u64, "router mega")
  runCombo(engine, famBf16, 4, 256, 64, 4, 8, 0xC04D0523'u64, "router small")
  runMergeCombo(engine, 8, 256, 8, 8, 0xC04D0524'u64)
  echo "CERAMIC MOE_ROUTER VERDICT: ids exact, weights inside the stated bands"

main()
