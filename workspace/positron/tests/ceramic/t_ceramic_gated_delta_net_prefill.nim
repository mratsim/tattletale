# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

## Run commands, from the repo root (the aggregate runner is nim test_positron_properties):
## - nim c -r -d:release --warnings:off --outdir:build/tests workspace/positron/tests/ceramic/t_ceramic_gated_delta_net_prefill.nim
##
## Ceramic GDN prefill suite, chunked scan against the reference tier:
##
## | judgment   | reference side                                                                                                        |
## | ---------- | --------------------------------------------------------------------------------------------------------------------- |
## | y, state   | the fp64 chunked WY/UT reference `gdnPrefillChunked` (tests/properties/refs.nim)                                      |
## | continuity | the fp64 per-token walk `gdnPrefillPerToken`, the ceramic chunked scan's final state against the stepwise final state |
##
## - the reference-side continuity pair (chunked vs per-token, y and final state) is judged inside the suite itself
## - the reference walks are per-sequence, B sequences take B reference calls on per-sequence input slices
## - per-sequence results are assembled on the stacked axes
##
## Inputs, all seeded xorshift64, fixture-free:
##
## | input     | rule                                                                                                                    |
## | --------- | ----------------------------------------------------------------------------------------------------------------------- |
## | q, k      | l2-normalized per (head, token) row in fp32, then rounded to the element dtype, the kernel contract's post-l2norm shape |
## | v, state0 | [-1, 1), the initial state never zero                                                                                   |
## | beta      | [0.2, 0.8), edge combos carry the exact-zero beta and the near-zero decay (g -> 0-) inside the same band                |
## | g         | [-0.5, -0.01) log-decay                                                                                                 |
##
## - the q, k normalization also keeps the delta-rule recursion bounded over 256 tokens in fp16 y range
##
## - the element-dtype bits are the shared input, the reference sides widen them exactly, no input rounding divergence
##
## Band model, stated before measurement and judged per element, u₃₂ = 2⁻²⁴ the fp32
## unit roundoff. The chunked scan reassociates the per-token recurrence:
##
## | reassociation | structure                                                                                        |
## | ------------- | ------------------------------------------------------------------------------------------------ |
## | decay         | carried as the cumulative log decay cumulogdecay per chunk, pair decay from exponent differences |
## | updates       | the token updates u_t solved in token order through the A-matrix recurrence                      |
## | carry         | one decayed carry read plus the u outer products, assembled once per chunk                       |
##
##   per chunk:   cumulogdecay ─→ per token t: G_t read ─→ u_t solve ─→ y_t store
##                (t in token order)                └─────────┐
##            └──→ S ← decayed carry read + Σ_s pd(end, s)·k_s [x] u_s
##
## - the per-token walk instead applies exp(g) per token, reading and updating the state each step
## - both reference spellings run fp64, the ceramic core runs fp32 state math with element-dtype handoffs
## - the judged divergence is the ceramic side's rounding against the fp64 chunked reference structure
##
## | site             | bound                                                                    |
## | ---------------- | ------------------------------------------------------------------------ |
## | cumulogdecay     | per entry ≤ i·u₃₂·max abs(cumulogdecay), uniform per chunk cLen·u₃₂·cmax |
## | decay factors    | relative ≤ cLen·u₃₂·cmax + 4·u₃₂ (log2e multiply, exp2 form)             |
## | k·k / q̃·k dots  | relative ≤ Dk·u₃₂ (elementwise round plus row-sum tree)                  |
## | S·k / S·q̃ reads | Dk·u₃₂·Σ abs(S·k) plus the carried state error Σ abs(k)·ΔS               |
## | q̃ scale         | relative ≤ 2·2⁻²¹ (rsqrt-multiply vs the reference divide)               |
## | u solve          | β·(base error + Σ abs(A)·Δu + ΔA·abs(u) + t·u₃₂·Σ abs(A·u))              |
## | chunk carry      | decayed old state + Σ abs(pd·k)·Δu + cLen·u₃₂·Σ abs(pd·k·u)              |
## | y store          | one element-dtype RNE, u_step·abs(y) plus the subnormal grid floor       |
##
## - the reassociation budget per chunk is C token terms per sum (solve, y, carry) and Dk terms per dot over T/C chunks
##
## - every site's bound is a triangle-inequality sum of nonnegative per-op roundings, the model is additive
##
## - the per-element bars below are host-computed from an fp64 trace walk mirroring the chunked formulas, magnitudes only
## - the measured divergence corroborates the model and never sets a bar
##
## | bar            | bound                                                                                         |
## | -------------- | --------------------------------------------------------------------------------------------- |
## | y (bh, t, r)   | Δy(t, r) + u_step·abs(y_ref) + 2⁻²⁅                                                           |
## | state (bh,r,c) | the carried ΔS recursion across chunks, chunk-local terms in the table below                  |
## | continuity     | barS + 2·(T·Dk)·2⁻⁵³·max(abs(S_chunked), abs(S_walk)), the fp64 chunked-vs-walk reassociation |
##
## ΔS recursion per chunk, all terms nonnegative, computed per element:
##
## | component       | bound                                                         |
## | --------------- | ------------------------------------------------------------- |
## | decayed state   | decayEnd·(ΔS + abs(S)·(u₃₂ + relDecay)) + u₃₂·abs(decayEnd·S) |
## | u outer product | Σ_s abs(pd(end, s)·k_s)·(Δu_s + abs(u_s)·(relDecay + 3·u₃₂))  |
## | sum roundings   | cLen·u₃₂·Σ_s abs(pd·k_s·u_s) + u₃₂·abs(S_new)                 |
##
## Shapes (Dv = 16, TileR = 8, Dk = 32, grid (Dv div TileR, B·Hv), 32 lanes, one launch walks all chunks):
##
## | shape    | Hk | Hv | batch | hkRatio | dtype | T   | chunk |
## | -------- | --- | --- | ----- | ------- | ----- | --- | ----- |
## | baseline | 1  | 1  | 1     | 1       | fp16  | 8   | 32    |
## | baseline | 1  | 1  | 1     | 1       | fp16  | 64  | 32    |
## | gqa      | 2  | 4  | 2     | 2       | fp16  | 256 | 32    |
## | tail     | 1  | 1  | 1     | 1       | fp16  | 100 | 32    |
## | chunk64  | 1  | 1  | 1     | 1       | fp16  | 64  | 64    |
## | baseline | 1  | 1  | 1     | 1       | bf16  | 8   | 32    |
## | tail     | 1  | 1  | 1     | 1       | bf16  | 100 | 32    |
## | gqa64    | 2  | 4  | 2     | 2       | bf16  | 256 | 64    |
##
## - the T = 100 case carries a non-divisible tail chunk, 3 full chunks plus 4 tokens
## - every case starts from a non-zero random initial state
## - the GQA shape keeps both head-mapping terms live, the in-sequence ratio term
##   and the sequence-offset term, sequence 1 holding independent key heads
##
## - fp16 is the element dtype under test, bf16 the range-robust fallback
## - untouched-memory checks per launch, kernel-written buffers stay in their extents, kernel-read buffers stay bit-identical
## - run-to-run determinism, case 0 relaunched bit-identical per combination on Apple M4 Max

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/attn_ssm/gated_delta_net_prefill
import ceramic_pagebuf
import ceramic_dtype
import ../properties/refs

# ─── Device entries, one per (element dtype, chunk length) binding ─────

const GdnPrefillMsl = metal:
  proc cer_gdn_prefill_fp16_c32(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[float16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio, T: int32) {.global.} =
    gdnPrefillChunkScan(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, T, 32, 16, 8, 32)

  proc cer_gdn_prefill_fp16_c64(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[float16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio, T: int32) {.global.} =
    gdnPrefillChunkScan(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, T, 32, 16, 8, 64)

  proc cer_gdn_prefill_bf16_c32(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[bfloat16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio, T: int32) {.global.} =
    gdnPrefillChunkScan(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, T, 32, 16, 8, 32)

  proc cer_gdn_prefill_bf16_c64(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[bfloat16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio, T: int32) {.global.} =
    gdnPrefillChunkScan(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, T, 32, 16, 8, 64)

# ─── Host tolerance-model constants ───────────────────────────────────

const
  DecExp = 4.0 * U32                 # exp2(g·log2e) form relative bound, mul + exp2
  QScaleRel = 9.5367431640625e-7     # 2·2⁻²¹, rsqrt vs divide, relative
  UBf16 = 3.90625e-3                 # 2⁻⁸, the bf16 unit roundoff
  UF16 = 4.8828125e-4                # 2⁻¹¹, the fp16 unit roundoff
  FloorSub = 2.9802322387695312e-8   # 2⁻²⁵, half the constant fp16 subnormal ulp,
                                     # the rounding floor once |y| falls subnormal,
                                     # also covering the bf16 subnormal grid
  U64 = 1.1102230246251565e-16       # 2⁻⁵³, the fp64 unit roundoff

# ─── The bar helper, a magnitude trace over the chunked reference ────

type TraceBars = object
  ## Per-element bar arrays from the additive rounding model, magnitudes taken
  ## from the fp64 trace walk that mirrors the chunked reference formulas,
  ## magnitudes only, never the judged values:
  ##
  ## | field | shape           |
  ## | ----- | --------------- |
  ## | barY  | (bhMax, T, Dv)  |
  ## | barS  | (bhMax, Dv, Dk) |
  barY: seq[float64]
  barS: seq[float64]

proc gdnChunkTraceBars(
    s0w: Cube[float64], qw, kw: Cube[float64], vw: Cube[float64],
    bw, gw: Mat[float64],
    Hv, Hk, hkRatio, T, chunkLen, Dv, Dk: int,
    uStep: float64): TraceBars =
  let bhMax = s0w.planes
  result.barY = newSeq[float64](bhMax * T * Dv)
  result.barS = newSeq[float64](bhMax * Dv * Dk)
  let qScale = sqrt(float64(Dk))
  var kdotA = newSeq[float64](chunkLen * chunkLen)
  var kkAbsA = newSeq[float64](chunkLen * chunkLen)
  var pdRefA = newSeq[float64](chunkLen * chunkLen)
  var qkdotA = newSeq[float64](chunkLen * chunkLen)
  var qkAbsA = newSeq[float64](chunkLen * chunkLen)
  var bAbsA = newSeq[float64](chunkLen * chunkLen)
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    var carry = newSeq[float64](Dv * Dk)
    var dS = newSeq[float64](Dv * Dk)
    for r in 0 ..< Dv:
      for dk in 0 ..< Dk:
        carry[r * Dk + dk] = s0w.data[(bh * Dv + r) * Dk + dk]
    var cumulogdecay = newSeq[float64](chunkLen)
    var uRef = newSeq[float64](chunkLen * Dv)
    var du = newSeq[float64](chunkLen * Dv)
    var c0 = 0
    while c0 < T:
      let cLen = min(chunkLen, T - c0)
      # cumulative log decay in fp64 reference values, the fp32 drift bound uniform per chunk
      cumulogdecay[0] = gw.data[bh * T + c0]
      var cmax = abs(cumulogdecay[0])
      for i in 1 ..< cLen:
        cumulogdecay[i] = cumulogdecay[i - 1] + gw.data[bh * T + c0 + i]
        cmax = max(cmax, abs(cumulogdecay[i]))
      let decRel = float64(cLen) * U32 * cmax + DecExp
      for t in 0 ..< cLen:
        let gt = c0 + t
        let pdT = exp(cumulogdecay[t])
        let betaT = bw.data[bh * T + gt]
        # pair (t, s) dot magnitudes, shared across the state rows
        for s in 0 .. t:
          var kdot = 0.0'f64
          var kkAbs = 0.0'f64
          var qkdot = 0.0'f64
          var qkAbs = 0.0'f64
          for dk in 0 ..< Dk:
            let kt = kw.data[(hk * T + gt) * Dk + dk]
            let ks = kw.data[(hk * T + c0 + s) * Dk + dk]
            let qt = qw.data[(hk * T + gt) * Dk + dk] / qScale
            kdot += kt * ks
            kkAbs += abs(kt * ks)
            qkdot += qt * ks
            qkAbs += abs(qt * ks)
          let pdRef = exp(cumulogdecay[t] - cumulogdecay[s])
          kdotA[t * chunkLen + s] = kdot
          kkAbsA[t * chunkLen + s] = kkAbs
          pdRefA[t * chunkLen + s] = pdRef
          qkdotA[t * chunkLen + s] = qkdot
          qkAbsA[t * chunkLen + s] = qkAbs
          bAbsA[t * chunkLen + s] = pdRef * abs(qkdot)
        # solve bound recursion, per state row, the fp64 u values mirroring the reference
        for r in 0 ..< Dv:
          var kvAbs = 0.0'f64
          var dSG = 0.0'f64
          var kvN = 0.0'f64
          for dk in 0 ..< Dk:
            let term = carry[r * Dk + dk] * kw.data[(hk * T + gt) * Dk + dk]
            kvN += term
            kvAbs += abs(term)
            dSG += abs(kw.data[(hk * T + gt) * Dk + dk]) * dS[r * Dk + dk]
          let gRef = pdT * kvN
          let dG = pdT * (float64(Dk) * U32 * kvAbs + dSG) + abs(gRef) * (U32 + decRel)
          let vRef = vw.data[(bh * T + gt) * Dv + r]
          let base1 = betaT * (vRef - gRef)
          let dBase = betaT * (dG + U32 * (abs(vRef) + abs(gRef)))
          var duAcc = 0.0'f64
          var accAbs = 0.0'f64
          var acc = 0.0'f64
          for s in 0 ..< t:
            let aAbs = pdRefA[t * chunkLen + s] * abs(kdotA[t * chunkLen + s])
            let dA = aAbs * decRel + pdRefA[t * chunkLen + s] *
              float64(Dk) * U32 * kkAbsA[t * chunkLen + s]
            duAcc += aAbs * du[s * Dv + r] + dA * abs(uRef[s * Dv + r])
            let aTerm = aAbs * abs(uRef[s * Dv + r])
            accAbs += aTerm
            acc += pdRefA[t * chunkLen + s] * kdotA[t * chunkLen + s] * uRef[s * Dv + r]
          uRef[t * Dv + r] = base1 - betaT * acc
          du[t * Dv + r] = dBase + betaT * duAcc + betaT * float64(t) * U32 * accAbs +
            betaT * U32 * (abs(base1) + abs(acc))
        # output bars, the decayed carry read plus the ratio-form sum
        for r in 0 ..< Dv:
          var qkAbsSG = 0.0'f64
          var dSGq = 0.0'f64
          var cqN = 0.0'f64
          for dk in 0 ..< Dk:
            let qs = qw.data[(hk * T + gt) * Dk + dk] / qScale
            let term = carry[r * Dk + dk] * qs
            cqN += term
            qkAbsSG += abs(term)
            dSGq += abs(qs) * dS[r * Dk + dk]
          let crRef = pdT * cqN
          let dCR = pdT * (float64(Dk + 1) * U32 * qkAbsSG + dSGq) +
            abs(crRef) * (U32 + decRel)
          var yAbsSum = abs(crRef)
          var yErr = dCR
          var yRef = crRef
          for s in 0 .. t:
            let bAbs = bAbsA[t * chunkLen + s]
            let dB = bAbs * decRel + pdRefA[t * chunkLen + s] *
              (float64(Dk) * U32 * qkAbsA[t * chunkLen + s] +
                QScaleRel * abs(qkdotA[t * chunkLen + s]))
            let bTerm = bAbs * abs(uRef[s * Dv + r])
            yAbsSum += bTerm
            yErr += bAbs * du[s * Dv + r] + dB * abs(uRef[s * Dv + r])
            yRef += pdRefA[t * chunkLen + s] * qkdotA[t * chunkLen + s] * uRef[s * Dv + r]
          # the adds round once per term, the element-dtype store rounds once per element
          let dY = yErr + float64(t + 1) * U32 * yAbsSum + U32 * abs(yRef)
          result.barY[(bh * T + gt) * Dv + r] =
            dY + uStep * abs(yRef) + FloorSub
      # carry out of the chunk, the reference formula and its bound
      let decayEnd = exp(cumulogdecay[cLen - 1])
      for r in 0 ..< Dv:
        for dk in 0 ..< Dk:
          let idx = r * Dk + dk
          let sOld = carry[idx]
          let decTerm = decayEnd * sOld
          var sumTerm = 0.0'f64
          var sumAbs = 0.0'f64
          var sumErr = 0.0'f64
          for s in 0 ..< cLen:
            let pdEnd = exp(cumulogdecay[cLen - 1] - cumulogdecay[s])
            let ks = kw.data[(hk * T + c0 + s) * Dk + dk]
            let wAbs = abs(pdEnd * ks)
            let pTerm = pdEnd * ks * uRef[s * Dv + r]
            sumTerm += pTerm
            sumAbs += abs(pTerm)
            sumErr += wAbs * (du[s * Dv + r] +
              abs(uRef[s * Dv + r]) * (decRel + 3.0 * U32))
          let sNew = decTerm + sumTerm
          dS[idx] = decayEnd * (dS[idx] + abs(sOld) * (U32 + decRel)) +
            U32 * abs(decTerm) + sumErr +
            float64(cLen) * U32 * sumAbs + U32 * abs(sNew)
          carry[idx] = sNew
      c0 += cLen
    for r in 0 ..< Dv:
      for dk in 0 ..< Dk:
        result.barS[(bh * Dv + r) * Dk + dk] = dS[r * Dk + dk]

# ─── One case, seeded inputs → the reference walks → launch → judgment ──

proc sliceCube[T](c: Cube[T], plane0, planes: int): Cube[T] =
  ## Per-sequence slice of a stacked cube, the reference walks are per-sequence.
  result = Cube[T](planes: planes, rows: c.rows, cols: c.cols)
  result.data = newSeq[T](planes * c.rows * c.cols)
  for i in 0 ..< planes * c.rows * c.cols:
    result.data[i] = c.data[plane0 * c.rows * c.cols + i]

proc sliceMat[T](m: Mat[T], row0, rows: int): Mat[T] =
  ## Per-sequence slice of a stacked head-axis matrix.
  result = Mat[T](rows: rows, cols: m.cols)
  result.data = newSeq[T](rows * m.cols)
  for i in 0 ..< rows * m.cols:
    result.data[i] = m.data[row0 * m.cols + i]


type PrefillInputs = object
  ## One case's seeded inputs, element-dtype bits shared by the kernel and the reference
  ## sides through their exact widenings:
  ##
  ## | field    | shape              |
  ## | -------- | ------------------ |
  ## | qBits, k | (B·Hk, T, Dk)      |
  ## | vBits    | (B·Hv, T, Dv)      |
  ## | betaBits | (B·Hv, T)          |
  ## | gVals    | (B·Hv, T) f32      |
  ## | state0   | (B·Hv, Dv, Dk) f32 |
  qBits: seq[uint16]
  kBits: seq[uint16]
  vBits: seq[uint16]
  betaBits: seq[uint16]
  gVals: seq[float32]
  state0: seq[float32]

proc l2NormalizeRows(dst: var seq[uint16], dt: ScalarKind, rows, cols: int, rng: var PropRng) =
  ## Fills `dst` with l2-normalized element-dtype rows, the kernel contract's
  ## post-l2norm query/key shape:
  ##
  ## - each fp32 row is normalized to unit l2 norm, then rounded to the element dtype
  for r in 0 ..< rows:
    var norm2 = 0.0'f64
    for c in 0 ..< cols:
      let x = rng.nextF32(-1.0'f32, 1.0'f32).float64
      norm2 += x * x
      dst[r * cols + c] = x.float32.narrowTo(dt)
    let inv = 1.0 / sqrt(norm2)
    for c in 0 ..< cols:
      dst[r * cols + c] = (dst[r * cols + c].widenTo(dt).float64 * inv).float32.narrowTo(dt)

proc takeInputs(dt: ScalarKind, rng: var PropRng, bhMax, qkRows, T, Dv, Dk: int, gLoOverride = 0.0'f32, gHiOverride = 0.0'f32, betaZero = false): PrefillInputs =
  var qBits = newSeq[uint16](qkRows * T * Dk)
  var kBits = newSeq[uint16](qkRows * T * Dk)
  l2NormalizeRows(qBits, dt, qkRows * T, Dk, rng)
  l2NormalizeRows(kBits, dt, qkRows * T, Dk, rng)
  var vBits = newSeq[uint16](bhMax * T * Dv)
  for i in 0 ..< bhMax * T * Dv:
    vBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
  var betaBits = newSeq[uint16](bhMax * T)
  for i in 0 ..< bhMax * T:
    betaBits[i] = (if betaZero: 0.0'f32.narrowTo(dt)
                   else: rng.nextF32(0.2'f32, 0.8'f32).narrowTo(dt))
  var gVals = newSeq[float32](bhMax * T)
  for i in 0 ..< bhMax * T:
    # the span overrides exist for the edge combos, gLo 0.0 is the sentinel
    # meaning the suite's committed span
    gVals[i] = rng.nextF32((if gLoOverride != 0.0'f32: gLoOverride else: -0.5'f32),
      (if gLoOverride != 0.0'f32: gHiOverride else: -0.01'f32))
  var state0 = newSeq[float32](bhMax * Dv * Dk)
  for i in 0 ..< bhMax * Dv * Dk:
    state0[i] = rng.nextF32(-1.0'f32, 1.0'f32)
  result = PrefillInputs(qBits: qBits, kBits: kBits, vBits: vBits,
    betaBits: betaBits, gVals: gVals, state0: state0)

var suiteCases, suiteLaunches, suiteYExact, suiteYTotal = 0
var suiteWorstUse, suiteWorstState, suiteWorstCont, suiteWorstYUlp = 0.0'f64

proc runCase(engine: HwEngine, dt: ScalarKind, Hv, Hk, hkRatio, B, T, chunkLen: int, seed: uint64, label: string, gLoOverride = 0.0'f32, gHiOverride = 0.0'f32, betaZero = false) =
  ## One (element dtype, shape) combination, judged per element against the fp64 chunked
  ## reference and the fp64 per-token walk under the band model, relaunched bit-identical.
  const Dv = 16
  const Dk = 32
  const TileR = 8
  let bhMax = B * Hv
  let qkRows = B * Hk
  let stateElems = bhMax * Dv * Dk
  let yElems = bhMax * T * Dv
  let kernelName =
    if dt == kFloat16:
      if chunkLen == 32: "cer_gdn_prefill_fp16_c32" else: "cer_gdn_prefill_fp16_c64"
    else:
      if chunkLen == 32: "cer_gdn_prefill_bf16_c32" else: "cer_gdn_prefill_bf16_c64"
  let ulpG = if dt == kBfloat16: ulpBf16 else: ulpFp16
  let uStep = binadeStep(ulpG, -1)

  var stateB = allocPageBuf[float32](stateElems)
  var yB = allocPageBuf[uint16](yElems)
  var kB = allocPageBuf[uint16](qkRows * T * Dk)
  var qB = allocPageBuf[uint16](qkRows * T * Dk)
  var vB = allocPageBuf[uint16](bhMax * T * Dv)
  var gB = allocPageBuf[float32](bhMax * T)
  var betaB = allocPageBuf[uint16](bhMax * T)
  defer:
    freePageBuf(stateB); freePageBuf(yB); freePageBuf(kB); freePageBuf(qB)
    freePageBuf(vB); freePageBuf(gB); freePageBuf(betaB)
  var statePA = stateB.pa()
  var yPA = yB.pa()
  var kPA = kB.pa()
  var qPA = qB.pa()
  var vPA = vB.pa()
  var gPA = gB.pa()
  var betaPA = betaB.pa()

  # Launch-site contracts, see the kernel modules' binding and state ABI docs
  var worstState = 0.0'f64
  var worstStateUse = 0.0'f64
  var worstCont = 0.0'f64
  var worstYUse = 0.0'f64
  var worstYUlp = 0.0'f64
  var yExact = 0
  var yTotal = 0
  var launches = 0

  proc fillInputs(si: PrefillInputs) =
    for i in 0 ..< qkRows * T * Dk:
      kB.hostPtr[i] = si.kBits[i]
      qB.hostPtr[i] = si.qBits[i]
    for i in 0 ..< bhMax * T * Dv:
      vB.hostPtr[i] = si.vBits[i]
    for i in 0 ..< bhMax * T:
      gB.hostPtr[i] = si.gVals[i]
      betaB.hostPtr[i] = si.betaBits[i]
    for i in 0 ..< stateElems:
      stateB.hostPtr[i] = si.state0[i]
    for i in 0 ..< yElems:
      yB.hostPtr[i] = 0

  proc launch(si: PrefillInputs) =
    engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
      (kernelName, statePA,
        (yPA, kPA, qPA, vPA, betaPA, gPA,
          int32(Hv), int32(Hk), int32(hkRatio), int32(T)))
    inc launches

  proc sentinels(si: PrefillInputs) =
    assertTailZero(yB, yElems)
    assertTailZero(stateB, stateElems)
    assertReadUnchanged(kB, si.kBits)
    assertReadUnchanged(qB, si.qBits)
    assertReadUnchanged(vB, si.vBits)
    assertReadUnchanged(betaB, si.betaBits)
    for i in 0 ..< bhMax * T:
      doAssert gB.hostPtr[i] == si.gVals[i], "kernel-read buffer modified"

  proc judge(si: PrefillInputs, record: bool) =
    ## Naive references, bars, launch, sentinels and the per-element judgment.
    ## `record` false marks the determinism relaunch, bit-compared against the first pass.
    fillInputs(si)
    # fp64 widened inputs, the reference sides consume exactly the element bits
    var s0w = Cube[float64](planes: bhMax, rows: Dv, cols: Dk)
    s0w.data = newSeq[float64](stateElems)
    for i in 0 ..< stateElems:
      s0w.data[i] = si.state0[i].float64
    var qw = Cube[float64](planes: qkRows, rows: T, cols: Dk)
    var kw = Cube[float64](planes: qkRows, rows: T, cols: Dk)
    qw.data = newSeq[float64](qkRows * T * Dk)
    kw.data = newSeq[float64](qkRows * T * Dk)
    for i in 0 ..< qkRows * T * Dk:
      qw.data[i] = si.qBits[i].widenTo(dt).float64
      kw.data[i] = si.kBits[i].widenTo(dt).float64
    var vw = Cube[float64](planes: bhMax, rows: T, cols: Dv)
    vw.data = newSeq[float64](bhMax * T * Dv)
    for i in 0 ..< bhMax * T * Dv:
      vw.data[i] = si.vBits[i].widenTo(dt).float64
    var bw = Mat[float64](rows: bhMax, cols: T)
    bw.data = newSeq[float64](bhMax * T)
    var gw = Mat[float64](rows: bhMax, cols: T)
    gw.data = newSeq[float64](bhMax * T)
    for i in 0 ..< bhMax * T:
      bw.data[i] = si.betaBits[i].widenTo(dt).float64
      gw.data[i] = si.gVals[i].float64

    # the reference walks are per-sequence, B sequences take B independent reference
    # calls on per-sequence input slices, results assembled on the stacked axes
    var sN = newSeq[float64](stateElems)
    var yN = newSeq[float64](yElems)
    var sPerSeq = newSeq[float64](stateElems)
    var yPerGlobal = newSeq[float64](yElems)
    for b in 0 ..< B:
      let s0Seq = sliceCube(s0w, b * Hv, Hv)
      let qSeq = sliceCube(qw, b * Hk, Hk)
      let kSeq = sliceCube(kw, b * Hk, Hk)
      let vSeq = sliceCube(vw, b * Hv, Hv)
      let bSeq = sliceMat(bw, b * Hv, Hv)
      let gSeq = sliceMat(gw, b * Hv, Hv)
      let chunked = gdnPrefillChunked(s0Seq, qSeq, kSeq, vSeq, bSeq, gSeq,
        Hv, Hk, hkRatio, chunkLen)
      var sWalk = copyTensor(s0Seq)
      var yWalk = Cube[float64](planes: Hv, rows: T, cols: Dv)
      yWalk.data = newSeq[float64](Hv * T * Dv)
      gdnPrefillPerToken[float64](sWalk, yWalk, qSeq, kSeq, vSeq, bSeq, gSeq,
        Hv, Hk, hkRatio)
      for i in 0 ..< Hv * Dv * Dk:
        sN[(b * Hv) * Dv * Dk + i] = chunked.state.data[i]
        sPerSeq[(b * Hv) * Dv * Dk + i] = sWalk.data[i]
      for i in 0 ..< Hv * T * Dv:
        yN[(b * Hv) * T * Dv + i] = chunked.y.data[i]
        yPerGlobal[(b * Hv) * T * Dv + i] = yWalk.data[i]

    let bars = gdnChunkTraceBars(s0w, qw, kw, vw, bw, gw,
      Hv, Hk, hkRatio, T, chunkLen, Dv, Dk, uStep)

    launch(si)
    sentinels(si)

    # the bar trace and the reference tier must agree to fp64 noise on y, a loose
    # consistency check on the magnitude source, never the judged divergence
    for bh in 0 ..< bhMax:
      for t in 0 ..< T:
        for r in 0 ..< Dv:
          let i = (bh * T + t) * Dv + r
          doAssert abs(yN[i] - yPerGlobal[i]) <=
            1e-9 * (1.0 + abs(yN[i])),
            "the bar trace and the reference tier disagree on y"

    # continuity bar, the fp64 chunked-vs-walk reassociation term on top of the state bar
    var contBar = newSeq[float64](stateElems)
    for i in 0 ..< stateElems:
      contBar[i] = bars.barS[i] +
        2.0 * float64(T * Dk) * U64 * max(abs(sN[i]), abs(sPerSeq[i]))

    if record:
      for bh in 0 ..< bhMax:
        let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
        doAssert hk >= 0 and hk < qkRows, "head mapping inside the key-head count"
        for t in 0 ..< T:
          for r in 0 ..< Dv:
            let i = (bh * T + t) * Dv + r
            let yWant = yN[i]
            let yGot = yB.hostPtr[i].widenTo(dt).float64
            let yDiff = abs(yGot - yWant)
            doAssert yDiff <= bars.barY[i],
              &"y outside the bar at (bh {bh}, t {t}, r {r}): " &
              &"{yDiff:.3e} > {bars.barY[i]:.3e}"
            let uAt = ulpStepAt(ulpG, yWant)
            if uAt > 0.0 and yDiff > 0.0:
              worstYUlp = max(worstYUlp, yDiff / uAt)
            if yDiff == 0.0:
              inc yExact
            inc yTotal
            if bars.barY[i] > 0.0: worstYUse = max(worstYUse, yDiff / bars.barY[i])
      for i in 0 ..< stateElems:
        let sWant = sN[i]
        let sDiff = abs(stateB.hostPtr[i].float64 - sWant)
        doAssert sDiff <= bars.barS[i],
          &"state outside the bar at element {i}: {sDiff:.3e} > {bars.barS[i]:.3e}"
        worstState = max(worstState, sDiff)
        if bars.barS[i] > 0.0: worstStateUse = max(worstStateUse, sDiff / bars.barS[i])
        let cDiff = abs(stateB.hostPtr[i].float64 - sPerSeq[i])
        doAssert cDiff <= contBar[i],
          &"continuity outside the bar at element {i}: {cDiff:.3e} > {contBar[i]:.3e}"
        if contBar[i] > 0.0: worstCont = max(worstCont, cDiff / contBar[i])

  proc record(): tuple[st: seq[float32], y: seq[uint16]] =
    var st = newSeq[float32](stateElems)
    for i in 0 ..< stateElems: st[i] = stateB.hostPtr[i]
    var yy = newSeq[uint16](yElems)
    for i in 0 ..< yElems: yy[i] = yB.hostPtr[i]
    (st, yy)

  var rng = initPropRng(seed)
  var case0: tuple[st: seq[float32], y: seq[uint16]]
  const cases = 4
  for caseId in 0 ..< cases:
    let si = takeInputs(dt, rng, bhMax, qkRows, T, Dv, Dk, gLoOverride,
      gHiOverride, betaZero)
    judge(si, record = true)
    if caseId == 0: case0 = record()
  # determinism relaunch of case 0, bit-identical across launches
  block determinism:
    var rng0 = initPropRng(seed)
    let si = takeInputs(dt, rng0, bhMax, qkRows, T, Dv, Dk, gLoOverride,
      gHiOverride, betaZero)
    judge(si, record = false)
    let again = record()
    for i in 0 ..< stateElems:
      doAssert again.st[i] == case0.st[i], "state differs run to run"
    for i in 0 ..< yElems:
      doAssert again.y[i] == case0.y[i], "y differs run to run"

  echo &"[{label} {ulpDatatypeName(ulpG)} T={T} C={chunkLen}] cases={cases} launches={launches} | " &
    &"state worst |ΔS| {worstState:.3e}, worst bar usage {worstStateUse:.3f} | " &
    &"continuity worst usage {worstCont:.3f} | y worst {worstYUlp:.2f} {ulpDatatypeName(ulpG)} ulp, " &
    &"bit-exact {yExact}/{yTotal}, worst bar usage {worstYUse:.3f}"
  suiteCases += cases
  suiteLaunches += launches
  suiteWorstUse = max(suiteWorstUse, max(max(worstStateUse, worstYUse), worstCont))
  suiteWorstState = max(suiteWorstState, worstState)
  suiteWorstCont = max(suiteWorstCont, worstCont)
  suiteWorstYUlp = max(suiteWorstYUlp, worstYUlp)
  suiteYExact += yExact
  suiteYTotal += yTotal

# ─── Main, the shape × dtype matrix ──────────────────────────────────

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(GdnPrefillMsl)

  proc secF16T8 =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 8, 32, 0xC04D0401'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16T64 =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 64, 32, 0xC04D0402'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16Gqa =
    let t0 = epochTime()
    runCase(engine, kFloat16, 4, 2, 2, 2, 256, 32, 0xC04D0403'u64,
      "gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16Tail =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 100, 32, 0xC04D0404'u64,
      "tail Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16C64 =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 64, 64, 0xC04D0405'u64,
      "chunk64 Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16T8 =
    let t0 = epochTime()
    runCase(engine, kBfloat16, 1, 1, 1, 1, 8, 32, 0xC04D0406'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Tail =
    let t0 = epochTime()
    runCase(engine, kBfloat16, 1, 1, 1, 1, 100, 32, 0xC04D0407'u64,
      "tail Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Gqa64 =
    let t0 = epochTime()
    runCase(engine, kBfloat16, 4, 2, 2, 2, 256, 64, 0xC04D0408'u64,
      "gqa64 Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secEdgeNearZeroG =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 64, 32, 0xC04D04A7'u64,
      "edge g->0- Hk=1/Hv=1/B=1", -0.001'f32, 0.0'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secEdgeBetaZero =
    let t0 = epochTime()
    runCase(engine, kBfloat16, 1, 1, 1, 1, 64, 32, 0xC04D04A8'u64,
      "edge beta=0 Hk=1/Hv=1/B=1", betaZero = true)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  secEdgeNearZeroG()
  secEdgeBetaZero()
  secF16T8()
  secF16T64()
  secF16Gqa()
  secF16Tail()
  secF16C64()
  secBf16T8()
  secBf16Tail()
  secBf16Gqa64()
  echo &"CERAMIC GDN PREFILL VERDICT: cases={suiteCases} launches={suiteLaunches} " &
    &"state worst |ΔS| {suiteWorstState:.3e}, continuity worst usage {suiteWorstCont:.3f}, " &
    &"y worst {suiteWorstYUlp:.2f} ulp, worst bar usage {suiteWorstUse:.3f}, " &
    &"y bit-exact {suiteYExact}/{suiteYTotal}"

main()
