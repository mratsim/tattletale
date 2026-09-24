# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

## Run commands, from the repo root (the aggregate runner is nim test_positron_naive):
## - nim c -r -d:release --warnings:off --outdir:build/tests workspace/positron/tests/ceramic/t_ceramic_kda_prefill.nim
##
## Ceramic KDA prefill suite, chunked scan against the naive tier:
##
## | judgment   | naive side                                                                                      |
## | ---------- | ----------------------------------------------------------------------------------------------- |
## | y, state   | the fp64 chunked WY/UT reference `kdaPrefillChunked` (tests/naive/naive_kda.nim)                |
## | continuity | the fp64 per-token walk `kdaPrefillPerToken`, the ceramic chunked scan's final state against it |
##
## - the naive-side continuity pair (chunked vs per-token, y and final state) lives in tests/naive/t_naive_kda.nim
## - the naive references are per-sequence, B sequences take B naive calls on per-sequence input slices
## - per-sequence results are assembled on the stacked axes
##
## Inputs, all seeded xorshift64, fixture-free:
##
## | input        | rule                                                                                     |
## | ------------ | ---------------------------------------------------------------------------------------- |
## | q, k         | l2-normalized per (head, token) row in fp32, the kernel contract's post-l2norm f32 shape |
## | v, state0    | [-1, 1), the initial state never zero                                                    |
## | beta         | [0.2, 0.8) f32                                                                           |
## | g            | [-0.5, -0.01) f32 log-decay, one per KEY channel, (B·Hk, T, Dk)                          |
## | cumulogdecay | the per-channel f32 cumulative log decay within each chunk, host-computed from g         |
##
## - the q, k normalization also keeps the delta-rule recursion bounded over 256 tokens in fp16 y range
## - the f32 inputs are the shared values, the naive sides widen them exactly, no input rounding divergence
##
## Band model, stated before measurement and judged per element, u₃₂ = 2⁻²⁴ the fp32
## unit roundoff, a reassociation of the per-token recurrence over the per-channel
## decay γ_c = exp(g_c), one decay per KEY channel:
##
## | reassociation | structure                                                                                          |
## | ------------- | -------------------------------------------------------------------------------------------------- |
## | decay         | carried as the per-channel cumulative log decay cumulogdecay, pair decay dT·invd_s per key channel |
## | updates       | the token updates u_t solved in token order through the A-matrix recurrence                        |
## | carry         | one per-channel decayed carry read plus the u outer products, assembled once per chunk             |
##
##   per chunk:   cumulogdecay ─→ per token t: dT, G_t read ─→ u_t solve ─→ y_t store
##                (t in token order)                          └─────────┐
##            └──→ S ← dEnd ⊙ carry read + Σ_s (dEnd·invd_s ⊙ k_s) [x] u_s
##
## - the per-token walk instead applies exp(g) per token per channel, reading and updating the state each step
## - both naive spellings run fp64, the ceramic core runs fp32 state math with element-dtype handoffs
## - the judged divergence is the ceramic side's rounding against the fp64 chunked reference structure
##
## | site                    | bound                                                                                        |
## | ----------------------- | -------------------------------------------------------------------------------------------- |
## | cumulogdecay (host f32) | per entry ≤ (i+1)·u₃₂·cmax, cmax the chunk's max abs(cumulogdecay) per channel               |
## | decay factors           | relative ≤ cErr + u₃₂·abs(cumulogdecay) + 4·u₃₂ (log2e multiply, exp2 form)                  |
## | pair decay              | relative ≤ relT[t] + relT[s] + u₃₂ (dT·invd_s, two exp2 forms plus the product)              |
## | k·k / q̃·k dots         | Dk·u₃₂ of the absolute sum (elementwise round plus row-sum tree)                             |
## | S·k̃ / S·q̃ reads       | per-channel decay rel terms, Dk·u₃₂ of the absolute sum, the carried state error Σ abs(k)·ΔS |
## | q̃ scale                | relative ≤ 4·u₃₂ (f32 qScale vs the naive f64 divide, plus the division rounding)            |
## | u solve                 | β·(base error + Σ abs(A)·Δu + ΔA·abs(u) + t·u₃₂·Σ abs(A·u))                                  |
## | chunk carry             | decayed old state per channel + Σ abs(pd·k)·Δu + cLen·u₃₂·Σ abs(pd·k·u)                      |
## | y store                 | one element-dtype RNE, u_step·abs(y) plus the subnormal grid floor                           |
##
## - the exponent-sensitivity factors collapse (ln2·log2e = 1), a log-domain error cErr mapping to an equal relative error on exp2(c·log2e)
## - the reassociation budget per chunk is C token terms per sum (solve, y, carry) and Dk terms per dot over T/C chunks
##
## - every site's bound is a triangle-inequality sum of nonnegative per-op roundings, the model is additive
## - the per-element bars below are host-computed from an fp64 trace walk mirroring the chunked formulas, magnitudes only
## - the measured divergence corroborates the model and never sets a bar
##
## | bar            | bound                                                                                         |
## | -------------- | --------------------------------------------------------------------------------------------- |
## | y (bh, t, r)   | Δy(t, r) + u_step·abs(y_ref) + 2⁻²⁵                                                           |
## | state (bh,r,c) | the carried ΔS recursion across chunks, chunk-local terms in the table below                  |
## | continuity     | barS + 2·(T·Dk)·2⁻⁵³·max(abs(S_chunked), abs(S_walk)), the fp64 chunked-vs-walk reassociation |
##
## ΔS recursion per chunk, all terms nonnegative, computed per element per channel:
##
## | component       | bound                                               |
## | --------------- | --------------------------------------------------- |
## | decayed state   | dEnd·(ΔS + abs(S)·(u₃₂ + relEnd)) + u₃₂·abs(dEnd·S) |
## | u outer product | Σ_s abs(pd·k_s)·(Δu_s + abs(u_s)·(relPd + 3·u₃₂))   |
## | sum roundings   | cLen·u₃₂·Σ_s abs(pd·k_s·u_s) + u₃₂·abs(S_new)       |
##
## - the β = 0 edge has exact-zero delta terms, u = 0, both spellings judging the same decay arithmetic, in-band
##
##
## Shapes (Dv = 16, TileR = 8, grid (Dv div TileR, B·Hv), 32 lanes, one launch walks all chunks):
##
## | shape    | Hk | Hv | batch | hkRatio | dtype | T   | chunk | Dk |
## | -------- | --- | --- | ----- | ------- | ----- | --- | ----- | --- |
## | baseline | 1  | 1  | 1     | 1       | fp16  | 8   | 32    | 32 |
## | baseline | 1  | 1  | 1     | 1       | fp16  | 64  | 32    | 32 |
## | gqa      | 2  | 4  | 2     | 2       | fp16  | 256 | 32    | 32 |
## | tail     | 1  | 1  | 1     | 1       | fp16  | 100 | 32    | 32 |
## | chunk64  | 1  | 1  | 1     | 1       | fp16  | 64  | 64    | 32 |
## | dk64     | 1  | 1  | 1     | 1       | fp16  | 64  | 32    | 64 |
## | beta0    | 1  | 1  | 1     | 1       | fp16  | 64  | 32    | 32 |
## | baseline | 1  | 1  | 1     | 1       | bf16  | 8   | 32    | 32 |
## | tail     | 1  | 1  | 1     | 1       | bf16  | 100 | 32    | 32 |
## | gqa64    | 2  | 4  | 2     | 2       | bf16  | 256 | 64    | 32 |
## | overflow | 1  | 1  | 1     | 1       | fp16  | 64  | 64    | 32 |
## | overflow | 1  | 1  | 1     | 1       | bf16  | 64  | 64    | 32 |
##
## - the T = 100 case carries a non-divisible tail chunk, 3 full chunks plus 4 tokens
## - the overflow fixture's g ≈ −3 per token per channel, the 64-token chunk's
## | cumulogdecay |
##
## - every case starts from a non-zero random initial state
## - the GQA shape keeps both head-mapping terms live, sequence 1 holding independent key heads
##
## - fp16 is the element dtype under test, bf16 the range-robust fallback
## - untouched-memory checks per launch, kernel-written buffers stay in their extents, kernel-read buffers stay bit-identical
## - run-to-run determinism, case 0 relaunched bit-identical per combination on Apple M4 Max

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/attn_ssm/gated_delta_net_kda_prefill
import ../naive/naive_rng
import ../naive/naive_tensors
import ../naive/naive_kda
import ceramic_pagebuf
import ceramic_dtype

# ─── Device entries, one per (element dtype, chunk length, Dk) binding ──

const KdaPrefillMsl = metal:
  proc cer_kda_prefill_fp16_c32(
      state: ptr UncheckedArray[float32],
      y, v: ptr UncheckedArray[float16],
      k, q, cumulogdecay, beta: ptr UncheckedArray[float32],
      qScale: float32,
      Hv, Hk, hkRatio, T: int32) {.global.} =
    kdaPrefillChunkScan(state, y, k, q, cumulogdecay, v, beta, qScale,
      Hv, Hk, hkRatio, T, 32, 16, 8, 32)

  proc cer_kda_prefill_fp16_c64(
      state: ptr UncheckedArray[float32],
      y, v: ptr UncheckedArray[float16],
      k, q, cumulogdecay, beta: ptr UncheckedArray[float32],
      qScale: float32,
      Hv, Hk, hkRatio, T: int32) {.global.} =
    kdaPrefillChunkScan(state, y, k, q, cumulogdecay, v, beta, qScale,
      Hv, Hk, hkRatio, T, 32, 16, 8, 64)

  proc cer_kda_prefill_fp16_dk64_c32(
      state: ptr UncheckedArray[float32],
      y, v: ptr UncheckedArray[float16],
      k, q, cumulogdecay, beta: ptr UncheckedArray[float32],
      qScale: float32,
      Hv, Hk, hkRatio, T: int32) {.global.} =
    kdaPrefillChunkScan(state, y, k, q, cumulogdecay, v, beta, qScale,
      Hv, Hk, hkRatio, T, 64, 16, 8, 32)

  proc cer_kda_prefill_bf16_c32(
      state: ptr UncheckedArray[float32],
      y, v: ptr UncheckedArray[bfloat16],
      k, q, cumulogdecay, beta: ptr UncheckedArray[float32],
      qScale: float32,
      Hv, Hk, hkRatio, T: int32) {.global.} =
    kdaPrefillChunkScan(state, y, k, q, cumulogdecay, v, beta, qScale,
      Hv, Hk, hkRatio, T, 32, 16, 8, 32)

  proc cer_kda_prefill_bf16_c64(
      state: ptr UncheckedArray[float32],
      y, v: ptr UncheckedArray[bfloat16],
      k, q, cumulogdecay, beta: ptr UncheckedArray[float32],
      qScale: float32,
      Hv, Hk, hkRatio, T: int32) {.global.} =
    kdaPrefillChunkScan(state, y, k, q, cumulogdecay, v, beta, qScale,
      Hv, Hk, hkRatio, T, 32, 16, 8, 64)

# ─── Host tolerance-model constants ───────────────────────────────────

const
  DecExp = 4.0 * U32                 # exp2(c·log2e) form relative bound, mul + exp2
  QScaleRel = 4.0 * U32              # f32 qScale vs the naive f64 divide, plus the division
  UBf16 = 3.90625e-3                 # 2⁻⁸, the bf16 unit roundoff
  UF16 = 4.8828125e-4                # 2⁻¹¹, the fp16 unit roundoff
  FloorSub = 2.9802322387695312e-8   # 2⁻²⁵, half the constant fp16 subnormal ulp,
                                     # the rounding floor once |y| falls subnormal,
                                     # also covering the bf16 subnormal grid
  U64 = 1.1102230246251565e-16       # 2⁻⁵³, the fp64 unit roundoff
  Log2e = 1.4426950408889634'f64     # log2(e), the exp2 form's constant
  FlushLog2 = 126.0                  # fp32 normal range's floor in log2 units,
                                     # a flushed factor's error is its full magnitude

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

proc kdaChunkTraceBars(
    s0w: NaiveCube[float64], qw, kw, gw: NaiveCube[float64], vw: NaiveCube[float64],
    bw: NaiveMat[float64],
    Hv, Hk, hkRatio, T, chunkLen, Dv, Dk: int,
    uStep: float64, underflowFloors = false): TraceBars =
  ## `underflowFloors` adds the fp32 exp2 underflow floor to every decay
  ## factor's relative bound
  ##
  ## - a factor whose exp2 argument drops past 126/log2e ≈ 87.3 hits fp32 zero,
  ##   the flushed factor's error is its full true magnitude
  ## - the fixture's factors all sit past the bound, the recorded model's
  ##   relative bounds alone would underestimate there
  let bhMax = s0w.planes
  result.barY = newSeq[float64](bhMax * T * Dv)
  result.barS = newSeq[float64](bhMax * Dv * Dk)
  let qScale = sqrt(float64(Dk))
  var aAbsA = newSeq[float64](chunkLen * chunkLen)
  var aRelA = newSeq[float64](chunkLen * chunkLen)
  var aAbsSumA = newSeq[float64](chunkLen * chunkLen)
  var aValA = newSeq[float64](chunkLen * chunkLen)
  var bAbsA = newSeq[float64](chunkLen * chunkLen)
  var bRelA = newSeq[float64](chunkLen * chunkLen)
  var bAbsSumA = newSeq[float64](chunkLen * chunkLen)
  var bValA = newSeq[float64](chunkLen * chunkLen)
  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    var carry = newSeq[float64](Dv * Dk)
    var dS = newSeq[float64](Dv * Dk)
    for r in 0 ..< Dv:
      for dk in 0 ..< Dk:
        carry[r * Dk + dk] = s0w.data[(bh * Dv + r) * Dk + dk]
    var cumulogdecayRef = newSeq[float64](chunkLen * Dk)
    var relT = newSeq[float64](chunkLen * Dk)
    var uRef = newSeq[float64](chunkLen * Dv)
    var du = newSeq[float64](chunkLen * Dv)
    var c0 = 0
    while c0 < T:
      let cLen = min(chunkLen, T - c0)
      # reference cumulative log decay in fp64 from g, one row per token per channel,
      # with the f32 host-cumulogdecay drift bound per channel
      for dk in 0 ..< Dk:
        cumulogdecayRef[dk] = gw.data[(hk * T + c0) * Dk + dk]
      var cmax = newSeq[float64](Dk)
      for dk in 0 ..< Dk:
        cmax[dk] = abs(cumulogdecayRef[dk])
      for i in 1 ..< cLen:
        for dk in 0 ..< Dk:
          cumulogdecayRef[i * Dk + dk] = cumulogdecayRef[(i - 1) * Dk + dk] +
            gw.data[(hk * T + c0 + i) * Dk + dk]
      for i in 0 ..< cLen:
        for dk in 0 ..< Dk:
          cmax[dk] = max(cmax[dk], abs(cumulogdecayRef[i * Dk + dk]))
      for i in 0 ..< cLen:
        for dk in 0 ..< Dk:
          # decay factor rel, f32 cumulogdecay drift + the log2e multiply + the exp2 form
          relT[i * Dk + dk] = float64(i + 1) * U32 * cmax[dk] +
            U32 * abs(cumulogdecayRef[i * Dk + dk]) + DecExp
          if underflowFloors and -cumulogdecayRef[i * Dk + dk] * Log2e > FlushLog2:
            # the per-token decay factor flushes to fp32 zero, the flushed
            # factor's full magnitude joins the bound
            relT[i * Dk + dk] += 1.0
      for t in 0 ..< cLen:
        let gt = c0 + t
        let betaT = bw.data[bh * T + gt]
        # pair (t, s) magnitudes, per channel, shared across the state rows
        for s in 0 .. t:
          var aAbs = 0.0'f64
          var aRel = 0.0'f64
          var aAbsSum = 0.0'f64
          var aVal = 0.0'f64
          var bAbs = 0.0'f64
          var bRel = 0.0'f64
          var bAbsSum = 0.0'f64
          var bVal = 0.0'f64
          for dk in 0 ..< Dk:
            let kt = kw.data[(hk * T + gt) * Dk + dk]
            let ks = kw.data[(hk * T + c0 + s) * Dk + dk]
            let qt = qw.data[(hk * T + gt) * Dk + dk] / qScale
            let pd = exp(cumulogdecayRef[t * Dk + dk] - cumulogdecayRef[s * Dk + dk])
            var relPd = relT[t * Dk + dk] + relT[s * Dk + dk] + U32
            if underflowFloors:
              # the pair decay's argument is cumulogdecay_s − cumulogdecay_t ≥ 0, a pair past
              # the flush bound loses its factor to fp32 zero entirely
              if (cumulogdecayRef[s * Dk + dk] - cumulogdecayRef[t * Dk + dk]) * Log2e > FlushLog2:
                relPd += 1.0
            let kkAbs = pd * abs(kt * ks)
            let qkAbs = pd * abs(qt * ks)
            aAbs += kkAbs
            aRel += kkAbs * relPd
            aAbsSum += abs(kt * ks)
            aVal += pd * kt * ks
            bAbs += qkAbs
            bRel += qkAbs * relPd
            bAbsSum += abs(qt * ks)
            bVal += pd * qt * ks
          aAbsA[t * chunkLen + s] = aAbs
          aRelA[t * chunkLen + s] = aRel
          aAbsSumA[t * chunkLen + s] = aAbsSum
          aValA[t * chunkLen + s] = aVal
          bAbsA[t * chunkLen + s] = bAbs
          bRelA[t * chunkLen + s] = bRel
          bAbsSumA[t * chunkLen + s] = bAbsSum
          bValA[t * chunkLen + s] = bVal
        # decayed carry reads, the A-magnitude and B-magnitude forms, per state row
        for r in 0 ..< Dv:
          var kvAbs = 0.0'f64
          var dSG = 0.0'f64
          var kvRel = 0.0'f64
          var kvN = 0.0'f64
          for dk in 0 ..< Dk:
            let dTk = exp(cumulogdecayRef[t * Dk + dk])
            let term = dTk * kw.data[(hk * T + gt) * Dk + dk] * carry[r * Dk + dk]
            kvN += term
            kvAbs += abs(term)
            kvRel += abs(term) * relT[t * Dk + dk]
            dSG += abs(kw.data[(hk * T + gt) * Dk + dk]) * dS[r * Dk + dk]
          let gRef = kvN
          let dG = kvRel + 2.0 * U32 * kvAbs +
            float64(Dk) * U32 * kvAbs + dSG
          let vRef = vw.data[(bh * T + gt) * Dv + r]
          let base1 = betaT * (vRef - gRef)
          let dBase = betaT * (dG + U32 * (abs(vRef) + abs(gRef)))
          var duAcc = 0.0'f64
          var accAbs = 0.0'f64
          var acc = 0.0'f64
          for s in 0 ..< t:
            let dA = aRelA[t * chunkLen + s] + 2.0 * U32 * aAbsA[t * chunkLen + s] +
              float64(Dk) * U32 * aAbsSumA[t * chunkLen + s]
            duAcc += aAbsA[t * chunkLen + s] * du[s * Dv + r] +
              dA * abs(uRef[s * Dv + r])
            let aTerm = aAbsA[t * chunkLen + s] * abs(uRef[s * Dv + r])
            accAbs += aTerm
            acc += aValA[t * chunkLen + s] * uRef[s * Dv + r]
          uRef[t * Dv + r] = base1 - betaT * acc
          du[t * Dv + r] = dBase + betaT * duAcc + betaT * float64(t) * U32 * accAbs +
            betaT * U32 * (abs(base1) + abs(acc))
          # output bar, the decayed carry read plus the B-form sum
          var qvAbs = 0.0'f64
          var dSGq = 0.0'f64
          var qvRel = 0.0'f64
          var qvN = 0.0'f64
          for dk in 0 ..< Dk:
            let dTk = exp(cumulogdecayRef[t * Dk + dk])
            let term = dTk * (qw.data[(hk * T + gt) * Dk + dk] / qScale) *
              carry[r * Dk + dk]
            qvN += term
            qvAbs += abs(term)
            qvRel += abs(term) * relT[t * Dk + dk]
            dSGq += abs(qw.data[(hk * T + gt) * Dk + dk] / qScale) * dS[r * Dk + dk]
          let hRef = qvN
          let dH = qvRel + 2.0 * U32 * qvAbs + QScaleRel * qvAbs +
            float64(Dk) * U32 * qvAbs + dSGq
          var yAbsSum = abs(hRef)
          var yErr = dH
          var yRef = hRef
          for s in 0 .. t:
            let dB = bRelA[t * chunkLen + s] + 2.0 * U32 * bAbsA[t * chunkLen + s] +
              float64(Dk) * U32 * bAbsSumA[t * chunkLen + s] +
              QScaleRel * bAbsA[t * chunkLen + s]
            let bTerm = bAbsA[t * chunkLen + s] * abs(uRef[s * Dv + r])
            yAbsSum += bTerm
            yErr += bAbsA[t * chunkLen + s] * du[s * Dv + r] + dB * abs(uRef[s * Dv + r])
            yRef += bValA[t * chunkLen + s] * uRef[s * Dv + r]
          # the adds round once per term, the element-dtype store rounds once per element
          let dY = yErr + float64(t + 1) * U32 * yAbsSum + U32 * abs(yRef)
          result.barY[(bh * T + gt) * Dv + r] =
            dY + uStep * abs(yRef) + FloorSub
      # carry out of the chunk, the reference formula and its bound, per channel
      for r in 0 ..< Dv:
        for dk in 0 ..< Dk:
          let idx = r * Dk + dk
          let sOld = carry[idx]
          let dEnd = exp(cumulogdecayRef[(cLen - 1) * Dk + dk])
          let relEnd = relT[(cLen - 1) * Dk + dk]
          let decTerm = dEnd * sOld
          var sumTerm = 0.0'f64
          var sumAbs = 0.0'f64
          var sumErr = 0.0'f64
          for s in 0 ..< cLen:
            let pdEnd = exp(cumulogdecayRef[(cLen - 1) * Dk + dk] - cumulogdecayRef[s * Dk + dk])
            let ks = kw.data[(hk * T + c0 + s) * Dk + dk]
            let relPdEnd = relT[(cLen - 1) * Dk + dk] + relT[s * Dk + dk] + U32
            let wAbs = abs(pdEnd * ks)
            let pTerm = pdEnd * ks * uRef[s * Dv + r]
            sumTerm += pTerm
            sumAbs += abs(pTerm)
            sumErr += wAbs * (du[s * Dv + r] +
              abs(uRef[s * Dv + r]) * (relPdEnd + 3.0 * U32))
          let sNew = decTerm + sumTerm
          dS[idx] = dEnd * (dS[idx] + abs(sOld) * (U32 + relEnd)) +
            U32 * abs(decTerm) + sumErr +
            float64(cLen) * U32 * sumAbs + U32 * abs(sNew)
          carry[idx] = sNew
      c0 += cLen
    for r in 0 ..< Dv:
      for dk in 0 ..< Dk:
        result.barS[(bh * Dv + r) * Dk + dk] = dS[r * Dk + dk]

# ─── One case, seeded inputs → naive references → launch → judgment ──

proc sliceCube[T](c: NaiveCube[T], plane0, planes: int): NaiveCube[T] =
  ## Per-sequence slice of a stacked cube, the naive references are per-sequence.
  result = NaiveCube[T](planes: planes, rows: c.rows, cols: c.cols)
  result.data = newSeq[T](planes * c.rows * c.cols)
  for i in 0 ..< planes * c.rows * c.cols:
    result.data[i] = c.data[plane0 * c.rows * c.cols + i]

proc sliceMat[T](m: NaiveMat[T], row0, rows: int): NaiveMat[T] =
  ## Per-sequence slice of a stacked head-axis matrix.
  result = NaiveMat[T](rows: rows, cols: m.cols)
  result.data = newSeq[T](rows * m.cols)
  for i in 0 ..< rows * m.cols:
    result.data[i] = m.data[row0 * m.cols + i]

type PrefillInputs = object
  ## One case's seeded inputs, the f32 values shared by the kernel and the naive
  ## sides through their exact widenings, the per-channel cumulogdecay host-computed
  ##
  ## | field        | shape              |
  ## | ------------ | ------------------ |
  ## | q, k         | (B·Hk, T, Dk) f32  |
  ## | g            | (B·Hk, T, Dk) f32  |
  ## | cumulogdecay | (B·Hk, T, Dk) f32  |
  ## | vBits        | (B·Hv, T, Dv)      |
  ## | beta         | (B·Hv, T) f32      |
  ## | state0       | (B·Hv, Dv, Dk) f32 |
  qVals: seq[float32]
  kVals: seq[float32]
  gVals: seq[float32]
  cumulogdecayVals: seq[float32]
  vBits: seq[uint16]
  betaVals: seq[float32]
  state0: seq[float32]

proc l2NormalizeRowsF32(dst: var seq[float32], rows, cols: int, rng: var NaiveRng) =
  ## Fills `dst` with l2-normalized f32 rows, the kernel contract's post-l2norm
  ## query/key shape:
  ##
  ## - each row is normalized to unit l2 norm, the norm taken over the f32 row values
  for r in 0 ..< rows:
    var norm2 = 0.0'f64
    for c in 0 ..< cols:
      let x = rng.nextF32(-1.0'f32, 1.0'f32)
      norm2 += x.float64 * x.float64
      dst[r * cols + c] = x
    let inv = 1.0 / sqrt(norm2)
    for c in 0 ..< cols:
      dst[r * cols + c] = (dst[r * cols + c].float64 * inv).float32

proc hostCumulogdecay(dst: var seq[float32], g: seq[float32], qkRows, T, Dk, chunkLen: int) =
  ## Computes the per-channel f32 cumulative log decay, the chunk-relative prefix
  ## of g per (head, chunk, channel), the kernel's decay input, matching the
  ## chunked reference's cumulogdecay structure.
  for hk in 0 ..< qkRows:
    for t in 0 ..< T:
      for dk in 0 ..< Dk:
        let gi = (hk * T + t) * Dk + dk
        if t mod chunkLen == 0:
          dst[gi] = g[gi]
        else:
          dst[gi] = dst[gi - Dk] + g[gi]

proc takeInputs(dt: ScalarKind, rng: var NaiveRng, bhMax, qkRows, T, Dv, Dk, chunkLen: int, betaZero: bool, overflowG = false): PrefillInputs =
  ## `overflowG` generates the decay-overflow fixture's g, |g| ≈ 3 per token
  ## per channel so the 64-token chunk's |cumulogdecay| crosses the exp2 overflow
  ## bound 88.7 inside the chunk
  var qVals = newSeq[float32](qkRows * T * Dk)
  var kVals = newSeq[float32](qkRows * T * Dk)
  l2NormalizeRowsF32(qVals, qkRows * T, Dk, rng)
  l2NormalizeRowsF32(kVals, qkRows * T, Dk, rng)
  var gVals = newSeq[float32](qkRows * T * Dk)
  for i in 0 ..< qkRows * T * Dk:
    gVals[i] = if overflowG: rng.nextF32(-3.2'f32, -2.8'f32)
               else: rng.nextF32(-0.5'f32, -0.01'f32)
  var cumulogdecayVals = newSeq[float32](qkRows * T * Dk)
  hostCumulogdecay(cumulogdecayVals, gVals, qkRows, T, Dk, chunkLen)
  var vBits = newSeq[uint16](bhMax * T * Dv)
  for i in 0 ..< bhMax * T * Dv:
    vBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
  var betaVals = newSeq[float32](bhMax * T)
  for i in 0 ..< bhMax * T:
    betaVals[i] = if betaZero: 0.0'f32 else: rng.nextF32(0.2'f32, 0.8'f32)
  var state0 = newSeq[float32](bhMax * Dv * Dk)
  for i in 0 ..< bhMax * Dv * Dk:
    state0[i] = rng.nextF32(-1.0'f32, 1.0'f32)
  result = PrefillInputs(qVals: qVals, kVals: kVals, gVals: gVals,
    cumulogdecayVals: cumulogdecayVals, vBits: vBits, betaVals: betaVals, state0: state0)

var suiteCases, suiteLaunches, suiteYExact, suiteYTotal = 0
var suiteWorstUse, suiteWorstState, suiteWorstCont, suiteWorstYUlp = 0.0'f64

proc runCase(engine: HwEngine, dt: ScalarKind, Hv, Hk, hkRatio, B, T, chunkLen, Dk: int, betaZero: bool, seed: uint64, label: string, overflowG = false) =
  ## One (element dtype, shape) combination, judged per element against the fp64 chunked
  ## reference and the fp64 per-token walk under the band model, relaunched bit-identical.
  ##
  ## `overflowG` runs the decay-overflow fixture, |cumulogdecay| past the exp2 overflow
  ## bound 88.7 inside the first chunk
  ##
  ## - the judgment adds the exact NaN/Inf check on y and the carried state
  ## - the band gains the underflow floors
  const Dv = 16
  const TileR = 8
  let bhMax = B * Hv
  let qkRows = B * Hk
  let stateElems = bhMax * Dv * Dk
  let yElems = bhMax * T * Dv
  let kernelName =
    if dt == kFloat16:
      if Dk == 64: "cer_kda_prefill_fp16_dk64_c32"
      elif chunkLen == 32: "cer_kda_prefill_fp16_c32"
      else: "cer_kda_prefill_fp16_c64"
    else:
      if chunkLen == 32: "cer_kda_prefill_bf16_c32" else: "cer_kda_prefill_bf16_c64"
  let ulpG = if dt == kBfloat16: ulpBf16 else: ulpFp16
  let uStep = binadeStep(ulpG, -1)
  let qScale = float32(sqrt(float64(Dk)))

  var stateB = allocPageBuf[float32](stateElems)
  var yB = allocPageBuf[uint16](yElems)
  var kB = allocPageBuf[float32](qkRows * T * Dk)
  var qB = allocPageBuf[float32](qkRows * T * Dk)
  var cumulogdecayB = allocPageBuf[float32](qkRows * T * Dk)
  var vB = allocPageBuf[uint16](bhMax * T * Dv)
  var betaB = allocPageBuf[float32](bhMax * T)
  defer:
    freePageBuf(stateB); freePageBuf(yB); freePageBuf(kB); freePageBuf(qB)
    freePageBuf(cumulogdecayB); freePageBuf(vB); freePageBuf(betaB)
  var statePA = stateB.pa()
  var yPA = yB.pa()
  var kPA = kB.pa()
  var qPA = qB.pa()
  var cumulogdecayPA = cumulogdecayB.pa()
  var vPA = vB.pa()
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
      kB.hostPtr[i] = si.kVals[i]
      qB.hostPtr[i] = si.qVals[i]
      cumulogdecayB.hostPtr[i] = si.cumulogdecayVals[i]
    for i in 0 ..< bhMax * T * Dv:
      vB.hostPtr[i] = si.vBits[i]
    for i in 0 ..< bhMax * T:
      betaB.hostPtr[i] = si.betaVals[i]
    for i in 0 ..< stateElems:
      stateB.hostPtr[i] = si.state0[i]
    for i in 0 ..< yElems:
      yB.hostPtr[i] = 0

  proc launch(si: PrefillInputs) =
    engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
      (kernelName, statePA,
        (yPA, vPA, kPA, qPA, cumulogdecayPA, betaPA, qScale,
          int32(Hv), int32(Hk), int32(hkRatio), int32(T)))
    inc launches

  proc sentinels(si: PrefillInputs) =
    assertTailZero(yB, yElems)
    assertTailZero(stateB, stateElems)
    assertReadUnchanged(kB, si.kVals)
    assertReadUnchanged(qB, si.qVals)
    assertReadUnchanged(cumulogdecayB, si.cumulogdecayVals)
    assertReadUnchanged(vB, si.vBits)
    assertReadUnchanged(betaB, si.betaVals)

  proc judge(si: PrefillInputs, record: bool) =
    ## Naive references, bars, launch, sentinels and the per-element judgment.
    ## `record` false marks the determinism relaunch, bit-compared against the first pass.
    fillInputs(si)
    # fp64 widened inputs, the naive sides consume exactly the kernel's f32 values
    var s0w = NaiveCube[float64](planes: bhMax, rows: Dv, cols: Dk)
    s0w.data = newSeq[float64](stateElems)
    for i in 0 ..< stateElems:
      s0w.data[i] = si.state0[i].float64
    var qw = NaiveCube[float64](planes: qkRows, rows: T, cols: Dk)
    var kw = NaiveCube[float64](planes: qkRows, rows: T, cols: Dk)
    var gw = NaiveCube[float64](planes: qkRows, rows: T, cols: Dk)
    qw.data = newSeq[float64](qkRows * T * Dk)
    kw.data = newSeq[float64](qkRows * T * Dk)
    gw.data = newSeq[float64](qkRows * T * Dk)
    for i in 0 ..< qkRows * T * Dk:
      qw.data[i] = si.qVals[i].float64
      kw.data[i] = si.kVals[i].float64
      gw.data[i] = si.gVals[i].float64
    var vw = NaiveCube[float64](planes: bhMax, rows: T, cols: Dv)
    vw.data = newSeq[float64](bhMax * T * Dv)
    for i in 0 ..< bhMax * T * Dv:
      vw.data[i] = si.vBits[i].widenTo(dt).float64
    var bw = NaiveMat[float64](rows: bhMax, cols: T)
    bw.data = newSeq[float64](bhMax * T)
    for i in 0 ..< bhMax * T:
      bw.data[i] = si.betaVals[i].float64

    # the naive references are per-sequence, B sequences take B independent naive
    # calls on per-sequence input slices, results assembled on the stacked axes
    var sN = newSeq[float64](stateElems)
    var yN = newSeq[float64](yElems)
    var sPerSeq = newSeq[float64](stateElems)
    var yPerGlobal = newSeq[float64](yElems)
    for b in 0 ..< B:
      let s0Seq = sliceCube(s0w, b * Hv, Hv)
      let qSeq = sliceCube(qw, b * Hk, Hk)
      let kSeq = sliceCube(kw, b * Hk, Hk)
      let gSeq = sliceCube(gw, b * Hk, Hk)
      let vSeq = sliceCube(vw, b * Hv, Hv)
      let bSeq = sliceMat(bw, b * Hv, Hv)
      let chunked = kdaPrefillChunked(s0Seq, qSeq, kSeq, gSeq, vSeq, bSeq,
        Hv, Hk, hkRatio, chunkLen)
      var sWalk = copyOf(s0Seq)
      var yWalk = NaiveCube[float64](planes: Hv, rows: T, cols: Dv)
      yWalk.data = newSeq[float64](Hv * T * Dv)
      kdaPrefillPerToken[float64](sWalk, yWalk, qSeq, kSeq, gSeq, vSeq, bSeq,
        Hv, Hk, hkRatio)
      for i in 0 ..< Hv * Dv * Dk:
        sN[(b * Hv) * Dv * Dk + i] = chunked.state.data[i]
        sPerSeq[(b * Hv) * Dv * Dk + i] = sWalk.data[i]
      for i in 0 ..< Hv * T * Dv:
        yN[(b * Hv) * T * Dv + i] = chunked.y.data[i]
        yPerGlobal[(b * Hv) * T * Dv + i] = yWalk.data[i]

    let bars = kdaChunkTraceBars(s0w, qw, kw, gw, vw, bw,
      Hv, Hk, hkRatio, T, chunkLen, Dv, Dk, uStep,
      underflowFloors = overflowG)

    launch(si)
    sentinels(si)

    # the bar trace and the naive tier must agree to fp64 noise on y, a loose
    # consistency check on the magnitude source, never the judged divergence
    for bh in 0 ..< bhMax:
      for t in 0 ..< T:
        for r in 0 ..< Dv:
          let i = (bh * T + t) * Dv + r
          doAssert abs(yN[i] - yPerGlobal[i]) <=
            1e-9 * (1.0 + abs(yN[i])),
            "the bar trace and the naive tier disagree on y"

    # continuity bar, the fp64 chunked-vs-walk reassociation term on top of the state bar
    var contBar = newSeq[float64](stateElems)
    for i in 0 ..< stateElems:
      contBar[i] = bars.barS[i] +
        2.0 * float64(T * Dk) * U64 * max(abs(sN[i]), abs(sPerSeq[i]))

    if record:
      if overflowG:
        # the fixture's exact finiteness check, one NaN or Inf in the y
        # output or the carried state is a failure on its own, the bars
        # would catch the same failure late
        for i in 0 ..< yElems:
          let yc = classify(yB.hostPtr[i].widenTo(dt).float64)
          doAssert yc notin {fcNan, fcInf, fcNegInf},
            &"y not finite at element {i} (classify {yc})"
        for i in 0 ..< stateElems:
          let sc = classify(stateB.hostPtr[i].float64)
          doAssert sc notin {fcNan, fcInf, fcNegInf},
            &"carried state not finite at element {i} (classify {sc})"
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

  var rng = initNaiveRng(seed)
  var case0: tuple[st: seq[float32], y: seq[uint16]]
  const cases = 4
  for caseId in 0 ..< cases:
    let si = takeInputs(dt, rng, bhMax, qkRows, T, Dv, Dk, chunkLen, betaZero,
      overflowG)
    judge(si, record = true)
    if caseId == 0: case0 = record()
  # determinism relaunch of case 0, bit-identical across launches
  block determinism:
    var rng0 = initNaiveRng(seed)
    let si = takeInputs(dt, rng0, bhMax, qkRows, T, Dv, Dk, chunkLen, betaZero,
      overflowG)
    judge(si, record = false)
    let again = record()
    for i in 0 ..< stateElems:
      doAssert again.st[i] == case0.st[i], "state differs run to run"
    for i in 0 ..< yElems:
      doAssert again.y[i] == case0.y[i], "y differs run to run"

  let betaTag = (if betaZero: " beta=0" else: "") &
    (if overflowG: " overflow-g" else: "")
  echo &"[{label} {ulpDatatypeName(ulpG)} T={T} C={chunkLen} Dk={Dk}{betaTag}] " &
    &"cases={cases} launches={launches} | " &
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
  engine.ingest(KdaPrefillMsl)

  proc secF16T8 =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 8, 32, 32, false, 0xC04D04B1'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16T64 =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 64, 32, 32, false, 0xC04D04B2'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16Gqa =
    let t0 = epochTime()
    runCase(engine, kFloat16, 4, 2, 2, 2, 256, 32, 32, false, 0xC04D04B3'u64,
      "gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16Tail =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 100, 32, 32, false, 0xC04D04B4'u64,
      "tail Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16C64 =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 64, 64, 32, false, 0xC04D04B5'u64,
      "chunk64 Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16Dk64 =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 64, 32, 64, false, 0xC04D04B6'u64,
      "dk64 Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16Beta0 =
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 64, 32, 32, true, 0xC04D04B7'u64,
      "beta0 Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16T8 =
    let t0 = epochTime()
    runCase(engine, kBfloat16, 1, 1, 1, 1, 8, 32, 32, false, 0xC04D04B8'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Tail =
    let t0 = epochTime()
    runCase(engine, kBfloat16, 1, 1, 1, 1, 100, 32, 32, false, 0xC04D04B9'u64,
      "tail Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Gqa64 =
    let t0 = epochTime()
    runCase(engine, kBfloat16, 4, 2, 2, 2, 256, 64, 32, false, 0xC04D04BA'u64,
      "gqa64 Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  secF16T8()
  secF16T64()
  secF16Gqa()
  secF16Tail()
  secF16C64()
  secF16Dk64()
  secF16Beta0()
  secBf16T8()
  secBf16Tail()
  secBf16Gqa64()

  proc secF16Overflow =
    # the decay-overflow fixture, |cumulogdecay| crosses the exp2 overflow bound 88.7
    # inside the first 64-token chunk (g ≈ −3 per token per channel)
    let t0 = epochTime()
    runCase(engine, kFloat16, 1, 1, 1, 1, 64, 64, 32, false, 0xC04D04BB'u64,
      "overflow Hk=1/Hv=1/B=1", overflowG = true)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Overflow =
    let t0 = epochTime()
    runCase(engine, kBfloat16, 1, 1, 1, 1, 64, 64, 32, false, 0xC04D04BC'u64,
      "overflow Hk=1/Hv=1/B=1", overflowG = true)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  secF16Overflow()
  secBf16Overflow()
  echo &"CERAMIC KDA PREFILL VERDICT: cases={suiteCases} launches={suiteLaunches} " &
    &"state worst |ΔS| {suiteWorstState:.3e}, continuity worst usage {suiteWorstCont:.3f}, " &
    &"y worst {suiteWorstYUlp:.2f} ulp, worst bar usage {suiteWorstUse:.3f}, " &
    &"y bit-exact {suiteYExact}/{suiteYTotal}"

main()
