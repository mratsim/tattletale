# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_kda_decode.nim
##
## Ceramic KDA decode step suite, `src/kernels/ceramic/sequence_mixers/state_space/kda/kda_decode_single.nim`
## compared per element against the naive `kdaDecodeStep`:
## - the naive reference is per-sequence, B sequences take B independent naive calls
## - the kernel runs one launch over the stacked head axis (B·Hv threadgroups, one per head)
##
## Checks, all model-bar assertions, one step: bars → walk → launch → judgment:
##
## | check            | form                                                               |
## | ---------------- | ------------------------------------------------------------------ |
## | single-step band | closed-form, 64 seeded random cases per (family dtype, shape)      |
## | chain            | 10 steps, carried state, under the chain recursion band            |
## | edge combos      | near-zero decay (g → 0⁻) and exact-zero beta, inside the same band |
## | cross-check      | uniform g, ceramic KDA vs ceramic GDN, state asserted bit-exact    |
##
## - run-to-run determinism, case 0 relaunched per combination and the whole chain relaunched
## - untouched-memory checks every launch, kernel writes stay inside their extents, kernel reads stay bit-identical
##
## Shapes (Dv = 16, TileR = 8, Dk = 32, grid (Dv div TileR, B·Hv), 32 lanes):
##
## | shape    | Hk | Hv | batch | hkRatio | family     | chain |
## | -------- | --- | --- | ----- | ------- | ---------- | ----- |
## | baseline | 1  | 1  | 1     | 1       | fp16, bf16 | bf16  |
## | gqa      | 2  | 4  | 2     | 2       | fp16, bf16 | fp16  |
##
## | regime   | note                                                                            |
## | -------- | ------------------------------------------------------------------------------- |
## | initial  | every case starts from a non-zero random initial state                          |
## | g span   | -3 <= g < -0.1 single-step, -0.5 <= g < -0.01 chains, -0.001 <= g < 0 near-zero |
## | stress   | the near-unitary decay stresses the chain recursion hardest                     |
## | dtype    | q/k/g/beta are f32 per the recorded KDA contract, f32 through the naive side    |
## | dtype    | v/y carry the family dtype, fp16 the primary, bf16 the range-robust fallback    |
## | gqa      | both mapping terms live, in-sequence ratio plus sequence offset                 |
## | sabotage | sequence 1 holds independent key heads, a dropped sequence offset cannot pass   |
##
## Band model, stated before measurement, u₃₂ = 2⁻²⁴ the fp32 unit roundoff, γ_c = exp(g_c):
##
## | symbol             | value                                                                                                                                            |
## | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
## | a, b               | abs(γ_c·S), abs(k_c·δ)                                                                                                                           |
## | kvAbs              | Σ_dkc abs(γ_dkc·S·k_dkc) over the row r                                                                                                          |
## | δ                  | β·(v − Σ_dkc γ_dkc·S·k_dkc)                                                                                                                      |
## | yAbs               | Σ_dkc abs(S'·q̃_dkc) over the row                                                                                                                |
## | u_fam              | 2⁻¹¹ for fp16, 2⁻⁸ for bf16                                                                                                                      |
## | bar                | bound                                                                                                                                            |
## | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
## | state (bh, r, dkc) | γ_dkc·ΔS_t + 4·2⁻²⁴·a + abs(k)·(β·2·Dk·2⁻²⁴·kvAbs + 2·2⁻²⁴·β·(abs(v)+abs(kv)) + 2·2⁻²⁴·abs(δ)) + 4·2⁻²⁴·(a + b) + abs(k)·β·Σ_c abs(k_c)·γ_c·ΔS_t |
## | y (bh, r)          | 2·u_fam·abs(y) + (2·Dk·2⁻²⁴ + 4·2⁻²⁴)·yAbs + 2·2⁻²⁴·abs(y) + 2⁻²⁵                                                                                |
## | term               | covers                                                                                                                                           |
## | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
## | 4·2⁻²⁴·a           | the exp (naive) vs exp2(g·log2e) (kernel) decay form                                                                                             |
## | 2·u_fam·abs(y)     | both sides round the same fp32 value once                                                                                                        |
## | (2·Dk·2⁻²⁴)·yAbs   | the two dot orders                                                                                                                               |
## | 4·2⁻²⁴·yAbs        | the qScale difference, the naive's f32 √Dk vs the host's f64 √Dk cast to f32, at most one extra ulp each side                                    |
## | 2⁻²⁵               | the fp16 subnormal grid floor, also covering the bf16 grid                                                                                       |
##
## Chain model, the two sides run the same fp32 arithmetic on state values that differ:
##
## - the accumulated error ΔS propagates linearly, all terms nonnegative
## - the triangle inequality bounds each op, giving this recursion:
##
## | chain bound      | recursion                                                                       |
## | ---------------- | ------------------------------------------------------------------------------- |
## | ΔS_{t+1}[r, dkc] | γ_dkc·ΔS_t[r, dkc] + barStep[r, dkc] + abs(k_dkc)·β·Σ_c abs(k_c)·γ_c·ΔS_t[r, c] |
## | Δy_t[r]          | Σ_c abs(q̃_c)·ΔS_{t+1}[r, c] + barY_t[r]                                        |
##
## - barStep and barY are the single-step bars on the naive-side trajectory of each step
## - the measured divergence justifies the model, never sets the bar
## - the β = 0 edge combo stays within the band, the delta terms are exact zero, the decay drifts by at most 4·2⁻²⁴ relative
##
## Uniform-g cross-check band, ceramic KDA vs ceramic GDN fed family-exact operands, state bit-exact:
##
## - family-exact, q/k/beta the exact f32 widenings of the GDN side's family bits, v the same bits, g one uniform scalar
## - y differs only through the q̃ spelling, judged against the cross-check bar
##
## | bar         | bound                                                                                    |
## | ----------- | ---------------------------------------------------------------------------------------- |
## | state       | bit-exact, identical f32 operands through identical fp32 ops                             |
## | y (bh, r)   | 2·u_fam·abs(y) + (2·2⁻²¹ + 2·2⁻²⁴)·yAbs + 2⁻²⁵                                           |
## | term        | covers                                                                                   |
## | ----------- | ---------------------------------------------------------------------------------------- |
## | 2·2⁻²¹·yAbs | the q̃ spelling, GDN's rsqrt-multiply vs KDA's divide by the runtime qScale              |
## | 2·2⁻²⁴·yAbs | slack for compiler-level reassociation between the two separately compiled kernel bodies |
## | 2⁻²⁵        | the family-dtype subnormal grid floor                                                    |
##
## - adjudicated on Apple M4 Max with fresh seeded xorshift64 inputs

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/sequence_mixers/state_space/kda/kda_decode_single
import ../../src/kernels/ceramic/sequence_mixers/state_space/gdn/gdn_decode_single
import ../naive/naive_rng
import ../naive/naive_tensors
import ../naive/naive_kda
import ../naive/naive_gdn
import ceramic_pagebuf
import ceramic_fam

# ─── Device entries, one per (family dtype, Dk) binding ──────────────

const CeramicDecodeMsl = metal:
  # One `metal:` block for all four entries, the Metal engine's ingest replaces the previous
  # library artifact, and helpers dedupe within a single compilation unit.
  proc cer_kda_step_fp16_dk32(
      state: ptr UncheckedArray[float32],
      y: ptr UncheckedArray[float16],
      k, q: ptr UncheckedArray[float32],
      v: ptr UncheckedArray[float16],
      g, beta: ptr UncheckedArray[float32],
      qScale: float32,
      Hv, Hk, hkRatio: int32) {.global.} =
    kdaDecodeStepTileF16(state, y, k, q, v, g, beta, qScale, Hv, Hk, hkRatio, 32, 16, 8)

  proc cer_kda_step_bf16_dk32(
      state: ptr UncheckedArray[float32],
      y: ptr UncheckedArray[bfloat16],
      k, q: ptr UncheckedArray[float32],
      v: ptr UncheckedArray[bfloat16],
      g, beta: ptr UncheckedArray[float32],
      qScale: float32,
      Hv, Hk, hkRatio: int32) {.global.} =
    kdaDecodeStepTileBf16(state, y, k, q, v, g, beta, qScale, Hv, Hk, hkRatio, 32, 16, 8)

  proc cer_gdn_step_fp16_dk32(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[float16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio: int32) {.global.} =
    gdnDecodeStepTileF16(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, 32, 16, 8)

  proc cer_gdn_step_bf16_dk32(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[bfloat16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio: int32) {.global.} =
    gdnDecodeStepTileBf16(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, 32, 16, 8)

# ─── Host tolerance-model constants ───────────────────────────────────

const
  RelDecay = 4.0 * U32               # exp vs exp2(g·log2e) relative bound
  RelQScale = 4.0 * U32              # naive's f32 sqrt vs the host's f64 sqrt cast to f32
  RelQScaleX = 2.0 * 4.76837158203125e-7  # 2·2⁻²¹, rsqrt-multiply vs divide, cross-check
  UBf16 = 3.90625e-3                 # 2⁻⁸, the bf16 unit roundoff
  UF16 = 4.8828125e-4                # 2⁻¹¹, the fp16 unit roundoff
  FloorSub = 2.9802322387695312e-8   # 2⁻²⁵, half the constant fp16 subnormal ulp,
                                     # the rounding floor once |y| falls subnormal

type StepInputs = object
  ## One decode step's seeded inputs, q/k/g/beta f32 shared verbatim by the kernel
  ## and the naive sides, v as family-dtype bits the naive side sees through its exact
  ## widening:
  ##
  ##   | field        | shape          |
  ##   | ------------ | -------------- |
  ##   | qVals, kVals | (B·Hk, Dk) f32 |
  ##   | gVals        | (B·Hk, Dk) f32 |
  ##   | vBits        | (B·Hv, Dv)     |
  ##   | betaVals     | (B·Hv,) f32    |
  qVals: seq[float32]
  kVals: seq[float32]
  gVals: seq[float32]
  vBits: seq[uint16]
  betaVals: seq[float32]

type StepSnap = object
  ## Bit snapshots of one step's kernel-written buffers.
  state: seq[float32]
  y: seq[uint16]

proc runCombo(engine: HwEngine, fam: Family, Hv, Hk, hkRatio, B, dk, steps, cases: int, seed: uint64, label: string, gLo, gHi: float32, betaZero = false) =
  ## One (family dtype, shape, edge) combination over `cases` independent seeded runs
  ## of `steps` decode steps each, judged per element against the naive reference
  ## under the band model, case 0 relaunched bit-identical.
  const Dv = 16
  const TileR = 8
  let bhMax = B * Hv
  let qkRows = B * Hk
  let stateElems = bhMax * Dv * dk
  let kernelName = if fam == famF16: "cer_kda_step_fp16_dk32" else: "cer_kda_step_bf16_dk32"
  let uFam = if fam == famBf16: UBf16 else: UF16

  var stateB = allocPageBuf[float32](stateElems)
  var yB = allocPageBuf[uint16](bhMax * Dv)
  var kB = allocPageBuf[float32](qkRows * dk)
  var qB = allocPageBuf[float32](qkRows * dk)
  var vB = allocPageBuf[uint16](bhMax * Dv)
  var gB = allocPageBuf[float32](qkRows * dk)
  var betaB = allocPageBuf[float32](bhMax)
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
  var worstYUse = 0.0'f64
  var worstYUlp = 0.0'f64
  var yExact = 0
  var yTotal = 0
  var launches = 0

  proc setState(state0: seq[float32]) =
    for i in 0 ..< stateElems:
      stateB.hostPtr[i] = state0[i]
    for i in 0 ..< bhMax * Dv:
      yB.hostPtr[i] = 0

  proc copyStepInputs(si: StepInputs) =
    for i in 0 ..< qkRows * dk:
      kB.hostPtr[i] = si.kVals[i]
      qB.hostPtr[i] = si.qVals[i]
      gB.hostPtr[i] = si.gVals[i]
    for i in 0 ..< bhMax * Dv:
      vB.hostPtr[i] = si.vBits[i]
    for h in 0 ..< bhMax:
      betaB.hostPtr[h] = si.betaVals[h]

  proc launch() =
    engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
      (kernelName, statePA,
        (yPA, kPA, qPA, vPA, gPA, betaPA,
          float32(sqrt(float64(dk))),
          int32(Hv), int32(Hk), int32(hkRatio)))
    inc launches

  proc sentinels(si: StepInputs) =
    assertTailZero(yB, bhMax * Dv)
    assertTailZero(stateB, stateElems)
    assertReadUnchanged(kB, si.kVals)
    assertReadUnchanged(qB, si.qVals)
    assertReadUnchanged(gB, si.gVals)
    assertReadUnchanged(betaB, si.betaVals)
    assertReadUnchanged(vB, si.vBits)

  # The step walk computes bars from the naive-side trajectory, then the naive walk,
  # the kernel launch and the per-element judgment.
  # `judge` false marks the determinism relaunch, bit-compared against the first pass instead.
  proc runSteps(state0: seq[float32], chain: seq[StepInputs], snaps: var seq[StepSnap], judge: bool) =
    setState(state0)
    var stateN = state0                       # naive-side fp32 state, flat
    var dS = newSeq[float64](stateElems)      # running accumulated-error bound
    for t in 0 ..< chain.len:
      let si = chain[t]
      var qF = newSeq[float32](qkRows * dk)
      var kF = newSeq[float32](qkRows * dk)
      var gF = newSeq[float32](qkRows * dk)
      var vF = newSeq[float32](bhMax * Dv)
      var betaF = newSeq[float32](bhMax)
      for i in 0 ..< qkRows * dk:
        qF[i] = si.qVals[i]
        kF[i] = si.kVals[i]
        gF[i] = si.gVals[i]
      for i in 0 ..< bhMax * Dv:
        vF[i] = famWiden(fam, si.vBits[i])
      for h in 0 ..< bhMax:
        betaF[h] = si.betaVals[h]

      # state bars from the naive pre-step state, per key channel gamma
      var barS = newSeq[float64](stateElems)
      for bh in 0 ..< bhMax:
        let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
        for r in 0 ..< Dv:
          var kvN = 0.0'f64
          var kvAbs = 0.0'f64
          var kvProp = 0.0'f64
          for c in 0 ..< dk:
            let idx = (bh * Dv + r) * dk + c
            let gamma = exp(gF[hk * dk + c].float64)
            let term = gamma * stateN[idx].float64 * kF[hk * dk + c].float64
            kvN += term
            kvAbs += abs(term)
            kvProp += abs(kF[hk * dk + c].float64) * gamma * dS[idx]
          let vAbs = abs(vF[bh * Dv + r].float64)
          let d = betaF[bh].float64 * (vF[bh * Dv + r].float64 - kvN)
          let dDelta = betaF[bh].float64 * (2.0 * dk.float64 * U32 * kvAbs) +
            2.0 * U32 * betaF[bh].float64 * (vAbs + abs(kvN)) + 2.0 * U32 * abs(d)
          # the chain recursion's carried-error term, |k_dkc|·β·Σ_c abs(k_c)·γ_c·ΔS_t[r, c]
          let deltaErr = betaF[bh].float64 * kvProp
          for c in 0 ..< dk:
            let idx = (bh * Dv + r) * dk + c
            let gamma = exp(gF[hk * dk + c].float64)
            let a = abs(gamma * stateN[idx].float64)
            let bTerm = abs(kF[hk * dk + c].float64 * d)
            barS[idx] = gamma * dS[idx] +
              (RelDecay * a + abs(kF[hk * dk + c].float64) * dDelta +
                4.0 * U32 * (a + bTerm)) + abs(kF[hk * dk + c].float64) * deltaErr

      # the naive walk, per sequence (the naive reference is per-sequence)
      var yN = newSeq[float32](bhMax * Dv)
      for b in 0 ..< B:
        var stateSeq = NaiveCube[float32](planes: Hv, rows: Dv, cols: dk)
        stateSeq.data = newSeq[float32](Hv * Dv * dk)
        let base = b * Hv * Dv * dk
        for i in 0 ..< Hv * Dv * dk:
          stateSeq.data[i] = stateN[base + i]
        var qMat = NaiveMat[float32](rows: Hk, cols: dk)
        qMat.data = newSeq[float32](Hk * dk)
        var kMat = NaiveMat[float32](rows: Hk, cols: dk)
        kMat.data = newSeq[float32](Hk * dk)
        var gMat = NaiveMat[float32](rows: Hk, cols: dk)
        gMat.data = newSeq[float32](Hk * dk)
        for i in 0 ..< Hk * dk:
          qMat.data[i] = qF[(b * Hk) * dk + i]
          kMat.data[i] = kF[(b * Hk) * dk + i]
          gMat.data[i] = gF[(b * Hk) * dk + i]
        var vMat = NaiveMat[float32](rows: Hv, cols: Dv)
        vMat.data = newSeq[float32](Hv * Dv)
        var yMat = NaiveMat[float32](rows: Hv, cols: Dv)
        yMat.data = newSeq[float32](Hv * Dv)
        var betaSeq = newSeq[float32](Hv)
        for p in 0 ..< Hv:
          for c in 0 ..< Dv:
            vMat.data[p * Dv + c] = vF[(b * Hv + p) * Dv + c]
          betaSeq[p] = betaF[b * Hv + p]
        kdaDecodeStep(stateSeq, yMat, qMat, kMat, gMat, vMat, betaSeq,
          int32(Hv), int32(Hk), int32(hkRatio))
        for i in 0 ..< Hv * Dv * dk:
          stateN[base + i] = stateSeq.data[i]
        for i in 0 ..< Hv * Dv:
          yN[(b * Hv) * Dv + i] = yMat.data[i]

      copyStepInputs(si)
      launch()
      sentinels(si)

      # y bars from the naive post-step state, then the y judgment
      let qScaleN = sqrt(float32(dk))
      for bh in 0 ..< bhMax:
        let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
        for r in 0 ..< Dv:
          var yAbs = 0.0'f64
          var yProp = 0.0'f64
          for c in 0 ..< dk:
            let idx = (bh * Dv + r) * dk + c
            let qs = qF[hk * dk + c].float64 / qScaleN.float64
            yAbs += abs(stateN[idx].float64 * qs)
            yProp += abs(qs) * barS[idx]
          let yWant = yN[bh * Dv + r].float64
          let barY = 2.0 * uFam * abs(yWant) +
            (2.0 * dk.float64 * U32 + RelQScale) * yAbs +
            2.0 * U32 * abs(yWant) + FloorSub + yProp
          let yGot = famWiden(fam, yB.hostPtr[bh * Dv + r]).float64
          let yDiff = abs(yGot - yWant)
          if judge:
            doAssert yDiff <= barY,
              &"y outside the bar at (bh {bh}, r {r}, step {t}): " &
              &"{yDiff:.3e} > {barY:.3e}"
            let uAt = famUlp(fam, yWant)
            if uAt > 0.0 and yDiff > 0.0:
              worstYUlp = max(worstYUlp, yDiff / uAt)
            if yDiff == 0.0:
              inc yExact
            inc yTotal
            if barY > 0.0: worstYUse = max(worstYUse, yDiff / barY)

      # state judgment against the step bounds, the bounds carry forward
      for i in 0 ..< stateElems:
        let got = stateB.hostPtr[i].float64
        let want = stateN[i].float64
        let diff = abs(got - want)
        if judge:
          doAssert diff <= barS[i],
            &"state outside the bar at element {i}, step {t}: " &
            &"{diff:.3e} > {barS[i]:.3e}"
          worstState = max(worstState, diff)
          if barS[i] > 0.0: worstStateUse = max(worstStateUse, diff / barS[i])
        dS[i] = barS[i]

      snaps.add(StepSnap(
        state: readInto(stateB.hostPtr, stateElems),
        y: readInto(yB.hostPtr, bhMax * Dv)))

  proc takeInputs(rng: var NaiveRng): seq[StepInputs] =
    ## Seeded inputs for one chain, q/k/g/beta f32 for every step.
    for step in 0 ..< steps:
      var qVals = newSeq[float32](qkRows * dk)
      var kVals = newSeq[float32](qkRows * dk)
      var gVals = newSeq[float32](qkRows * dk)
      var vBits = newSeq[uint16](bhMax * Dv)
      var betaVals = newSeq[float32](bhMax)
      for i in 0 ..< qkRows * dk:
        qVals[i] = rng.nextF32(-1.0'f32, 1.0'f32)
        kVals[i] = rng.nextF32(-1.0'f32, 1.0'f32)
        gVals[i] = rng.nextF32(gLo, gHi)
      for i in 0 ..< bhMax * Dv:
        vBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
      for h in 0 ..< bhMax:
        if betaZero:
          betaVals[h] = 0.0'f32
        else:
          betaVals[h] = rng.nextF32(0.2'f32, 0.8'f32)
      result.add(StepInputs(qVals: qVals, kVals: kVals, gVals: gVals,
        vBits: vBits, betaVals: betaVals))

  var case0Snaps: seq[StepSnap]             # the determinism reference

  var rng = initNaiveRng(seed)
  for caseId in 0 ..< cases:
    var state0 = newSeq[float32](stateElems)
    for i in 0 ..< stateElems:
      state0[i] = rng.nextF32(-1.0'f32, 1.0'f32)
    let chain = takeInputs(rng)
    var caseSnaps: seq[StepSnap]
    runSteps(state0, chain, caseSnaps, judge = true)
    if caseId == 0:
      case0Snaps = caseSnaps

  # determinism relaunch of case 0, bit-identical across launches
  block determinism:
    var rng0 = initNaiveRng(seed)
    var state0 = newSeq[float32](stateElems)
    for i in 0 ..< stateElems:
      state0[i] = rng0.nextF32(-1.0'f32, 1.0'f32)
    let chain = takeInputs(rng0)
    var relaunchSnaps: seq[StepSnap]
    runSteps(state0, chain, relaunchSnaps, judge = false)
    for t in 0 ..< relaunchSnaps.len:
      for i in 0 ..< stateElems:
        doAssert relaunchSnaps[t].state[i] == case0Snaps[t].state[i],
          "state differs run to run"
      for i in 0 ..< bhMax * Dv:
        doAssert relaunchSnaps[t].y[i] == case0Snaps[t].y[i],
          "y differs run to run"

  echo &"[{label} {famName(fam)} Dk={dk}] steps={steps} cases={cases} " &
    &"launches={launches} | state worst |ΔS| {worstState:.3e}, worst bar usage " &
    &"{worstStateUse:.3f} | y worst {worstYUlp:.2f} {famName(fam)} ulp, " &
    &"bit-exact {yExact}/{yTotal}, worst bar usage {worstYUse:.3f}"

type CrossInputs = object
  qBits: seq[uint16]
  kBits: seq[uint16]
  vBits: seq[uint16]
  betaBits: seq[uint16]
  g0: float32
  state0: seq[float32]

proc takeCrossInputs(rng: var NaiveRng, fam: Family, qkRows, bhMax, stateElems: int): CrossInputs =
  ## Seeded cross-check inputs, the KDA side reads the exact f32 widenings.
  var qBits = newSeq[uint16](qkRows * 32)
  var kBits = newSeq[uint16](qkRows * 32)
  var vBits = newSeq[uint16](bhMax * 16)
  var betaBits = newSeq[uint16](bhMax)
  for i in 0 ..< qkRows * 32:
    qBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
    kBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
  for i in 0 ..< bhMax * 16:
    vBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
  for h in 0 ..< bhMax:
    betaBits[h] = toFamBits(fam, rng.nextF32(0.2'f32, 0.8'f32))
  let g0 = rng.nextF32(-3.0'f32, -0.1'f32)
  var state0 = newSeq[float32](stateElems)
  for i in 0 ..< stateElems:
    state0[i] = rng.nextF32(-1.0'f32, 1.0'f32)
  CrossInputs(qBits: qBits, kBits: kBits, vBits: vBits, betaBits: betaBits,
    g0: g0, state0: state0)

proc runCrossCheck(engine: HwEngine, fam: Family, Hv, Hk, hkRatio, B, dk, cases: int, seed: uint64, label: string) =
  ## Uniform-g kernel-tier cross-check, ceramic KDA vs ceramic GDN.
  ##
  ## - both kernels get family-exact operands, the GDN side's family bits widened
  ##   to f32 for q/k/beta, v the same bits, g the same uniform scalar
  ## - the state arithmetic then sees identical operands, the state is asserted
  ##   bit-exact, y differs only through the q̃ spelling against the cross-check bar
  const Dv = 16
  const TileR = 8
  let bhMax = B * Hv
  let qkRows = B * Hk
  let stateElems = bhMax * Dv * dk
  let kdaName = if fam == famF16: "cer_kda_step_fp16_dk32" else: "cer_kda_step_bf16_dk32"
  let gdnName = if fam == famF16: "cer_gdn_step_fp16_dk32" else: "cer_gdn_step_bf16_dk32"
  let uFam = if fam == famBf16: UBf16 else: UF16

  var stateB = allocPageBuf[float32](stateElems)
  var yB = allocPageBuf[uint16](bhMax * Dv)
  var kFamB = allocPageBuf[uint16](qkRows * dk)
  var qFamB = allocPageBuf[uint16](qkRows * dk)
  var kF32B = allocPageBuf[float32](qkRows * dk)
  var qF32B = allocPageBuf[float32](qkRows * dk)
  var vB = allocPageBuf[uint16](bhMax * Dv)
  var gHeadB = allocPageBuf[float32](bhMax)
  var gMatB = allocPageBuf[float32](qkRows * dk)
  var betaFamB = allocPageBuf[uint16](bhMax)
  var betaF32B = allocPageBuf[float32](bhMax)
  defer:
    freePageBuf(stateB); freePageBuf(yB); freePageBuf(kFamB); freePageBuf(qFamB)
    freePageBuf(kF32B); freePageBuf(qF32B); freePageBuf(vB); freePageBuf(gHeadB)
    freePageBuf(gMatB); freePageBuf(betaFamB); freePageBuf(betaF32B)
  var statePA = stateB.pa()
  var yPA = yB.pa()
  var kFamPA = kFamB.pa()
  var qFamPA = qFamB.pa()
  var kF32PA = kF32B.pa()
  var qF32PA = qF32B.pa()
  var vPA = vB.pa()
  var gHeadPA = gHeadB.pa()
  var gMatPA = gMatB.pa()
  var betaFamPA = betaFamB.pa()
  var betaF32PA = betaF32B.pa()

  var worstYUse = 0.0'f64
  var worstYUlp = 0.0'f64
  var yExact = 0
  var yTotal = 0
  var launches = 0

  proc loadCase(ci: CrossInputs) =
    for i in 0 ..< qkRows * dk:
      kFamB.hostPtr[i] = ci.kBits[i]
      qFamB.hostPtr[i] = ci.qBits[i]
      kF32B.hostPtr[i] = famWiden(fam, ci.kBits[i])
      qF32B.hostPtr[i] = famWiden(fam, ci.qBits[i])
      gMatB.hostPtr[i] = ci.g0
    for i in 0 ..< bhMax * Dv:
      vB.hostPtr[i] = ci.vBits[i]
    for h in 0 ..< bhMax:
      gHeadB.hostPtr[h] = ci.g0
      betaFamB.hostPtr[h] = ci.betaBits[h]
      betaF32B.hostPtr[h] = famWiden(fam, ci.betaBits[h])
    for i in 0 ..< stateElems:
      stateB.hostPtr[i] = ci.state0[i]
    for i in 0 ..< bhMax * Dv:
      yB.hostPtr[i] = 0

  proc launchKda() =
    engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
      (kdaName, statePA,
        (yPA, kF32PA, qF32PA, vPA, gMatPA, betaF32PA,
          float32(sqrt(float64(dk))),
          int32(Hv), int32(Hk), int32(hkRatio)))
    inc launches

  proc launchGdn() =
    engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
      (gdnName, statePA,
        (yPA, kFamPA, qFamPA, vPA, betaFamPA, gHeadPA,
          int32(Hv), int32(Hk), int32(hkRatio)))
    inc launches

  # One case walks KDA launch then GDN launch, sentinels per launch, state bit-exact,
  # y judged per element against the cross-check bar from the shared post-step state.
  proc runCase(ci: CrossInputs, caseId: int, judge: bool, refState: var seq[float32], refY: var seq[uint16]) =
    loadCase(ci)
    launchKda()
    assertTailZero(yB, bhMax * Dv)
    assertTailZero(stateB, stateElems)
    var kWant = newSeq[float32](qkRows * dk)
    for i in 0 ..< qkRows * dk:
      kWant[i] = famWiden(fam, ci.kBits[i])
    assertReadUnchanged(kF32B, kWant)
    let stateK = readInto(stateB.hostPtr, stateElems)
    let yK = readInto(yB.hostPtr, bhMax * Dv)

    loadCase(ci)
    launchGdn()
    assertTailZero(yB, bhMax * Dv)
    assertTailZero(stateB, stateElems)
    let stateG = readInto(stateB.hostPtr, stateElems)
    let yG = readInto(yB.hostPtr, bhMax * Dv)

    # the state is asserted bit-exact, identical f32 operands through identical fp32 ops
    for i in 0 ..< stateElems:
      doAssert stateK[i] == stateG[i],
        &"cross-check state not bit-exact at element {i}, case {caseId}: " &
        &"{stateK[i]} vs {stateG[i]}"

    if not judge:
      # determinism relaunch, bit-compared against the first pass
      for i in 0 ..< stateElems:
        doAssert stateK[i] == refState[i], "cross-check state differs run to run"
      for i in 0 ..< bhMax * Dv:
        doAssert yK[i] == refY[i], "cross-check y differs run to run"
      return

    # the y judgment against the shared post-step state
    let gdnScale = 1.0 / sqrt(dk.float64)
    for bh in 0 ..< bhMax:
      let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
      for r in 0 ..< Dv:
        var yAbs = 0.0'f64
        for c in 0 ..< dk:
          let idx = (bh * Dv + r) * dk + c
          let qs = famWiden(fam, ci.qBits[hk * dk + c]).float64 * gdnScale
          yAbs += abs(stateG[idx].float64 * qs)
        let yWant = famWiden(fam, yG[bh * Dv + r]).float64
        let barY = 2.0 * uFam * abs(yWant) +
          (RelQScaleX + 2.0 * U32) * yAbs + FloorSub
        let yGot = famWiden(fam, yK[bh * Dv + r]).float64
        let yDiff = abs(yGot - yWant)
        doAssert yDiff <= barY,
          &"cross-check y outside the bar at (bh {bh}, r {r}, case {caseId}): " &
          &"{yDiff:.3e} > {barY:.3e}"
        let uAt = famUlp(fam, yWant)
        if uAt > 0.0 and yDiff > 0.0:
          worstYUlp = max(worstYUlp, yDiff / uAt)
        if yDiff == 0.0:
          inc yExact
        inc yTotal
        if barY > 0.0: worstYUse = max(worstYUse, yDiff / barY)
    if caseId == 0:
      refState = stateK
      refY = yK

  var rng = initNaiveRng(seed)
  var refState: seq[float32]
  var refY: seq[uint16]
  for caseId in 0 ..< cases:
    let ci = takeCrossInputs(rng, fam, qkRows, bhMax, stateElems)
    runCase(ci, caseId, judge = true, refState, refY)
  # determinism relaunch of case 0, bit-identical across launches
  block determinism:
    var rng0 = initNaiveRng(seed)
    let ci0 = takeCrossInputs(rng0, fam, qkRows, bhMax, stateElems)
    runCase(ci0, 0, judge = false, refState, refY)

  echo &"[{label} {famName(fam)} Dk={dk}] cross-check cases={cases} " &
    &"launches={launches} | state bit-exact all | y worst {worstYUlp:.2f} " &
    &"{famName(fam)} ulp, bit-exact {yExact}/{yTotal}, worst bar usage {worstYUse:.3f}"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(CeramicDecodeMsl)

  proc secF16Baseline =
    let t0 = epochTime()
    runCombo(engine, famF16, 1, 1, 1, 1, 32, 1, 64, 0xC04D04A1'u64,
      "baseline Hk=1/Hv=1/B=1", -3.0'f32, -0.1'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16Gqa =
    let t0 = epochTime()
    runCombo(engine, famF16, 4, 2, 2, 2, 32, 1, 64, 0xC04D04A2'u64,
      "gqa Hk=2/Hv=4/B=2", -3.0'f32, -0.1'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Baseline =
    let t0 = epochTime()
    runCombo(engine, famBf16, 1, 1, 1, 1, 32, 1, 64, 0xC04D04A3'u64,
      "baseline Hk=1/Hv=1/B=1", -3.0'f32, -0.1'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Gqa =
    let t0 = epochTime()
    runCombo(engine, famBf16, 4, 2, 2, 2, 32, 1, 64, 0xC04D04A4'u64,
      "gqa Hk=2/Hv=4/B=2", -3.0'f32, -0.1'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secChainF16Gqa =
    let t0 = epochTime()
    runCombo(engine, famF16, 4, 2, 2, 2, 32, 10, 2, 0xC04D04A5'u64,
      "chain gqa Hk=2/Hv=4/B=2", -0.5'f32, -0.01'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secChainBf16Baseline =
    let t0 = epochTime()
    runCombo(engine, famBf16, 1, 1, 1, 1, 32, 10, 2, 0xC04D04A6'u64,
      "chain baseline Hk=1/Hv=1/B=1", -0.5'f32, -0.01'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secEdgeNearZeroG =
    let t0 = epochTime()
    runCombo(engine, famF16, 1, 1, 1, 1, 32, 1, 64, 0xC04D04A7'u64,
      "edge g->0- Hk=1/Hv=1/B=1", -0.001'f32, 0.0'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secEdgeBetaZero =
    let t0 = epochTime()
    runCombo(engine, famF16, 1, 1, 1, 1, 32, 1, 64, 0xC04D04A8'u64,
      "edge beta=0 Hk=1/Hv=1/B=1", -3.0'f32, -0.1'f32, betaZero = true)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secCrossF16 =
    let t0 = epochTime()
    runCrossCheck(engine, famF16, 4, 2, 2, 2, 32, 64, 0xC04D04A9'u64,
      "cross-check gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secCrossBf16 =
    let t0 = epochTime()
    runCrossCheck(engine, famBf16, 4, 2, 2, 2, 32, 64, 0xC04D04AA'u64,
      "cross-check gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  secF16Baseline()
  secF16Gqa()
  secBf16Baseline()
  secBf16Gqa()
  secChainF16Gqa()
  secChainBf16Baseline()
  secEdgeNearZeroG()
  secEdgeBetaZero()
  secCrossF16()
  secCrossBf16()
  echo "CERAMIC KDA DECODE VERDICT: all combinations inside the stated per-element bars, cross-check state bit-exact"

main()
