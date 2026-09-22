# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_gdn_decode.nim
##
## Ceramic GDN decode step suite:
## - `src/kernels/ceramic/sequence_mixers/state_space/gdn/gdn_decode_single.nim` compared per element against the naive `gdnDecodeStep`
## - the naive reference is per-sequence, B sequences take B independent naive calls
## - the kernel runs one launch over the stacked head axis (B·Hv threadgroups, one per head)
##
## One step reads state bars → naive walk → kernel launch → y and state judgment, the bounds carry into the next step
##
## Checks, all model-bar assertions:
## - the single-step closed-form band, 64 seeded random cases per (family dtype, shape)
## - the multi-step chain (10 steps, carried state) under the chain recursion band
##
## - run-to-run determinism, case 0 relaunched per combination and the whole chain relaunched
## - untouched-memory checks every launch, kernel-written buffers stay inside their extents and kernel-read buffers stay bit-identical
##
## Shapes (Dv = 16, TileR = 8, Dk = 32, grid (Dv div TileR, B·Hv), 32 lanes):
##
## | shape    | Hk | Hv | batch | hkRatio | family     | chain |
## | -------- | --- | --- | ----- | ------- | ---------- | ----- |
## | baseline | 1  | 1  | 1     | 1       | fp16, bf16 | bf16  |
## | gqa      | 2  | 4  | 2     | 2       | fp16, bf16 | fp16  |
##
## - every case starts from a non-zero random initial state
## - g spans -3 <= g < -0.1 in the single-step cases, -0.5 <= g < -0.01 in the chains,
##   the near-unitary decay stresses the chain recursion hardest
## - edge combos carry the near-zero decay (g -> 0-) and the exact-zero beta inside the same band
##
## - fp16 is the family dtype under test, bf16 the range-robust fallback
## - the GQA shape keeps both mapping terms live, in-sequence ratio term plus sequence-offset term
## - sequence 1 holds independent key heads, a dropped sequence offset cannot pass
##
## Band model, stated before measurement and judged per element, u₃₂ = 2⁻²⁴ is the fp32 unit roundoff:
##
## | symbol | value                         |
## | ------ | ----------------------------- |
## | a, b   | abs(S·exp(g)), abs(k·δ)       |
## | kvAbs  | Σ_dkc abs(S·k) over the row r |
## | δ      | β·(v − Σ_dkc S·k)             |
## | yAbs   | Σ_dkc abs(S'·q̃) over the row |
## | u_fam  | 2⁻¹¹ for fp16, 2⁻⁸ for bf16   |
##
## | bar                | bound                                                                                              |
## | ------------------ | -------------------------------------------------------------------------------------------------- |
## | state (bh, r, dkc) | 4·2⁻²⁴·a + abs(k)·(β·2·Dk·2⁻²⁴·kvAbs + 2·2⁻²⁴·β·(abs(v)+abs(kv)) + 2·2⁻²⁴·abs(δ)) + 4·2⁻²⁴·(a + b) |
## | y (bh, r)          | 2·u_fam·abs(y) + (2·Dk·2⁻²⁴ + 2·2⁻²¹)·yAbs + 2·2⁻²⁴·abs(y) + 2⁻²⁵                                  |
##
## | term             | covers                                                     |
## | ---------------- | ---------------------------------------------------------- |
## | 2·u_fam·abs(y)   | both sides round the same fp32 value once                  |
## | (2·Dk·2⁻²⁴)·yAbs | the two dot orders                                         |
## | 2·2⁻²¹·yAbs      | the rsqrt-vs-divide q̃ difference                          |
## | 2⁻²⁵             | the fp16 subnormal grid floor, also covering the bf16 grid |
##
## - Chain model, the two sides run the same fp32 arithmetic on state values that differ
##   - the accumulated error ΔS propagates linearly, all terms nonnegative
##   - the triangle inequality bounds each op, giving this recursion:
##
## | chain bound      | recursion                                                                           |
## | ---------------- | ----------------------------------------------------------------------------------- |
## | ΔS_{t+1}[r, dkc] | exp(g)·ΔS_t[r, dkc] + barStep[r, dkc] + abs(k_dkc)·β·Σ_c abs(k_c)·exp(g)·ΔS_t[r, c] |
## | Δy_t[r]          | Σ_c abs(q̃_c)·ΔS_{t+1}[r, c] + barY_t[r]                                            |
##
## - barStep and barY are the single-step bars above, evaluated on the naive-side trajectory of each step
## - the measured divergence justifies the model, never sets the bar
## - adjudicated on Apple M4 Max with fresh seeded xorshift64 inputs

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/sequence_mixers/state_space/gdn/gdn_decode_single
import ../naive/naive_rng
import ../naive/naive_tensors
import ../naive/naive_gdn
import ceramic_pagebuf
import ../../src/kernels/ceramic/launch_contract
import ceramic_fam

# ─── Device entries, one per (family dtype, Dk) binding ──────────────

const GdnDecodeMsl = metal:
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
  RelQScale = 2.0 * 4.76837158203125e-7  # 2·2⁻²¹, rsqrt vs divide, relative
  UBf16 = 3.90625e-3                 # 2⁻⁸, the bf16 unit roundoff
  UF16 = 4.8828125e-4                # 2⁻¹¹, the fp16 unit roundoff
  FloorSub = 2.9802322387695312e-8   # 2⁻²⁵, half the constant fp16 subnormal ulp,
                                     # the rounding floor once |y| falls subnormal

type StepInputs = object
  ## One decode step's seeded inputs, family-dtype bits shared by the kernel and the naive sides through their exact fp32 widenings:
  ##
  ##   | field        | shape                 |
  ##   | ------------ | --------------------- |
  ##   | qBits, kBits | (B·Hk, Dk)            |
  ##   | vBits        | (B·Hv, Dv)            |
  ##   | betaBits     | (B·Hv,)               |
  ##   | gVals        | (B·Hv,) f32 log-decay |
  qBits: seq[uint16]
  kBits: seq[uint16]
  vBits: seq[uint16]
  betaBits: seq[uint16]
  gVals: seq[float32]

type StepSnap = object
  ## Bit snapshots of one step's kernel-written buffers.
  state: seq[float32]
  y: seq[uint16]

proc runCombo(engine: HwEngine, fam: Family, Hv, Hk, hkRatio, B, dk, steps, cases: int, seed: uint64, label: string, gLoOverride = 0.0'f32, gHiOverride = 0.0'f32, betaZero = false) =
  ## One (family dtype, shape) combination over `cases` independent seeded
  ## runs of `steps` decode steps each, judged per element against the naive
  ## reference under the band model, case 0 relaunched bit-identical.
  const Dv = 16
  const TileR = 8
  let bhMax = B * Hv
  let qkRows = B * Hk
  let stateElems = bhMax * Dv * dk
  let kernelName = if fam == famF16: "cer_gdn_step_fp16_dk32" else: "cer_gdn_step_bf16_dk32"
  # the span overrides exist for the edge combos, gLo 0.0 is the sentinel
  # meaning derive the span from the step count (no edge case wants gLo = 0)
  let gLo = if gLoOverride != 0.0'f32: gLoOverride
            else: (if steps == 1: -3.0'f32 else: -0.5'f32)
  let gHi = if gLoOverride != 0.0'f32: gHiOverride
            else: (if steps == 1: -0.1'f32 else: -0.01'f32)
  let uFam = if fam == famBf16: UBf16 else: UF16

  var stateB = allocPageBuf[float32](stateElems)
  var yB = allocPageBuf[uint16](bhMax * Dv)
  var kB = allocPageBuf[uint16](qkRows * dk)
  var qB = allocPageBuf[uint16](qkRows * dk)
  var vB = allocPageBuf[uint16](bhMax * Dv)
  var gB = allocPageBuf[float32](bhMax)
  var betaB = allocPageBuf[uint16](bhMax)
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
  assertHeadMapping(Hv, Hk, hkRatio)
  assertLanes32(LaneWidth)
  assertNocopyBinding(statePA)
  assertNocopyBinding(yPA)
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
      kB.hostPtr[i] = si.kBits[i]
      qB.hostPtr[i] = si.qBits[i]
    for i in 0 ..< bhMax * Dv:
      vB.hostPtr[i] = si.vBits[i]
    for h in 0 ..< bhMax:
      gB.hostPtr[h] = si.gVals[h]
      betaB.hostPtr[h] = si.betaBits[h]

  proc launch(si: StepInputs) =
    assertDecayFinite(gPA, bhMax)
    engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (LaneWidth, 1, 1)) >>
      (kernelName, statePA,
        (yPA, kPA, qPA, vPA, betaPA, gPA,
          int32(Hv), int32(Hk), int32(hkRatio)))
    inc launches

  proc sentinels(si: StepInputs) =
    assertTailZero(yB, bhMax * Dv)
    assertTailZero(stateB, stateElems)
    assertReadUnchanged(kB, si.kBits)
    assertReadUnchanged(qB, si.qBits)
    assertReadUnchanged(vB, si.vBits)
    assertReadUnchanged(betaB, si.betaBits)
    for i in 0 ..< bhMax:
      doAssert gB.hostPtr[i] == si.gVals[i], "kernel-read buffer modified"

  # The step walk computes bars from the naive-side trajectory, then the naive walk, the kernel launch and the per-element judgment.
  # `judge` false marks the determinism relaunch, bit-compared against the first pass instead.
  proc runSteps(state0: seq[float32], chain: seq[StepInputs], snaps: var seq[StepSnap], judge: bool) =
    setState(state0)
    var stateN = state0                       # naive-side fp32 state, flat
    var dS = newSeq[float64](stateElems)      # running accumulated-error bound
    for t in 0 ..< chain.len:
      let si = chain[t]
      var qF = newSeq[float32](qkRows * dk)
      var kF = newSeq[float32](qkRows * dk)
      var vF = newSeq[float32](bhMax * Dv)
      var betaF = newSeq[float32](bhMax)
      for i in 0 ..< qkRows * dk:
        qF[i] = famWiden(fam, si.qBits[i])
        kF[i] = famWiden(fam, si.kBits[i])
      for i in 0 ..< bhMax * Dv:
        vF[i] = famWiden(fam, si.vBits[i])
      for h in 0 ..< bhMax:
        betaF[h] = famWiden(fam, si.betaBits[h])

      # state bars from the naive pre-step state
      var barS = newSeq[float64](stateElems)
      for bh in 0 ..< bhMax:
        let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
        let gamma = exp(si.gVals[bh].float64)
        for r in 0 ..< Dv:
          var kvN = 0.0'f64
          var kvAbs = 0.0'f64
          var kvProp = 0.0'f64
          for c in 0 ..< dk:
            let idx = (bh * Dv + r) * dk + c
            let term = gamma * stateN[idx].float64 * kF[hk * dk + c].float64
            kvN += term
            kvAbs += abs(term)
            kvProp += abs(kF[hk * dk + c].float64) * gamma * dS[idx]
          let d = betaF[bh].float64 * (vF[bh * Dv + r].float64 - kvN)
          let dDelta = betaF[bh].float64 * (2.0 * dk.float64 * U32 * kvAbs) +
            2.0 * U32 * betaF[bh].float64 *
              (abs(vF[bh * Dv + r].float64) + abs(kvN)) + 2.0 * U32 * abs(d)
          # the chain recursion's carried-error term, |k_dkc|·β·Σ_c abs(k_c)·exp(g)·ΔS_t[r, c]
          let deltaErr = betaF[bh].float64 * kvProp
          for c in 0 ..< dk:
            let idx = (bh * Dv + r) * dk + c
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
        for i in 0 ..< Hk * dk:
          qMat.data[i] = qF[(b * Hk) * dk + i]
          kMat.data[i] = kF[(b * Hk) * dk + i]
        var vMat = NaiveMat[float32](rows: Hv, cols: Dv)
        vMat.data = newSeq[float32](Hv * Dv)
        var yMat = NaiveMat[float32](rows: Hv, cols: Dv)
        yMat.data = newSeq[float32](Hv * Dv)
        var betaSeq = newSeq[float32](Hv)
        var gSeq = newSeq[float32](Hv)
        for p in 0 ..< Hv:
          for c in 0 ..< Dv:
            vMat.data[p * Dv + c] = vF[(b * Hv + p) * Dv + c]
          betaSeq[p] = betaF[b * Hv + p]
          gSeq[p] = si.gVals[b * Hv + p]
        gdnDecodeStep(stateSeq, yMat, qMat, kMat, vMat, betaSeq, gSeq, Hv, Hk, hkRatio)
        for i in 0 ..< Hv * Dv * dk:
          stateN[base + i] = stateSeq.data[i]
        for i in 0 ..< Hv * Dv:
          yN[(b * Hv) * Dv + i] = yMat.data[i]

      copyStepInputs(si)
      launch(si)
      sentinels(si)

      # y bars from the naive post-step state, then the y judgment
      for bh in 0 ..< bhMax:
        let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
        for r in 0 ..< Dv:
          var yAbs = 0.0'f64
          var yProp = 0.0'f64
          for c in 0 ..< dk:
            let qs = qF[hk * dk + c].float64 / sqrt(dk.float32).float64
            yAbs += abs(stateN[(bh * Dv + r) * dk + c].float64 * qs)
            yProp += abs(qs) * barS[(bh * Dv + r) * dk + c]
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
    ## Seeded inputs for one chain, family-dtype bits for every step.
    for step in 0 ..< steps:
      var qBits = newSeq[uint16](qkRows * dk)
      var kBits = newSeq[uint16](qkRows * dk)
      var vBits = newSeq[uint16](bhMax * Dv)
      var betaBits = newSeq[uint16](bhMax)
      var gVals = newSeq[float32](bhMax)
      for i in 0 ..< qkRows * dk:
        qBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
        kBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
      for i in 0 ..< bhMax * Dv:
        vBits[i] = toFamBits(fam, rng.nextF32(-1.0'f32, 1.0'f32))
      for h in 0 ..< bhMax:
        betaBits[h] = (if betaZero: toFamBits(fam, 0.0'f32)
                       else: toFamBits(fam, rng.nextF32(0.2'f32, 0.8'f32)))
        gVals[h] = rng.nextF32(gLo, gHi)
      result.add(StepInputs(qBits: qBits, kBits: kBits, vBits: vBits,
        betaBits: betaBits, gVals: gVals))

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

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(GdnDecodeMsl)

  proc secF16Baseline =
    let t0 = epochTime()
    runCombo(engine, famF16, 1, 1, 1, 1, 32, 1, 64, 0xC04D0401'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16Gqa =
    let t0 = epochTime()
    runCombo(engine, famF16, 4, 2, 2, 2, 32, 1, 64, 0xC04D0402'u64,
      "gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Baseline =
    let t0 = epochTime()
    runCombo(engine, famBf16, 1, 1, 1, 1, 32, 1, 64, 0xC04D0403'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Gqa =
    let t0 = epochTime()
    runCombo(engine, famBf16, 4, 2, 2, 2, 32, 1, 64, 0xC04D0404'u64,
      "gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secChainF16Gqa =
    let t0 = epochTime()
    runCombo(engine, famF16, 4, 2, 2, 2, 32, 10, 2, 0xC04D0405'u64,
      "chain gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secChainBf16Baseline =
    let t0 = epochTime()
    runCombo(engine, famBf16, 1, 1, 1, 1, 32, 10, 2, 0xC04D0406'u64,
      "chain baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secEdgeNearZeroG =
    let t0 = epochTime()
    runCombo(engine, famF16, 1, 1, 1, 1, 32, 1, 64, 0xC04D04A7'u64,
      "edge g->0- Hk=1/Hv=1/B=1", -0.001'f32, 0.0'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secEdgeBetaZero =
    let t0 = epochTime()
    runCombo(engine, famBf16, 1, 1, 1, 1, 32, 1, 64, 0xC04D04A8'u64,
      "edge beta=0 Hk=1/Hv=1/B=1", betaZero = true)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  secEdgeNearZeroG()
  secEdgeBetaZero()
  secF16Baseline()
  secF16Gqa()
  secBf16Baseline()
  secBf16Gqa()
  secChainF16Gqa()
  secChainBf16Baseline()
  echo "CERAMIC GDN DECODE VERDICT: all combinations inside the stated per-element bars"

main()
