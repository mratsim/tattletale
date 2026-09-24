# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

## Run commands, from the repo root (the aggregate runner is nim test_positron_properties):
## - nim c -r -d:release --warnings:off \
##   --outdir:build/tests --nimcache:nimcache/cer workspace/positron/tests/ceramic/t_ceramic_gated_delta_net_decode_single.nim
##
## Ceramic GDN decode step suite, launch contract over seeded fixtures:
## - `src/kernels/ceramic/attn_ssm/gated_delta_net_decode_single.nim` runs one launch
##   over the stacked head axis (B·Hv threadgroups, one per head)
##
## One step writes state → kernel launch → snap of the kernel-written buffers,
## the state carrying into the next step.
##
## Value check, the property suite's step-count identity
## (decode against the same prefill, `tests/properties/t_prop_gdn.nim`).
## This suite carries the fixture, sentinel and determinism checks:
##
## - run-to-run determinism, case 0 relaunched per combination and the whole chain relaunched
## - untouched-memory checks every launch, kernel-written buffers stay inside their extents and kernel-read buffers stay bit-identical
##
## Shapes (Dv = 16, TileR = 8, Dk = 32, grid (Dv div TileR, B·Hv), 32 lanes):
##
## | shape    | Hk | Hv | batch | hkRatio | dtype      | chain |
## | -------- | --- | --- | ----- | ------- | ---------- | ----- |
## | baseline | 1  | 1  | 1     | 1       | fp16, bf16 | bf16  |
## | gqa      | 2  | 4  | 2     | 2       | fp16, bf16 | fp16  |
##
## - every case starts from a non-zero random initial state
## - g spans -3 <= g < -0.1 in the single-step cases, -0.5 <= g < -0.01 in the chains,
##   the near-unitary decay exercises the carried-state path hardest
## - edge combos carry the near-zero decay (g -> 0-) and the exact-zero beta
##
## - fp16 is the element dtype under test, bf16 the range-robust fallback
## - the GQA shape keeps both mapping terms live, in-sequence ratio term plus sequence-offset term
## - sequence 1 holds independent key heads, a dropped sequence offset changes the reads
##
## - adjudicated on Apple M4 Max with fresh seeded xorshift64 inputs

import std/[strformat, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/attn_ssm/gated_delta_net_decode_single
import ceramic_pagebuf
import ceramic_dtype

# ─── Device entries, one per (element dtype, Dk) binding ──────────────

const GdnDecodeMsl = metal:
  proc cer_gdn_step_fp16_dk32(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[float16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio: int32) {.global.} =
    gdnDecodeStepTile(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, 32, 16, 8)

  proc cer_gdn_step_bf16_dk32(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[bfloat16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio: int32) {.global.} =
    gdnDecodeStepTile(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, 32, 16, 8)

type StepInputs = object
  ## One decode step's seeded inputs, element-dtype bits:
  ##
  ## | field        | shape                 |
  ## | ------------ | --------------------- |
  ## | qBits, kBits | (B·Hk, Dk)            |
  ## | vBits        | (B·Hv, Dv)            |
  ## | betaBits     | (B·Hv,)               |
  ## | gVals        | (B·Hv,) f32 log-decay |
  qBits: seq[uint16]
  kBits: seq[uint16]
  vBits: seq[uint16]
  betaBits: seq[uint16]
  gVals: seq[float32]

type StepSnap = object
  ## Bit records of one step's kernel-written buffers.
  state: seq[float32]
  y: seq[uint16]

var suiteCases, suiteLaunches = 0

proc runCombo(engine: HwEngine, dt: ScalarKind, Hv, Hk, hkRatio, B, dk, steps, cases: int, seed: uint64, label: string, gLoOverride = 0.0'f32, gHiOverride = 0.0'f32, betaZero = false) =
  ## One (element dtype, shape) combination over `cases` independent seeded
  ## runs of `steps` decode steps each, case 0 relaunched bit-identical.
  const Dv = 16
  const TileR = 8
  let bhMax = B * Hv
  let qkRows = B * Hk
  let stateElems = bhMax * Dv * dk
  let kernelName = if dt == kFloat16: "cer_gdn_step_fp16_dk32" else: "cer_gdn_step_bf16_dk32"
  # the span overrides exist for the edge combos, gLo 0.0 is the sentinel
  # meaning derive the span from the step count (no edge case wants gLo = 0)
  let gLo = if gLoOverride != 0.0'f32: gLoOverride
            else: (if steps == 1: -3.0'f32 else: -0.5'f32)
  let gHi = if gLoOverride != 0.0'f32: gHiOverride
            else: (if steps == 1: -0.1'f32 else: -0.01'f32)

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
    engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
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

  # One step's launch and snap, the buffers carrying into the next step.
  proc runSteps(state0: seq[float32], chain: seq[StepInputs], snaps: var seq[StepSnap]) =
    setState(state0)
    for si in chain:
      copyStepInputs(si)
      launch(si)
      sentinels(si)
      snaps.add(StepSnap(
        state: readRecord(stateB.hostPtr, stateElems),
        y: readRecord(yB.hostPtr, bhMax * Dv)))

  proc takeInputs(rng: var PropRng): seq[StepInputs] =
    ## Seeded inputs for one chain, element-dtype bits for every step.
    for step in 0 ..< steps:
      var qBits = newSeq[uint16](qkRows * dk)
      var kBits = newSeq[uint16](qkRows * dk)
      var vBits = newSeq[uint16](bhMax * Dv)
      var betaBits = newSeq[uint16](bhMax)
      var gVals = newSeq[float32](bhMax)
      for i in 0 ..< qkRows * dk:
        qBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
        kBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
      for i in 0 ..< bhMax * Dv:
        vBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
      for h in 0 ..< bhMax:
        betaBits[h] = (if betaZero: 0.0'f32.narrowTo(dt)
                       else: rng.nextF32(0.2'f32, 0.8'f32).narrowTo(dt))
        gVals[h] = rng.nextF32(gLo, gHi)
      result.add(StepInputs(qBits: qBits, kBits: kBits, vBits: vBits,
        betaBits: betaBits, gVals: gVals))

  var case0Snaps: seq[StepSnap]             # the determinism reference

  var rng = initPropRng(seed)
  for caseId in 0 ..< cases:
    var state0 = newSeq[float32](stateElems)
    for i in 0 ..< stateElems:
      state0[i] = rng.nextF32(-1.0'f32, 1.0'f32)
    let chain = takeInputs(rng)
    var caseSnaps: seq[StepSnap]
    runSteps(state0, chain, caseSnaps)
    if caseId == 0:
      case0Snaps = caseSnaps

  # determinism relaunch of case 0, bit-identical across launches
  block determinism:
    var rng0 = initPropRng(seed)
    var state0 = newSeq[float32](stateElems)
    for i in 0 ..< stateElems:
      state0[i] = rng0.nextF32(-1.0'f32, 1.0'f32)
    let chain = takeInputs(rng0)
    var relaunchSnaps: seq[StepSnap]
    runSteps(state0, chain, relaunchSnaps)
    for t in 0 ..< relaunchSnaps.len:
      for i in 0 ..< stateElems:
        doAssert relaunchSnaps[t].state[i] == case0Snaps[t].state[i],
          "state differs run to run"
      for i in 0 ..< bhMax * Dv:
        doAssert relaunchSnaps[t].y[i] == case0Snaps[t].y[i],
          "y differs run to run"

  echo &"[{label} Dk={dk}] steps={steps} cases={cases} launches={launches} " &
    &"relaunch bit-identical"
  suiteCases += cases
  suiteLaunches += launches

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(GdnDecodeMsl)

  proc secF16Baseline =
    let t0 = epochTime()
    runCombo(engine, kFloat16, 1, 1, 1, 1, 32, 1, 64, 0xC04D0401'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secF16Gqa =
    let t0 = epochTime()
    runCombo(engine, kFloat16, 4, 2, 2, 2, 32, 1, 64, 0xC04D0402'u64,
      "gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Baseline =
    let t0 = epochTime()
    runCombo(engine, kBfloat16, 1, 1, 1, 1, 32, 1, 64, 0xC04D0403'u64,
      "baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secBf16Gqa =
    let t0 = epochTime()
    runCombo(engine, kBfloat16, 4, 2, 2, 2, 32, 1, 64, 0xC04D0404'u64,
      "gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secChainF16Gqa =
    let t0 = epochTime()
    runCombo(engine, kFloat16, 4, 2, 2, 2, 32, 10, 2, 0xC04D0405'u64,
      "chain gqa Hk=2/Hv=4/B=2")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secChainBf16Baseline =
    let t0 = epochTime()
    runCombo(engine, kBfloat16, 1, 1, 1, 1, 32, 10, 2, 0xC04D0406'u64,
      "chain baseline Hk=1/Hv=1/B=1")
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secEdgeNearZeroG =
    let t0 = epochTime()
    runCombo(engine, kFloat16, 1, 1, 1, 1, 32, 1, 64, 0xC04D04A7'u64,
      "edge g->0- Hk=1/Hv=1/B=1", -0.001'f32, 0.0'f32)
    echo &"  wall clock {epochTime() - t0:.2f} s"

  proc secEdgeBetaZero =
    let t0 = epochTime()
    runCombo(engine, kBfloat16, 1, 1, 1, 1, 32, 1, 64, 0xC04D04A8'u64,
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
  echo &"CERAMIC GDN DECODE VERDICT: cases={suiteCases} launches={suiteLaunches} " &
    &"relaunch bit-identical across all combinations"

main()
