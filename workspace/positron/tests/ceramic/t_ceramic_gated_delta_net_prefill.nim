# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

## Run commands, from the repo root (the aggregate runner is nim test_positron_properties):
## - nim c -r -d:release --warnings:off --outdir:build/tests workspace/positron/tests/ceramic/t_ceramic_gated_delta_net_prefill.nim
##
## Ceramic GDN prefill suite, chunked-scan launch contract over seeded fixtures:
## - `src/kernels/ceramic/attn_ssm/gated_delta_net_prefill.nim` runs one launch
##   over the stacked head axis (B·Hv threadgroups, one per head), the launch walking all chunks
##
## Value check, the property suite's split invariance plus the decode
## step-count identity against the same prefill (`tests/properties/t_prop_gdn.nim`),
## whole vs split chunking.
##
## This suite carries the fixture, sentinel and determinism checks:
##
## - run-to-run determinism, case 0 relaunched bit-identical per combination
## - untouched-memory checks per launch, kernel-written buffers stay in their extents, kernel-read buffers stay bit-identical
##
## Inputs, all seeded xorshift64, fixture-free:
##
## | input     | rule                                                                                                                    |
## | --------- | ----------------------------------------------------------------------------------------------------------------------- |
## | q, k      | l2-normalized per (head, token) row in fp32, then rounded to the element dtype, the kernel contract's post-l2norm shape |
## | v, state0 | [-1, 1), the initial state never zero                                                                                   |
## | beta      | [0.2, 0.8), edge combos carry the exact-zero beta and the near-zero decay (g -> 0-)                                     |
## | g         | [-0.5, -0.01) log-decay                                                                                                 |
##
## - the q, k normalization also keeps the delta-rule recursion bounded over 256 tokens in fp16 y range
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
## - adjudicated on Apple M4 Max with fresh seeded xorshift64 inputs

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/attn_ssm/gated_delta_net_prefill
import ceramic_pagebuf
import ceramic_dtype

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

type PrefillInputs = object
  ## One case's seeded inputs, element-dtype bits:
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

var suiteCases, suiteLaunches = 0

proc runCase(engine: HwEngine, dt: ScalarKind, Hv, Hk, hkRatio, B, T, chunkLen: int, seed: uint64, label: string, gLoOverride = 0.0'f32, gHiOverride = 0.0'f32, betaZero = false) =
  ## One (element dtype, shape) combination, case 0 relaunched bit-identical.
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

  proc run(si: PrefillInputs) =
    fillInputs(si)
    launch(si)
    sentinels(si)

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
    run(si)
    if caseId == 0: case0 = record()
  # determinism relaunch of case 0, bit-identical across launches
  block determinism:
    var rng0 = initPropRng(seed)
    let si = takeInputs(dt, rng0, bhMax, qkRows, T, Dv, Dk, gLoOverride,
      gHiOverride, betaZero)
    run(si)
    let again = record()
    for i in 0 ..< stateElems:
      doAssert again.st[i] == case0.st[i], "state differs run to run"
    for i in 0 ..< yElems:
      doAssert again.y[i] == case0.y[i], "y differs run to run"

  echo &"[{label} T={T} C={chunkLen}] cases={cases} launches={launches} " &
    &"relaunch bit-identical"
  suiteCases += cases
  suiteLaunches += launches

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
    &"relaunch bit-identical across all combinations"

main()
