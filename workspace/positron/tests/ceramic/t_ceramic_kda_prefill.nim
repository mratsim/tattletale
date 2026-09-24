# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

## Run commands, from the repo root (the aggregate runner is nim test_positron_properties):
## - nim c -r -d:release --warnings:off --outdir:build/tests workspace/positron/tests/ceramic/t_ceramic_kda_prefill.nim
##
## Ceramic KDA prefill suite, chunked-scan launch contract over seeded fixtures:
## - `src/kernels/ceramic/attn_ssm/gated_delta_net_kda_prefill.nim` runs one launch over
##   the stacked head axis (B·Hv threadgroups, one per head), the launch walking all chunks
##
## Value check, the property suite's split invariance plus the decode
## step-count identity against the same prefill (`tests/properties/t_prop_kda.nim`),
## whole vs split chunking.
##
## This suite carries the fixture, sentinel, finiteness and determinism checks:
##
## - run-to-run determinism, case 0 relaunched bit-identical per combination
## - untouched-memory checks per launch, kernel-written buffers stay in their extents, kernel-read buffers stay bit-identical
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
##   |cumulogdecay| crosses the exp2 overflow bound 88.7 inside the chunk, the
##   decay factors flush to fp32 zero and the outputs are asserted finite
##
## - every case starts from a non-zero random initial state
## - the GQA shape keeps both head-mapping terms live, sequence 1 holding independent key heads
##
## - fp16 is the element dtype under test, bf16 the range-robust fallback
## - adjudicated on Apple M4 Max with fresh seeded xorshift64 inputs

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/attn_ssm/gated_delta_net_kda_prefill
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

type PrefillInputs = object
  ## One case's seeded inputs, f32 values verbatim, the per-channel
  ## cumulogdecay host-computed from g:
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

proc l2NormalizeRowsF32(dst: var seq[float32], rows, cols: int, rng: var PropRng) =
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
  ## of g per (head, chunk, channel), the kernel's decay input.
  for hk in 0 ..< qkRows:
    for t in 0 ..< T:
      for dk in 0 ..< Dk:
        let gi = (hk * T + t) * Dk + dk
        if t mod chunkLen == 0:
          dst[gi] = g[gi]
        else:
          dst[gi] = dst[gi - Dk] + g[gi]

proc takeInputs(dt: ScalarKind, rng: var PropRng, bhMax, qkRows, T, Dv, Dk, chunkLen: int, betaZero: bool, overflowG = false): PrefillInputs =
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

var suiteCases, suiteLaunches = 0

proc runCase(engine: HwEngine, dt: ScalarKind, Hv, Hk, hkRatio, B, T, chunkLen, Dk: int, betaZero: bool, seed: uint64, label: string, overflowG = false) =
  ## One (element dtype, shape) combination, case 0 relaunched bit-identical.
  ##
  ## `overflowG` runs the decay-overflow fixture, |cumulogdecay| past the exp2 overflow
  ## bound 88.7 inside the first chunk, the y output and the carried state asserted finite
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

  proc run(si: PrefillInputs) =
    fillInputs(si)
    launch(si)
    sentinels(si)
    if overflowG:
      # the fixture's exact finiteness check, one NaN or Inf in the y
      # output or the carried state fails the case outright
      for i in 0 ..< yElems:
        let yc = classify(yB.hostPtr[i].widenTo(dt).float64)
        doAssert yc notin {fcNan, fcInf, fcNegInf},
          &"y not finite at element {i} (classify {yc})"
      for i in 0 ..< stateElems:
        let sc = classify(stateB.hostPtr[i].float64)
        doAssert sc notin {fcNan, fcInf, fcNegInf},
          &"carried state not finite at element {i} (classify {sc})"

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
    let si = takeInputs(dt, rng, bhMax, qkRows, T, Dv, Dk, chunkLen, betaZero,
      overflowG)
    run(si)
    if caseId == 0: case0 = record()
  # determinism relaunch of case 0, bit-identical across launches
  block determinism:
    var rng0 = initPropRng(seed)
    let si = takeInputs(dt, rng0, bhMax, qkRows, T, Dv, Dk, chunkLen, betaZero,
      overflowG)
    run(si)
    let again = record()
    for i in 0 ..< stateElems:
      doAssert again.st[i] == case0.st[i], "state differs run to run"
    for i in 0 ..< yElems:
      doAssert again.y[i] == case0.y[i], "y differs run to run"

  let betaTag = (if betaZero: " beta=0" else: "") &
    (if overflowG: " overflow-g" else: "")
  echo &"[{label} T={T} C={chunkLen} Dk={Dk}{betaTag}] cases={cases} " &
    &"launches={launches} relaunch bit-identical" &
    (if overflowG: ", outputs finite" else: "")
  suiteCases += cases
  suiteLaunches += launches

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
    &"relaunch bit-identical across all combinations, overflow outputs finite"

main()
