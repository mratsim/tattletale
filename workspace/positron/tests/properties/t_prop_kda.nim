# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_properties
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests positron/tests/properties/t_prop_kda.nim
##
## Property suite over the Kimi delta attention prefill scan and decode step, the production kernels from `src/kernels/ceramic/attn_ssm/`.
## - split invariance, one whole scan equals the segmented scan (y and state)
## - step-count identity, k decode steps equal the prefill through the k tokens
##
## Judged per element, the two kernel sides against each other, no value
## reference of any kind
## Pass band from the case's own g values, stated inside the judge docs below
##
import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/attn_ssm/gated_delta_net_kda_prefill
import ../../src/kernels/ceramic/attn_ssm/gated_delta_net_kda_decode_single
import properties
import ../ceramic/ceramic_pagebuf

# ─── Device entries, fp16 binding ─────────────────────────────────────

const KdaPropMsl = metal:
  proc prop_kda_prefill_fp16_c32(
      state: ptr UncheckedArray[float32],
      y, v: ptr UncheckedArray[float16],
      k, q, cumulogdecay, beta: ptr UncheckedArray[float32],
      qScale: float32,
      Hv, Hk, hkRatio, T: int32) {.global.} =
    kdaPrefillChunkScan(state, y, k, q, cumulogdecay, v, beta, qScale,
      Hv, Hk, hkRatio, T, 32, 16, 8, 32)

  proc prop_kda_step_fp16(
      state: ptr UncheckedArray[float32],
      y: ptr UncheckedArray[float16],
      k, q: ptr UncheckedArray[float32],
      v: ptr UncheckedArray[float16],
      g, beta: ptr UncheckedArray[float32],
      qScale: float32,
      Hv, Hk, hkRatio: int32) {.global.} =
    kdaDecodeStepTile(state, y, k, q, v, g, beta, qScale, Hv, Hk, hkRatio, 32, 16, 8)

# ─── The host cumulogdecay, one side's per-channel prefix ─────────────

proc kdaHostCld(g: seq[float32], qkRows, T, Dk, ChunkC: int, segments: seq[int]): seq[float32] =
  ## Host per-channel cumulative log decay feeding the prefill side, the inclusive
  ## prefix of g restarting at every segment start and at every t that satisfies
  ## (t − segment start) mod ChunkC == 0, the kernel's chunk contract.
  result = newSeq[float32](qkRows * T * Dk)
  var segStart = 0
  for segLen in segments:
    var c0 = 0
    while c0 < segLen:
      let cLen = min(ChunkC, segLen - c0)
      for h in 0 ..< qkRows:
        for dk in 0 ..< Dk:
          var acc = 0.0'f32
          for t in 0 ..< cLen:
            acc += g[(h * T + segStart + c0 + t) * Dk + dk]
            result[(h * T + segStart + c0 + t) * Dk + dk] = acc
      c0 += cLen
    segStart += segLen

proc segmentCld(g: seq[float32], qkRows, T, Dk, ChunkC, t0, segLen: int): seq[float32] =
  ## Per-channel cumulative log decay of one segment launch, the inclusive prefix of g packed (segLen, qkRows, Dk) over the segment's token rows,
  ##   restarting at the segment start and at every t that satisfies (t − start) mod ChunkC == 0, the chunk contract.
  ##

  result = newSeq[float32](segLen * qkRows * Dk)
  var c0 = 0
  while c0 < segLen:
    let cLen = min(ChunkC, segLen - c0)
    for h in 0 ..< qkRows:
      for dk in 0 ..< Dk:
        var acc = 0.0'f32
        for t in 0 ..< cLen:
          acc += g[(h * T + t0 + c0 + t) * Dk + dk]
          result[(h * segLen + c0 + t) * Dk + dk] = acc
    c0 += cLen

# ─── Device-side run machinery ────────────────────────────────────────

const
  Dv = 16
  Dk = 32
  TileR = 8
  FloorSub = 2.9802322387695312e-8   # 2⁻²⁵, half the constant fp16 subnormal ulp

type KdaRun* = object
  ## One side's kernel outputs, y as element bits per token and the final state.
  y: seq[uint16]
  state: seq[float32]

proc runKdaScan(
    engine: HwEngine, c: RecCase,
    bhMax, qkRows, T, t0, n, Hv, Hk: int,
    cldSeg: seq[float32],
    stateIn: seq[float32],
    stateB: PageBuf[float32], yB: PageBuf[uint16]): KdaRun =
  ## Launches the prefill chunk scan over the token rows [t0, t0 + n), the host
  ## cumulogdecay `cldSeg` already carrying the side's own resets, the state loaded
  ## from `stateIn` in place, y and the end state read back.
  let stateElems = bhMax * Dv * Dk
  for i in 0 ..< stateElems: stateB.hostPtr[i] = stateIn[i]
  for i in 0 ..< n * bhMax * Dv: yB.hostPtr[i] = 0
  var kSeg = allocPageBuf[float32](n * qkRows * Dk)
  var qSeg = allocPageBuf[float32](n * qkRows * Dk)
  var cldSegB = allocPageBuf[float32](n * qkRows * Dk)
  var vSeg = allocPageBuf[uint16](n * bhMax * Dv)
  var betaSeg = allocPageBuf[float32](n * bhMax)
  defer:
    freePageBuf(kSeg); freePageBuf(qSeg); freePageBuf(cldSegB)
    freePageBuf(vSeg); freePageBuf(betaSeg)
  # the segment's token rows scatter per head, the kernels' token axis is
  # head-interleaved in the stacked layout
  for h in 0 ..< qkRows:
    for j in 0 ..< n:
      for dk in 0 ..< Dk:
        kSeg.hostPtr[(h * n + j) * Dk + dk] =
          c.kBits[((h * T) + t0 + j) * Dk + dk].widenTo(kFloat16)
        qSeg.hostPtr[(h * n + j) * Dk + dk] =
          c.qBits[((h * T) + t0 + j) * Dk + dk].widenTo(kFloat16)
        cldSegB.hostPtr[(h * n + j) * Dk + dk] = cldSeg[(h * n + j) * Dk + dk]
  for h in 0 ..< bhMax:
    for j in 0 ..< n:
      for r in 0 ..< Dv:
        vSeg.hostPtr[(h * n + j) * Dv + r] = c.vBits[((h * T) + t0 + j) * Dv + r]
      betaSeg.hostPtr[h * n + j] = c.betaBits[(h * T) + t0 + j].widenTo(kFloat16)
  var stPA = stateB.pa()
  engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
    ("prop_kda_prefill_fp16_c32", stPA,
      (yB.pa(), vSeg.pa(), kSeg.pa(), qSeg.pa(), cldSegB.pa(), betaSeg.pa(),
        float32(sqrt(float64(Dk))),
        int32(Hv), int32(Hk), int32(Hv div Hk), int32(n)))
  var st = newSeq[float32](stateElems)
  for i in 0 ..< stateElems: st[i] = stateB.hostPtr[i]
  var yy = newSeq[uint16](n * bhMax * Dv)
  for i in 0 ..< n * bhMax * Dv: yy[i] = yB.hostPtr[i]
  KdaRun(y: yy, state: st)

proc runKdaSteps(
    engine: HwEngine, c: RecCase,
    bhMax, qkRows, T, Hv, Hk: int,
    stateB: PageBuf[float32], yB: PageBuf[uint16]): KdaRun =
  ## Launches T single-token decode steps over the same per-token rows, the state
  ## carried in place across launches, one y row read back per step.
  let stateElems = bhMax * Dv * Dk
  var state = c.state0
  var yAll = newSeq[uint16](T * bhMax * Dv)
  for t in 0 ..< T:
    for i in 0 ..< stateElems: stateB.hostPtr[i] = state[i]
    var kTok = allocPageBuf[float32](qkRows * Dk)
    var qTok = allocPageBuf[float32](qkRows * Dk)
    var gTok = allocPageBuf[float32](qkRows * Dk)
    var vTok = allocPageBuf[uint16](bhMax * Dv)
    var betaTok = allocPageBuf[float32](bhMax)
    for h in 0 ..< qkRows:
      for dk in 0 ..< Dk:
        kTok.hostPtr[h * Dk + dk] = c.kBits[((h * T) + t) * Dk + dk].widenTo(kFloat16)
        qTok.hostPtr[h * Dk + dk] = c.qBits[((h * T) + t) * Dk + dk].widenTo(kFloat16)
        gTok.hostPtr[h * Dk + dk] = c.gVals[((h * T) + t) * Dk + dk]
    for h in 0 ..< bhMax:
      for r in 0 ..< Dv: vTok.hostPtr[h * Dv + r] = c.vBits[((h * T) + t) * Dv + r]
      betaTok.hostPtr[h] = c.betaBits[(h * T) + t].widenTo(kFloat16)
    var stPA = stateB.pa()
    engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
      ("prop_kda_step_fp16", stPA,
        (yB.pa(), kTok.pa(), qTok.pa(), vTok.pa(), gTok.pa(), betaTok.pa(),
          float32(sqrt(float64(Dk))),
          int32(Hv), int32(Hk), int32(Hv div Hk)))
    for i in 0 ..< stateElems: state[i] = stateB.hostPtr[i]
    for i in 0 ..< bhMax * Dv: yAll[t * bhMax * Dv + i] = yB.hostPtr[i]
    freePageBuf(kTok); freePageBuf(qTok); freePageBuf(gTok)
    freePageBuf(vTok); freePageBuf(betaTok)
  KdaRun(y: yAll, state: state)

# ─── Judgment ─────────────────────────────────────────────────────────

const
  # Pass bands from the case, not fitted to the answer
  # - the state band is restarts x T x max|g| x u32 plus the exp2 and log2e roundings, two per side
  # - the y band is one fp16 ulp of the magnitude, one flip per side
  PrefixU32 = 5.9604644775390625e-8   # 2^-24, fp32 unit roundoff
  YFlip = 9.765625e-4                 # 2^-10, one fp16 ulp of the magnitude

proc judgeY(j: var Judge, ra, rb: KdaRun, bhMax, T: int, label: string) =
  ## Judges y per element, kernel A against kernel B
  ## Pass band is one fp16 ulp per side plus the 2^-25 subnormal floor
  for i in 0 ..< bhMax * T * Dv:
    let yA = fp16ToFp32(ra.y[i]).float64
    let yB = fp16ToFp32(rb.y[i]).float64
    let mag = max(abs(yA), abs(yB))
    j.judge(&"{label} y elem {i}", yA, yB, mag * (2.0 * YFlip) + FloorSub,
      ulpStepAt(ulpFp16, mag))

proc judgeState(j: var Judge, ra, rb: KdaRun, bhMax: int, allow: float64, label: string) =
  ## Judges the final state per element against the case's absolute band
  ## - the band carries the state's global scale, the terms feeding
  ##   one element share the input scale, a cancelling element
  ##   sits far below the scale of the terms that feed it
  let stateElems = bhMax * Dv * Dk
  for i in 0 ..< stateElems:
    j.judge(&"{label} state elem {i}", ra.state[i].float64, rb.state[i].float64, allow, 0.0)

# ─── Main, the property combinations ──────────────────────────────────

var suiteCases, suiteLaunches = 0
var suiteWorstUse, suiteWorstUlp, suiteWorstAbs = 0.0'f64
var suiteExact, suiteTotal = 0

proc runCase(engine: HwEngine, seed: uint64, Hv, Hk, B, T: int, splitsA, splitsB: seq[int], decodeSide: int, label: string) =
  ## One property combination, two kernel sides over the same seeded case
  ##
  ## - `decodeSide` 1 runs side A on the decode steps, side B on the chunk scan
  ## - `decodeSide` 2 swaps the two, 0 runs both sides on the chunk scan
  let bhMax = B * Hv
  let qkRows = B * Hk
  var rng = initPropRng(seed)
  let c = takeRecCase(rng, kFloat16, qkRows, bhMax, T, Dv, Dk, kda = true)
  var stateB = allocPageBuf[float32](bhMax * Dv * Dk)
  var yB = allocPageBuf[uint16](T * bhMax * Dv)
  defer:
    freePageBuf(stateB); freePageBuf(yB)

  var cldA = kdaHostCld(c.gVals, qkRows, T, Dk, 32, splitsA)
  var cldB = kdaHostCld(c.gVals, qkRows, T, Dk, 32, splitsB)

  var runA: KdaRun
  var runB: KdaRun
  var launches = 0
  if decodeSide == 1:
    runA = runKdaSteps(engine, c, bhMax, qkRows, T, Hv, Hk, stateB, yB)
    inc launches, T
    runB = runKdaScan(engine, c, bhMax, qkRows, T, 0, T, Hv, Hk, cldB, c.state0, stateB, yB)
    inc launches
  elif decodeSide == 2:
    runA = runKdaScan(engine, c, bhMax, qkRows, T, 0, T, Hv, Hk, cldA, c.state0, stateB, yB)
    inc launches
    runB = runKdaSteps(engine, c, bhMax, qkRows, T, Hv, Hk, stateB, yB)
    inc launches, T
  else:
    var t0 = 0
    var yA = newSeq[uint16](T * bhMax * Dv)
    var stA = c.state0
    for segLen in splitsA:
      # the side's own cld slice, resets recomputed from the segment start
      let seg = runKdaScan(engine, c, bhMax, qkRows, T, t0, segLen, Hv, Hk,
        segmentCld(c.gVals, qkRows, T, Dk, 32, t0, segLen),
        stA, stateB, yB)
      for bh in 0 ..< bhMax:
        for j in 0 ..< segLen:
          for r in 0 ..< Dv:
            yA[((bh * T) + t0 + j) * Dv + r] = seg.y[(bh * segLen + j) * Dv + r]
      stA = seg.state
      t0 += segLen
      inc launches
    var t0B = 0
    var yBs = newSeq[uint16](T * bhMax * Dv)
    var stBs = c.state0
    for segLen in splitsB:
      let seg = runKdaScan(engine, c, bhMax, qkRows, T, t0B, segLen, Hv, Hk,
        segmentCld(c.gVals, qkRows, T, Dk, 32, t0B, segLen),
        stBs, stateB, yB)
      for bh in 0 ..< bhMax:
        for j in 0 ..< segLen:
          for r in 0 ..< Dv:
            yBs[((bh * T) + t0B + j) * Dv + r] = seg.y[(bh * segLen + j) * Dv + r]
      stBs = seg.state
      t0B += segLen
      inc launches
    runA = KdaRun(y: yA, state: stA)
    runB = KdaRun(y: yBs, state: stBs)

  var j = Judge()
  judgeY(j, runA, runB, bhMax, T, label)
  let gmax = block:
    var m = 0.0'f32
    for g in c.gVals: m = max(m, abs(g))
    m.float64
  # the state band carries the state's global scale, not the element's value
  # cancelling elements sit far below the scale of the terms that feed it
  var stateScale = 0.0'f64
  for i in 0 ..< bhMax * Dv * Dk:
    stateScale = max(stateScale, abs(runA.state[i].float64))
    stateScale = max(stateScale, abs(runB.state[i].float64))
  let bandRel = (2.0 * float64(max(splitsA.len, splitsB.len)) * float64(T) *
    gmax + 8.0) * PrefixU32
  judgeState(j, runA, runB, bhMax, stateScale * bandRel + FloorSub, label)

  echo &"[{label} Hv={Hv} Hk={Hk} B={B} T={T}] launches={launches} " &
    &"worst |Δ| {j.worstAbs:.3e}, worst usage {j.worstUse:.3f}, " &
    &"worst {j.worstUlp:.2f} ulp, bit-exact {j.exact}/{j.total}"
  suiteCases += 1
  suiteLaunches += launches
  suiteWorstUse = max(suiteWorstUse, j.worstUse)
  suiteWorstUlp = max(suiteWorstUlp, j.worstUlp)
  suiteWorstAbs = max(suiteWorstAbs, j.worstAbs)
  suiteExact += j.exact
  suiteTotal += j.total

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(KdaPropMsl)
  runCase(engine, 0xC04D0621'u64, 1, 1, 1, 64, @[64], @[32, 32], 0, "split aligned")
  runCase(engine, 0xC04D0622'u64, 1, 1, 1, 64, @[64], @[40, 24], 0, "split ragged")
  runCase(engine, 0xC04D0623'u64, 1, 1, 1, 64, @[64], @[3, 61], 0, "split head ragged")
  runCase(engine, 0xC04D0624'u64, 1, 1, 1, 64, @[64], @[20, 20, 24], 0, "split three-way")
  runCase(engine, 0xC04D0625'u64, 1, 1, 1, 8, @[8], @[3, 5], 0, "split single chunk")
  runCase(engine, 0xC04D0626'u64, 2, 1, 2, 64, @[64], @[32, 32], 0, "split heads aligned")
  runCase(engine, 0xC04D0627'u64, 1, 1, 1, 8, @[1, 1, 1, 1, 1, 1, 1, 1], @[8], 1,
    "steps T=8 vs prefill")
  var ones33: seq[int]
  for i in 0 ..< 33: ones33.add(1)
  runCase(engine, 0xC04D0628'u64, 1, 1, 1, 33, @[33], ones33, 2, "steps T=33 vs prefill")
  echo &"PROPERTIES KDA VERDICT: cases={suiteCases} launches={suiteLaunches} " &
    &"worst |Δ| {suiteWorstAbs:.3e}, worst usage {suiteWorstUse:.3f}, " &
    &"worst {suiteWorstUlp:.2f} ulp, bit-exact {suiteExact}/{suiteTotal}"

main()
