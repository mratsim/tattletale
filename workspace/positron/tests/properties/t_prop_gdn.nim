# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_properties
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests workspace/positron/tests/properties/p_gdn.nim
##
## Property suite over the production GDN kernels of the gated delta rule, judging
## the chunked prefill scan against the single-token decode step of one recurrence:
##
## | property            | what it asserts                           |
## | ------------------- | ----------------------------------------- |
## | split invariance    | whole scan = segmented scan, y and state  |
## | step-count identity | k decode steps = prefill through k tokens |
##
## Under judgment, both entries from `src/kernels/ceramic/attn_ssm/`:
##
## | kernel              | module                            |
## | ------------------- | --------------------------------- |
## | gdnPrefillChunkScan | gated_delta_net_prefill.nim       |
## | gdnDecodeStepTile   | gated_delta_net_decode_single.nim |
##
##
## Both sides are the same kernels at different token decompositions, the divergence
## judged pure fp32 rounding, no value reference of any kind:
##
## - the sides share the exact widened inputs bit for bit, the state math staying fp32 and never rounding, the y store rounding once per launch
## - the sides differ in the decay-factored grouping, an exp2 of a prefix sum against a product of per-token factors at every chunk boundary
## - the sides differ further in the fp32 sum reassociations around the moved segment boundaries
##
## Allowance model, stated before measurement and judged per element,
## `allow = sideA.dTerm + sideB.dTerm`:
##
## | term       | bound                                                |
## | ---------- | ---------------------------------------------------- |
## | dTerm/side | one side's additive fp32 bound vs its own fp64 walk  |
## | relDecay   | prefix chain error + the exp2 form terms, per side   |
## | sum terms  | summed terms · u₃₂ of the summed absolute magnitudes |
## | y store    | one RNE per side + the 2⁻²⁵ subnormal floor          |
##
## - each dTerm is that side's additive fp32-rounding bound against its own exact fp64
##   structure walk, the walk mirroring the kernel's chunked formulas at the matching token decomposition, values only as magnitude scale
## - relDecay carries the fp32 prefix-of-g chain error, the n terms · u₃₂ · cmax bound
##   of the fp32 prefix, plus 4·u₃₂ for the log2e multiply and the exp2 form, with n the prefix length and cmax the chain's max magnitude
## - dots add Dk terms · u₃₂, the state judgment carrying no store rounding, both exact walks representing one real recurrence,
##   their difference staying fp64 noise below every judged term, the measured worst usage corroborating the model and never setting a bound
##
## Shapes (Dk = 32, Dv = 16, TileR = 8, ChunkC = 32, grid (2, bhMax), 32 lanes, fp16):
##
## | property            | shape                 | T  | sides                    |
## | ------------------- | --------------------- | --- | ------------------------ |
## | split invariance    | Hk = Hv = B = 1       | 64 | whole, [32,32], [40,24], |
## |                     |                       |    | [3,61], [20,20,24]       |
## | split invariance    | Hk = Hv = B = 1       | 8  | whole, [3,5]             |
## | split invariance    | Hk = 1, Hv = 2, B = 2 | 64 | whole, [32,32], [40,24]  |
## | step-count identity | Hk = Hv = B = 1       | 8  | prefill, 8 decode steps  |
## | step-count identity | Hk = Hv = B = 1       | 33 | prefill, 33 decode steps |
##
## - the T = 8 and T = 64 whole runs bracket the carried-error recursion depth at one
##   chunk versus two, and the T = 33 case adds a ragged tail chunk of 1 token
## - the 4-head case keeps the head-mapping term live under the split
## - inputs seeded per case, one seeded case per combination, q and k l2-normalized
import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/attn_ssm/gated_delta_net_prefill
import ../../src/kernels/ceramic/attn_ssm/gated_delta_net_decode_single
import properties
import ../ceramic/ceramic_pagebuf

# ─── Device entries, fp16 binding ─────────────────────────────────────

const GdnPropMsl = metal:
  proc prop_gdn_prefill_fp16_c32(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[float16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio, T: int32) {.global.} =
    gdnPrefillChunkScan(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, T, 32, 16, 8, 32)

  proc prop_gdn_step_fp16(
      state: ptr UncheckedArray[float32],
      y, k, q, v, beta: ptr UncheckedArray[float16],
      g: ptr UncheckedArray[float32],
      Hv, Hk, hkRatio: int32) {.global.} =
    gdnDecodeStepTile(state, y, k, q, v, g, beta, Hv, Hk, hkRatio, 32, 16, 8)

# ─── The fp64 structure walk, one side's magnitudes and error bounds ──

type GdnWalk = object
  ## One side's fp64 walk over the kernel's chunked formulas at the side's
  ## token decomposition, values only as magnitude scale:
  ##
  ## | field | shape           |
  ## | ----- | --------------- |
  ## | yVal  | (bhMax, T, Dv)  |
  ## | dY    | (bhMax, T, Dv)  |
  ## | sVal  | (bhMax, Dv, Dk) |
  ## | dS    | (bhMax, Dv, Dk) |
  yVal: seq[float64]
  dY: seq[float64]
  sVal: seq[float64]
  dS: seq[float64]

proc gdnWalkCase(
    w: var GdnWalk,
    qw, kw, vw, bw: seq[float64], gw: seq[float64], s0w: seq[float64],
    bhMax, qkRows, Hv, Hk, T, Dv, Dk, ChunkC: int,
    segments: seq[int]) =
  ## Walks the chunked GDN formulas (arXiv:2412.06464) in fp64 over the exact
  ## widened inputs, restarting at every segment boundary, and accumulates the additive
  ## fp32-rounding bounds of the side this decomposition models:
  ##
  ##   per chunk:  cumulogdecay ─→ per token t: G_t read ─→ u_t solve ─→ y_t
  ##               (t in token order)                └─────────┐
  ##           └──→ S ← decayed carry read + Σ_s pd(end, s)·k_s [x] u_s
  ##
  ## - a segment boundary restarts the walk with the previous segment's carry
  ##   state and its error bound, the fp32 state handoff is exact
  ## - `dY`/`dS` carry that side's error recursion, `yVal`/`sVal` the fp64
  ##   magnitudes the scale terms multiply
  w.yVal = newSeq[float64](bhMax * T * Dv)
  w.dY = newSeq[float64](bhMax * T * Dv)
  w.sVal = newSeq[float64](bhMax * Dv * Dk)
  w.dS = newSeq[float64](bhMax * Dv * Dk)

  for bh in 0 ..< bhMax:
    let hk = (bh mod Hv) div (Hv div Hk) + (bh div Hv) * Hk
    var carry = newSeq[float64](Dv * Dk)
    var dS = newSeq[float64](Dv * Dk)
    for r in 0 ..< Dv:
      for dk in 0 ..< Dk:
        carry[r * Dk + dk] = s0w[(bh * Dv + r) * Dk + dk]
    var kdot = newSeq[float64](ChunkC * ChunkC)
    var kkAbs = newSeq[float64](ChunkC * ChunkC)
    var pdRef = newSeq[float64](ChunkC * ChunkC)
    var qkdot = newSeq[float64](ChunkC * ChunkC)
    var qkAbs = newSeq[float64](ChunkC * ChunkC)
    var bAbs = newSeq[float64](ChunkC * ChunkC)
    var segStart = 0
    for segLen in segments:
      var c0 = 0
      while c0 < segLen:
        let cLen = min(ChunkC, segLen - c0)
        var cumulogdecay = newSeq[float64](cLen)
        cumulogdecay[0] = gw[bh * T + segStart + c0]
        var cmax = abs(cumulogdecay[0])
        for i in 1 ..< cLen:
          cumulogdecay[i] = cumulogdecay[i - 1] + gw[bh * T + segStart + c0 + i]
          cmax = max(cmax, abs(cumulogdecay[i]))
        # the fp32 prefix-of-g chain error plus the log2e multiply and the exp2 form,
        # the decay factors' relative bound on this side
        let decRel = float64(cLen) * U32 * cmax + 4.0 * U32
        var uVal = newSeq[float64](cLen * Dv)
        var du = newSeq[float64](cLen * Dv)
        for t in 0 ..< cLen:
          let gt = segStart + c0 + t
          let pdT = exp(cumulogdecay[t])
          let betaT = bw[bh * T + gt]
          for s in 0 .. t:
            var kd = 0.0'f64
            var ka = 0.0'f64
            var qd = 0.0'f64
            var qa = 0.0'f64
            for dk in 0 ..< Dk:
              let kt = kw[(hk * T + gt) * Dk + dk]
              let ks = kw[(hk * T + segStart + c0 + s) * Dk + dk]
              let qt = qw[(hk * T + gt) * Dk + dk] / sqrt(float64(Dk))
              kd += kt * ks
              ka += abs(kt * ks)
              qd += qt * ks
              qa += abs(qt * ks)
            kdot[t * ChunkC + s] = kd
            kkAbs[t * ChunkC + s] = ka
            pdRef[t * ChunkC + s] = exp(cumulogdecay[t] - cumulogdecay[s])
            qkdot[t * ChunkC + s] = qd
            qkAbs[t * ChunkC + s] = qa
            bAbs[t * ChunkC + s] = pdRef[t * ChunkC + s] * abs(qd)
          for r in 0 ..< Dv:
            # the decayed carry read, its bound from the state's carried error
            var kvN = 0.0'f64
            var kvAbs = 0.0'f64
            var kvErr = 0.0'f64
            for dk in 0 ..< Dk:
              let term = carry[r * Dk + dk] * kw[(hk * T + gt) * Dk + dk]
              kvN += term
              kvAbs += abs(term)
              kvErr += abs(kw[(hk * T + gt) * Dk + dk]) * dS[r * Dk + dk]
            let gRef = pdT * kvN
            let dG = pdT * (float64(Dk) * U32 * kvAbs + kvErr) +
              abs(gRef) * (U32 + decRel)
            let vRef = vw[(bh * T + gt) * Dv + r]
            let base = betaT * (vRef - gRef)
            let dBase = betaT * (dG + U32 * (abs(vRef) + abs(gRef)))
            var uErr = 0.0'f64
            var accAbs = 0.0'f64
            var acc = 0.0'f64
            for s in 0 ..< t:
              let aAbs = pdRef[t * ChunkC + s] * abs(kdot[t * ChunkC + s])
              let dA = aAbs * decRel + pdRef[t * ChunkC + s] *
                float64(Dk) * U32 * kkAbs[t * ChunkC + s]
              uErr += aAbs * du[s * Dv + r] + dA * abs(uVal[s * Dv + r])
              accAbs += aAbs * abs(uVal[s * Dv + r])
              acc += pdRef[t * ChunkC + s] * kdot[t * ChunkC + s] * uVal[s * Dv + r]
            uVal[t * Dv + r] = base - betaT * acc
            du[t * Dv + r] = dBase + betaT * uErr +
              betaT * float64(t) * U32 * accAbs +
              betaT * U32 * (abs(base) + abs(acc))
          for r in 0 ..< Dv:
            # the output, the decayed carry read plus the pair-form sum
              let gt = segStart + c0 + t
              var qkErr = 0.0'f64
              var cqN = 0.0'f64
              var cqAbs = 0.0'f64
              for dk in 0 ..< Dk:
                let qs = qw[(hk * T + gt) * Dk + dk] / sqrt(float64(Dk))
                let term = carry[r * Dk + dk] * qs
                cqN += term
                cqAbs += abs(term)
                qkErr += abs(qs) * dS[r * Dk + dk]
              let crRef = pdT * cqN
              let dCR = pdT * (float64(Dk) * U32 * cqAbs + qkErr) +
                abs(crRef) * (U32 + decRel)
              var yAbsSum = abs(crRef)
              var yErr = dCR
              var yRef = crRef
              for s in 0 .. t:
                yAbsSum += bAbs[t * ChunkC + s] * abs(uVal[s * Dv + r])
                yErr += bAbs[t * ChunkC + s] * du[s * Dv + r] +
                  (bAbs[t * ChunkC + s] * decRel +
                    pdRef[t * ChunkC + s] * float64(Dk) * U32 * qkAbs[t * ChunkC + s]) *
                    abs(uVal[s * Dv + r])
                yRef += pdRef[t * ChunkC + s] * qkdot[t * ChunkC + s] * uVal[s * Dv + r]
              let idx = (bh * T + gt) * Dv + r
              w.yVal[idx] = yRef
              w.dY[idx] = yErr + float64(t + 1) * U32 * yAbsSum + U32 * abs(yRef)
        # the carry update, the chunk-end decay and the u outer products
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
              let ks = kw[(hk * T + segStart + c0 + s) * Dk + dk]
              let wAbs = abs(pdEnd * ks)
              let pTerm = pdEnd * ks * uVal[s * Dv + r]
              sumTerm += pTerm
              sumAbs += abs(pTerm)
              sumErr += wAbs * (du[s * Dv + r] +
                abs(uVal[s * Dv + r]) * (decRel + 3.0 * U32))
            let sNew = decTerm + sumTerm
            dS[idx] = decayEnd * (dS[idx] + abs(sOld) * (U32 + decRel)) +
              U32 * abs(decTerm) + sumErr +
              float64(cLen) * U32 * sumAbs + U32 * abs(sNew)
            carry[idx] = sNew
        c0 += cLen
      segStart += segLen
    for r in 0 ..< Dv:
      for dk in 0 ..< Dk:
        w.sVal[(bh * Dv + r) * Dk + dk] = carry[r * Dk + dk]
        w.dS[(bh * Dv + r) * Dk + dk] = dS[r * Dk + dk]

# ─── Device-side run machinery ────────────────────────────────────────

const
  Dv = 16
  Dk = 32
  TileR = 8
  FloorSub = 2.9802322387695312e-8   # 2⁻²⁵, half the constant fp16 subnormal ulp

type GdnRun* = object
  ## One side's kernel outputs, y as element bits per token and the final state.
  y: seq[uint16]
  state: seq[float32]

proc runGdnScan(
    engine: HwEngine, c: RecCase,
    bhMax, qkRows, T, t0, n: int,
    stateIn: seq[float32],
    stateB: PageBuf[float32], yB: PageBuf[uint16]): GdnRun =
  ## Launches the prefill chunk scan over the token rows [t0, t0 + n), the state loaded
  ## from `stateIn` in place, y and the end state read back.
  let stateElems = bhMax * Dv * Dk
  for i in 0 ..< stateElems: stateB.hostPtr[i] = stateIn[i]
  for i in 0 ..< n * bhMax * Dv: yB.hostPtr[i] = 0
  var kSeg = allocPageBuf[uint16](n * Dk)
  var qSeg = allocPageBuf[uint16](n * Dk)
  var vSeg = allocPageBuf[uint16](n * Dv)
  var betaSeg = allocPageBuf[uint16](n)
  var gSeg = allocPageBuf[float32](n)
  defer:
    freePageBuf(kSeg); freePageBuf(qSeg); freePageBuf(vSeg)
    freePageBuf(betaSeg); freePageBuf(gSeg)
  for i in 0 ..< n * Dk:
    kSeg.hostPtr[i] = c.kBits[t0 * Dk + i]
    qSeg.hostPtr[i] = c.qBits[t0 * Dk + i]
  for i in 0 ..< n * Dv: vSeg.hostPtr[i] = c.vBits[t0 * Dv + i]
  for i in 0 ..< n:
    betaSeg.hostPtr[i] = c.betaBits[t0 + i]
    gSeg.hostPtr[i] = c.gVals[t0 + i]
  var stPA = stateB.pa()
  engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
    ("prop_gdn_prefill_fp16_c32", stPA,
      (yB.pa(), kSeg.pa(), qSeg.pa(), vSeg.pa(), betaSeg.pa(), gSeg.pa(),
        int32(bhMax), int32(qkRows div bhMax), int32(bhMax div qkRows), int32(n)))
  var st = newSeq[float32](stateElems)
  for i in 0 ..< stateElems: st[i] = stateB.hostPtr[i]
  var yy = newSeq[uint16](n * bhMax * Dv)
  for i in 0 ..< n * bhMax * Dv: yy[i] = yB.hostPtr[i]
  GdnRun(y: yy, state: st)

proc runGdnSteps(
    engine: HwEngine, c: RecCase,
    bhMax, qkRows, T: int,
    stateB: PageBuf[float32], yB: PageBuf[uint16]): GdnRun =
  ## Launches T single-token decode steps over the same per-token rows, the state
  ## carried in place across launches, one y row read back per step.
  let stateElems = bhMax * Dv * Dk
  var state = c.state0
  var yAll = newSeq[uint16](T * bhMax * Dv)
  for t in 0 ..< T:
    for i in 0 ..< stateElems: stateB.hostPtr[i] = state[i]
    var kTok = allocPageBuf[uint16](qkRows * Dk)
    var qTok = allocPageBuf[uint16](qkRows * Dk)
    var vTok = allocPageBuf[uint16](bhMax * Dv)
    var betaTok = allocPageBuf[uint16](bhMax)
    var gTok = allocPageBuf[float32](bhMax)
    for i in 0 ..< qkRows * Dk:
      kTok.hostPtr[i] = c.kBits[t * qkRows * Dk + i]
      qTok.hostPtr[i] = c.qBits[t * qkRows * Dk + i]
    for i in 0 ..< bhMax * Dv: vTok.hostPtr[i] = c.vBits[t * bhMax * Dv + i]
    for h in 0 ..< bhMax:
      betaTok.hostPtr[h] = c.betaBits[t * bhMax + h]
      gTok.hostPtr[h] = c.gVals[t * bhMax + h]
    var stPA = stateB.pa()
    engine.run << (grid: (Dv div TileR, bhMax, 1), blk: (32, 1, 1)) >>
      ("prop_gdn_step_fp16", stPA,
        (yB.pa(), kTok.pa(), qTok.pa(), vTok.pa(), betaTok.pa(), gTok.pa(),
          int32(bhMax), int32(qkRows div bhMax), int32(bhMax div qkRows)))
    for i in 0 ..< stateElems: state[i] = stateB.hostPtr[i]
    for i in 0 ..< bhMax * Dv: yAll[t * bhMax * Dv + i] = yB.hostPtr[i]
    freePageBuf(kTok); freePageBuf(qTok); freePageBuf(vTok)
    freePageBuf(betaTok); freePageBuf(gTok)
  GdnRun(y: yAll, state: state)

# ─── Judgment ─────────────────────────────────────────────────────────

proc judgeY(j: var Judge, a, b: GdnWalk, ra, rb: GdnRun, bhMax, T: int, label: string) =
  ## Judges y per element, kernel A against kernel B:
  ## the allowance sums the two sides' error recursions, one fp16 RNE per side
  ## and the subnormal floor.
  for i in 0 ..< bhMax * T * Dv:
    let yA = fp16ToFp32(ra.y[i]).float64
    let yB = fp16ToFp32(rb.y[i]).float64
    let allow = a.dY[i] + b.dY[i] +
      max(abs(a.yVal[i]), abs(b.yVal[i])) * 4.8828125e-4 + FloorSub
    j.judge(&"{label} y elem {i}", yA, yB, allow, ulpStepAt(ulpFp16, max(abs(yA), abs(yB))))

proc judgeState(j: var Judge, a, b: GdnWalk, ra, rb: GdnRun, bhMax: int, label: string) =
  ## Judges the final state per element, the allowance the two sides' carried
  ## error recursions, the state math never rounding.
  let stateElems = bhMax * Dv * Dk
  for i in 0 ..< stateElems:
    j.judge(&"{label} state elem {i}", ra.state[i].float64, rb.state[i].float64,
      a.dS[i] + b.dS[i], 0.0)

# ─── Main, the property combinations ──────────────────────────────────

var suiteCases, suiteLaunches = 0
var suiteWorstUse, suiteWorstUlp, suiteWorstAbs = 0.0'f64
var suiteExact, suiteTotal = 0

proc runCase(engine: HwEngine, seed: uint64, Hv, Hk, B, T: int, splitsA, splitsB: seq[int], decodeSide: int, label: string) =
  ## One property combination, two kernel sides over the same seeded case, each
  ## side's fp64 structure walk, then the per-element judgment:
  ##
  ## - `decodeSide` 1 runs side A on the decode steps, side B on the chunk scan
  ## - `decodeSide` 2 swaps the two, 0 runs both sides on the chunk scan
  let bhMax = B * Hv
  let qkRows = B * Hk
  var rng = initPropRng(seed)
  let c = takeRecCase(rng, kFloat16, qkRows, bhMax, T, Dv, Dk, kda = false)
  var stateB = allocPageBuf[float32](bhMax * Dv * Dk)
  var yB = allocPageBuf[uint16](T * bhMax * Dv)
  defer:
    freePageBuf(stateB); freePageBuf(yB)

  # widened fp64 copies, the walks' exact input surface
  var qw = newSeq[float64](qkRows * T * Dk)
  var kw = newSeq[float64](qkRows * T * Dk)
  var vw = newSeq[float64](bhMax * T * Dv)
  for i in 0 ..< qkRows * T * Dk:
    qw[i] = c.qBits[i].widenTo(kFloat16).float64
    kw[i] = c.kBits[i].widenTo(kFloat16).float64
  for i in 0 ..< bhMax * T * Dv:
    vw[i] = c.vBits[i].widenTo(kFloat16).float64
  var bw = newSeq[float64](bhMax * T)
  for i in 0 ..< bhMax * T: bw[i] = c.betaBits[i].widenTo(kFloat16).float64
  var gw = newSeq[float64](bhMax * T)
  for i in 0 ..< bhMax * T: gw[i] = c.gVals[i].float64
  var s0w = newSeq[float64](bhMax * Dv * Dk)
  for i in 0 ..< bhMax * Dv * Dk: s0w[i] = c.state0[i].float64

  var walkA: GdnWalk
  var walkB: GdnWalk
  gdnWalkCase(walkA, qw, kw, vw, bw, gw, s0w, bhMax, qkRows, Hv, Hk, T, Dv, Dk, 32, splitsA, )
  gdnWalkCase(walkB, qw, kw, vw, bw, gw, s0w, bhMax, qkRows, Hv, Hk, T, Dv, Dk, 32, splitsB, )

  var runA: GdnRun
  var runB: GdnRun
  var launches = 0
  if decodeSide == 1:
    runA = runGdnSteps(engine, c, bhMax, qkRows, T, stateB, yB)
    inc launches, T
    runB = runGdnScan(engine, c, bhMax, qkRows, T, 0, T, c.state0, stateB, yB)
    inc launches
  else:
    var t0 = 0
    var yA = newSeq[uint16](T * bhMax * Dv)
    var stA = c.state0
    for segLen in splitsA:
      let seg = runGdnScan(engine, c, bhMax, qkRows, T, t0, segLen, stA, stateB, yB)
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
      let seg = runGdnScan(engine, c, bhMax, qkRows, T, t0B, segLen, stBs, stateB, yB)
      for bh in 0 ..< bhMax:
        for j in 0 ..< segLen:
          for r in 0 ..< Dv:
            yBs[((bh * T) + t0B + j) * Dv + r] = seg.y[(bh * segLen + j) * Dv + r]
      stBs = seg.state
      t0B += segLen
      inc launches
    runA = GdnRun(y: yA, state: stA)
    runB = GdnRun(y: yBs, state: stBs)
  if decodeSide == 2:
    swap(runA, runB)

  var j = Judge()
  judgeY(j, walkA, walkB, runA, runB, bhMax, T, label)
  judgeState(j, walkA, walkB, runA, runB, bhMax, label)

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
  engine.ingest(GdnPropMsl)
  runCase(engine, 0xC04D0611'u64, 1, 1, 1, 64, @[64], @[32, 32], 0, "split aligned")
  runCase(engine, 0xC04D0612'u64, 1, 1, 1, 64, @[64], @[40, 24], 0, "split ragged")
  runCase(engine, 0xC04D0613'u64, 1, 1, 1, 64, @[64], @[3, 61], 0, "split head ragged")
  runCase(engine, 0xC04D0614'u64, 1, 1, 1, 64, @[64], @[20, 20, 24], 0, "split three-way")
  runCase(engine, 0xC04D0615'u64, 1, 1, 1, 8, @[8], @[3, 5], 0, "split single chunk")
  runCase(engine, 0xC04D0616'u64, 2, 1, 2, 64, @[64], @[32, 32], 0, "split heads aligned")
  runCase(engine, 0xC04D0617'u64, 2, 1, 2, 64, @[64], @[40, 24], 0, "split heads ragged")
  runCase(engine, 0xC04D0618'u64, 1, 1, 1, 8, @[1, 1, 1, 1, 1, 1, 1, 1], @[8], 1,
    "steps T=8 vs prefill")
  var ones33: seq[int]
  for i in 0 ..< 33: ones33.add(1)
  runCase(engine, 0xC04D0619'u64, 1, 1, 1, 33, @[33], ones33, 2, "steps T=33 vs prefill")
  echo &"PROPERTIES GDN VERDICT: cases={suiteCases} launches={suiteLaunches} " &
    &"worst |Δ| {suiteWorstAbs:.3e}, worst usage {suiteWorstUse:.3f}, " &
    &"worst {suiteWorstUlp:.2f} ulp, bit-exact {suiteExact}/{suiteTotal}"

main()
