# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_properties
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests workspace/positron/tests/properties/t_prop_kda.nim
##
## Property suite over the production KDA kernels of the Kimi delta attention rule,
## judging the chunked prefill scan against the single-token decode step:
##
## | property            | what it asserts                           |
## | ------------------- | ----------------------------------------- |
## | split invariance    | whole scan = segmented scan, y and state  |
## | step-count identity | k decode steps = prefill through k tokens |
##
## Under judgment, both entries from `src/kernels/ceramic/attn_ssm/`:
##
## | kernel              | module                                |
## | ------------------- | ------------------------------------- |
## | kdaPrefillChunkScan | gated_delta_net_kda_prefill.nim       |
## | kdaDecodeStepTile   | gated_delta_net_kda_decode_single.nim |
##
## Both sides are the same kernels at different token decompositions, the divergence
## judged pure fp32 rounding, no value reference of any kind:
##
## - the sides share the exact inputs bit for bit, q/k/g/beta f32 and v element bits widened exactly, the state math staying fp32
## - the sides differ in the decay-factored grouping, exp2 of a host prefix sum against a product of per-token factors, one factor per channel
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
##   structure walk, the walk mirroring the kernel's per-channel chunked formulas at the matching decomposition, magnitudes only
## - relDecay carries the per-channel fp32 prefix error, the n terms · u₃₂ · cmax bound
##   of the host prefix, plus 4·u₃₂ for the log2e multiply and the exp2 form, with n the prefix length and cmax the chain's max magnitude per channel
## - dots add Dk terms · u₃₂, the state judgment carrying no store rounding, both exact walks representing one real recurrence,
##   their difference staying fp64 noise below every judged term, the measured worst usage corroborating the model and never setting a bound
##
## An absolute error δ on the log-decay exponent maps to a relative factor error of at most δ,
##   ln2·log2e = 1, the collapse, so the cumulogdecay error enters the allowances
##   directly as a relative term.
##
## Q̃ divides by the host f32 qScale = √Dk on both sides, the identical spelling
## contributing no judged term.
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
## - each side's host cumulogdecay resets at its own segment starts and at every t
##   with (t − segment start) mod ChunkC == 0, the kernel's chunk contract
## - the 4-head case keeps the head-mapping term live under the split, inputs seeded
##   per case, one seeded case per combination, q and k l2-normalized
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

# ─── The fp64 structure walk, one side's magnitudes and error bounds ──

type KdaWalk = object
  ## One side's fp64 walk over the kernel's per-channel chunked formulas at the side's
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

const Log2eF64 = 1.4426950408889634      # log2(e), the exp2 form's constant

proc kdaWalkCase(
    w: var KdaWalk,
    qw, kw, vw, bw: seq[float64], gw: seq[float64], s0w: seq[float64],
    bhMax, qkRows, Hv, Hk, T, Dv, Dk, ChunkC: int,
    segments: seq[int], hostCld: seq[float64]) =
  ## Walks the per-channel chunked KDA formulas (arXiv:2510.26692) in fp64 over
  ## the exact inputs, restarting at every segment boundary, and accumulates
  ## the additive fp32-rounding bounds of the side this decomposition models:
  ##
  ##   per chunk:  cumulogdecay ─→ per token t: dT, G_t read ─→ u_t solve ─→ y_t
  ##               (t in token order)                          └─────────┐
  ##           └──→ S ← dEnd ⊙ carry read + Σ_s (dEnd·invd_s ⊙ k_s) [x] u_s
  ##
  ## - the walker reads the side's own per-channel decay chain from `hostCld`, the fp64 widening of what that side's kernel actually consumes,
  ##   so the prefill side's fp32 chain error vs the exact prefix enters the per-channel allowance as cLen·u₃₂·cmax
  ## - the single-token decode side reads its exact g and keeps the exp2-form terms of the allowance
  ## - a segment boundary restarts the walk with the previous segment's carry
  ##   state and its error bound, the fp32 state handoff is exact
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
    var segStart = 0
    for segLen in segments:
      var c0 = 0
      while c0 < segLen:
        let cLen = min(ChunkC, segLen - c0)
        # per-channel chain error:
        #   the prefill side's fp32 prefix rounding
        # bound per channel, 0 on a single-token side (its g is an exact operand)
        var dTRel = newSeq[float64](Dk)      # the cld error, a relative term
        var dTAbs = newSeq[float64](Dk)      # per-channel max |cld| over the chunk
        for dk in 0 ..< Dk:
          var cmax = abs(hostCld[(hk * T + segStart + c0) * Dk + dk])
          for i in 1 ..< cLen:
            cmax = max(cmax, abs(hostCld[(hk * T + segStart + c0 + i) * Dk + dk]))
          dTAbs[dk] = cmax
          dTRel[dk] = (if cLen == 1: 0.0 else: float64(cLen) * U32 * cmax) + 4.0 * U32
        var uVal = newSeq[float64](cLen * Dv)
        var du = newSeq[float64](cLen * Dv)
        var paADot = newSeq[float64](cLen * cLen)
        var paAAbs = newSeq[float64](cLen * cLen)
        var paBDot = newSeq[float64](cLen * cLen)
        var paBAbs = newSeq[float64](cLen * cLen)
        var paDASum = newSeq[float64](cLen * cLen)
        var paDBSum = newSeq[float64](cLen * cLen)
        for t in 0 ..< cLen:
          let gt = segStart + c0 + t
          let betaT = bw[bh * T + gt]
          # per-channel decay factor and pair-decay magnitudes, shared per state row
          var dT = newSeq[float64](Dk)
          for dk in 0 ..< Dk:
            dT[dk] = pow(2.0, hostCld[(hk * T + gt) * Dk + dk] * Log2eF64)
          for s in 0 .. t:
            var aAbs = 0.0'f64
            var aDot = 0.0'f64
            var bAbs = 0.0'f64
            var bDot = 0.0'f64
            var dASum = 0.0'f64
            var dBSum = 0.0'f64
            for dk in 0 ..< Dk:
              let pd = pow(2.0, (hostCld[(hk * T + gt) * Dk + dk] -
                hostCld[(hk * T + segStart + c0 + s) * Dk + dk]) * Log2eF64)
              let kt = kw[(hk * T + gt) * Dk + dk]
              let ks = kw[(hk * T + segStart + c0 + s) * Dk + dk]
              let qt = qw[(hk * T + gt) * Dk + dk] / sqrt(float64(Dk))
              let pdRel = dTRel[dk] + 4.0 * U32   # the pair's two exp2 forms plus the product
              aDot += pd * kt * ks
              aAbs += abs(pd * kt * ks)
              dASum += abs(pd * kt * ks) * pdRel + abs(pd) * Dk.float64 * U32 * abs(kt * ks)
              bDot += pd * qt * ks
              bAbs += abs(pd * qt * ks)
              dBSum += abs(pd * qt * ks) * pdRel + abs(pd) * Dk.float64 * U32 * abs(qt * ks)
            paADot[t * cLen + s] = aDot
            paAAbs[t * cLen + s] = aAbs
            paBDot[t * cLen + s] = bDot
            paBAbs[t * cLen + s] = bAbs
            paDASum[t * cLen + s] = dASum
            paDBSum[t * cLen + s] = dBSum
          # the solve and output, per state row
          for r in 0 ..< Dv:
            var gN = 0.0'f64
            var gAbs = 0.0'f64
            var gErr = 0.0'f64
            for dk in 0 ..< Dk:
              let term = dT[dk] * kw[(hk * T + gt) * Dk + dk] * carry[r * Dk + dk]
              gN += term
              gAbs += abs(term)
              gErr += abs(term) * dTRel[dk] + dT[dk] * abs(kw[(hk * T + gt) * Dk + dk]) *
                dS[r * Dk + dk]
            let dG = float64(Dk) * U32 * gAbs + gErr
            let vRef = vw[(bh * T + gt) * Dv + r]
            let base = betaT * (vRef - gN)
            let dBase = betaT * (dG + U32 * (abs(vRef) + abs(gN)))
            var uErr = 0.0'f64
            var accAbs = 0.0'f64
            var acc = 0.0'f64
            for s in 0 ..< t:
              uErr += paAAbs[t * cLen + s] * du[s * Dv + r] + paDASum[t * cLen + s] * abs(uVal[s * Dv + r])
              accAbs += paAAbs[t * cLen + s] * abs(uVal[s * Dv + r])
              acc += paADot[t * cLen + s] * uVal[s * Dv + r]
            uVal[t * Dv + r] = base - betaT * acc
            du[t * Dv + r] = dBase + betaT * uErr +
              betaT * float64(t) * U32 * accAbs +
              betaT * U32 * (abs(base) + abs(acc))
            # the output, the decayed carry read plus the pair-form sum
            var hN = 0.0'f64
            var hAbs = 0.0'f64
            var hErr = 0.0'f64
            for dk in 0 ..< Dk:
              let qs = qw[(hk * T + gt) * Dk + dk] / sqrt(float64(Dk))
              let term = dT[dk] * qs * carry[r * Dk + dk]
              hN += term
              hAbs += abs(term)
              hErr += abs(term) * dTRel[dk] + dT[dk] * abs(qs) * dS[r * Dk + dk]
            var yErr = float64(Dk) * U32 * hAbs + hErr
            var yAbsSum = abs(hN)
            var yRef = hN
            for s in 0 .. t:
              yAbsSum += paBAbs[t * cLen + s] * abs(uVal[s * Dv + r])
              yErr += paBAbs[t * cLen + s] * du[s * Dv + r] + paDBSum[t * cLen + s] * abs(uVal[s * Dv + r])
              yRef += paBDot[t * cLen + s] * uVal[s * Dv + r]
            let idx = (bh * T + gt) * Dv + r
            w.yVal[idx] = yRef
            w.dY[idx] = yErr + float64(t + 1) * U32 * yAbsSum + U32 * abs(yRef)
        # the carry update, the chunk-end per-channel decay and the u outer products
        for r in 0 ..< Dv:
          for dk in 0 ..< Dk:
            let idx = r * Dk + dk
            let dEnd = pow(2.0,
              hostCld[(hk * T + segStart + c0 + cLen - 1) * Dk + dk] * Log2eF64)
            let dEndRel = dTRel[dk]
            let sOld = carry[idx]
            let decTerm = dEnd * sOld
            var sumTerm = 0.0'f64
            var sumAbs = 0.0'f64
            var sumErr = 0.0'f64
            for s in 0 ..< cLen:
              let pdEnd = pow(2.0, (hostCld[(hk * T + segStart + c0 + cLen - 1) * Dk + dk] -
                hostCld[(hk * T + segStart + c0 + s) * Dk + dk]) * Log2eF64)
              let ks = kw[(hk * T + segStart + c0 + s) * Dk + dk]
              let wAbs = abs(pdEnd * ks)
              let pTerm = pdEnd * ks * uVal[s * Dv + r]
              sumTerm += pTerm
              sumAbs += abs(pTerm)
              sumErr += wAbs * (du[s * Dv + r] +
                abs(uVal[s * Dv + r]) * (dEndRel + 2.0 * U32 + 3.0 * U32))
            let sNew = decTerm + sumTerm
            dS[idx] = dEnd * (dS[idx] + abs(sOld) * (U32 + dEndRel)) +
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

proc judgeY(j: var Judge, a, b: KdaWalk, ra, rb: KdaRun, bhMax, T: int, label: string) =
  ## Judges y per element, kernel A against kernel B:
  ## the allowance sums the two sides' error recursions, one fp16 RNE per side
  ## and the subnormal floor.
  for i in 0 ..< bhMax * T * Dv:
    let yA = fp16ToFp32(ra.y[i]).float64
    let yB = fp16ToFp32(rb.y[i]).float64
    let allow = a.dY[i] + b.dY[i] +
      max(abs(a.yVal[i]), abs(b.yVal[i])) * 4.8828125e-4 + FloorSub
    j.judge(&"{label} y elem {i}", yA, yB, allow, ulpStepAt(ulpFp16, max(abs(yA), abs(yB))))

proc judgeState(j: var Judge, a, b: KdaWalk, ra, rb: KdaRun, bhMax: int, label: string) =
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
  let c = takeRecCase(rng, kFloat16, qkRows, bhMax, T, Dv, Dk, kda = true)
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
  var gw = newSeq[float64](qkRows * T * Dk)
  for i in 0 ..< qkRows * T * Dk: gw[i] = c.gVals[i].float64
  var s0w = newSeq[float64](bhMax * Dv * Dk)
  for i in 0 ..< bhMax * Dv * Dk: s0w[i] = c.state0[i].float64

  var walkA: KdaWalk
  var walkB: KdaWalk
  var cldA = kdaHostCld(c.gVals, qkRows, T, Dk, 32, splitsA)
  var cldB = kdaHostCld(c.gVals, qkRows, T, Dk, 32, splitsB)
  var cldAW = newSeq[float64](qkRows * T * Dk)
  var cldBW = newSeq[float64](qkRows * T * Dk)
  for i in 0 ..< qkRows * T * Dk:
    cldAW[i] = cldA[i].float64
    cldBW[i] = cldB[i].float64
  kdaWalkCase(walkA, qw, kw, vw, bw, gw, s0w, bhMax, qkRows, Hv, Hk, T, Dv, Dk, 32,
    splitsA, cldAW)
  kdaWalkCase(walkB, qw, kw, vw, bw, gw, s0w, bhMax, qkRows, Hv, Hk, T, Dv, Dk, 32,
    splitsB, cldBW)

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
