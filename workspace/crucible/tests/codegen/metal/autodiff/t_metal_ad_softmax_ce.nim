# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## setupAD on Metal: softmax cross-entropy forward and backward end to end
##
## `engine.setupAD` turns the loop-free softmax-CE pair into an AD handle whose
## launches are exact against the analytic closed forms and the CPU central
## finite differences (h = 1e-3, float64):
##
## - forward: `loss[0] = mean(ln(sum(exp(l-m)) + m - t*l))` over two rows of eight logits, statistics exported
## - convenience backward: `d_logits` matches `(p_j - t_j)*d_loss/2` and the finite differences, `d_target` matches `-d_loss*l_j/2`
## - nonzero incoming `d_rowMax`/`d_rowSumexp` match the augmented-loss finite differences, the statistics' cotangent path is live
## - doubling `d_loss` doubles every `d_logits` exactly, the +-90 fixture keeps both references exact, the primitive backward accumulates
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/autodiff/t_metal_ad_softmax_ce.nim

import std/math
import workspace/crucible

proc softmaxCEPlain(loss, logits, target, rowMax, rowSumexp: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1).} =
  ## One thread scores two rows of eight logits, `m` the per-row max.
  ## `loss[0] = mean(ln(sum(exp(l - m)) + m - t*l))`, statistics export into `rowMax`/`rowSumexp`.
  # Row 0: the max as a balanced tree of single-assignment ternaries
  # (max() is a refused call, loop-carried accumulation refuses too).
  let m01 = if logits[0] > logits[1]: logits[0] else: logits[1]
  let m23 = if logits[2] > logits[3]: logits[2] else: logits[3]
  let m45 = if logits[4] > logits[5]: logits[4] else: logits[5]
  let m67 = if logits[6] > logits[7]: logits[6] else: logits[7]
  let m0123 = if m01 > m23: m01 else: m23
  let m4567 = if m45 > m67: m45 else: m67
  let m0 = if m0123 > m4567: m0123 else: m4567
  # Row 0: sumexp as an unrolled sum of eight shifted exponentials.
  let e0 = exp(logits[0] - m0)
  let e1 = exp(logits[1] - m0)
  let e2 = exp(logits[2] - m0)
  let e3 = exp(logits[3] - m0)
  let e4 = exp(logits[4] - m0)
  let e5 = exp(logits[5] - m0)
  let e6 = exp(logits[6] - m0)
  let e7 = exp(logits[7] - m0)
  let s0 = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7
  rowMax[0] = m0
  rowSumexp[0] = s0
  # Row 0: the loss term, an unrolled dot against the one-hot target.
  let lt0 = target[0]*logits[0] + target[1]*logits[1] + target[2]*logits[2] +
            target[3]*logits[3] + target[4]*logits[4] + target[5]*logits[5] +
            target[6]*logits[6] + target[7]*logits[7]
  let ce0 = ln(s0) + m0 - lt0
  # Row 1, same shape over columns 8..15.
  let n01 = if logits[8] > logits[9]: logits[8] else: logits[9]
  let n23 = if logits[10] > logits[11]: logits[10] else: logits[11]
  let n45 = if logits[12] > logits[13]: logits[12] else: logits[13]
  let n67 = if logits[14] > logits[15]: logits[14] else: logits[15]
  let n0123 = if n01 > n23: n01 else: n23
  let n4567 = if n45 > n67: n45 else: n67
  let n0 = if n0123 > n4567: n0123 else: n4567
  let f0 = exp(logits[8] - n0)
  let f1 = exp(logits[9] - n0)
  let f2 = exp(logits[10] - n0)
  let f3 = exp(logits[11] - n0)
  let f4 = exp(logits[12] - n0)
  let f5 = exp(logits[13] - n0)
  let f6 = exp(logits[14] - n0)
  let f7 = exp(logits[15] - n0)
  let s1 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7
  rowMax[1] = n0
  rowSumexp[1] = s1
  let lt1 = target[8]*logits[8] + target[9]*logits[9] + target[10]*logits[10] +
            target[11]*logits[11] + target[12]*logits[12] +
            target[13]*logits[13] + target[14]*logits[14] +
            target[15]*logits[15]
  let ce1 = ln(s1) + n0 - lt1
  loss[0] = (ce0 + ce1) / 2.0'f32

const softmaxCEPair = bkMetal.bwd_diff(softmaxCEPlain)

type SceBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience overload reads that length for its grad allocation.
  p: DevMem
  n: int

const
  SceRows = 2
  SceCols = 8
  SceN = SceRows * SceCols
  DLoss = 0.75'f32
  H = 1.0e-3'f32

# ── Fixtures ──────────────────────────────────────────────────────────────

proc moderateLogits: array[SceN, float32] =
  ## Well-separated moderate magnitudes, per-row argmax distinct from the one-hot column.
  ## Row 0: argmax 7, target 5. Row 1: argmax 3, target 2.
  [0.5'f32, -1.25'f32, 2.0'f32, -0.75'f32, 1.5'f32, -2.0'f32, 0.25'f32, 3.0'f32,
   -0.5'f32, 1.0'f32, -1.5'f32, 2.5'f32, 0.0'f32, -2.5'f32, 1.75'f32, -0.25'f32]

proc extremeLogits: array[SceN, float32] =
  ## Logits spanning +-90, shifted exponentials underflow toward zero while the closed form stays exact.
  ## Row 0: argmax 0 (90), target 2 (12). Row 1: argmax 5 (89), target 5 (89), the one-hot column is the argmax.
  [90.0'f32, -90.0'f32, 12.0'f32, -3.5'f32, 45.0'f32, -60.0'f32, 7.0'f32, 33.0'f32,
   -88.0'f32, 15.0'f32, -40.0'f32, 70.0'f32, -12.0'f32, 89.0'f32, -77.0'f32, 3.0'f32]

proc targetHost(hot0, hot1: int): array[SceN, float32] =
  ## One-hot float targets, row 0 hot on `hot0`, row 1 hot on `hot1`.
  result[hot0] = 1.0'f32
  result[SceCols + hot1] = 1.0'f32

# ── CPU references, float64 ───────────────────────────────────────────────

proc rowMax64(l: openArray[float32], r: int): float64 =
  for j in 0 ..< SceCols:
    result = max(result, float64(l[r * SceCols + j]))

proc rowSumexp64(l: openArray[float32], r: int): float64 =
  let m = rowMax64(l, r)
  for j in 0 ..< SceCols:
    result += exp(float64(l[r * SceCols + j]) - m)

proc shiftedExp64(l: openArray[float32], r, j: int): float64 =
  ## exp(l_j - m) of row `r`, the value the kernel accumulates into rowSumexp.
  exp(float64(l[r * SceCols + j]) - rowMax64(l, r))

proc argmaxColumn(l: openArray[float32], r: int): int =
  ## Column of the row's maximum, the ternary tree's winner (no ties in the fixtures).
  result = 0
  for j in 1 ..< SceCols:
    if l[r * SceCols + j] > l[r * SceCols + result]:
      result = j

proc loss64(l, t: openArray[float32]): float64 =
  ## Kernel loss: mean over the rows of `ln(sumexp) + max - t.l`.
  for r in 0 ..< SceRows:
    result += ln(rowSumexp64(l, r)) + rowMax64(l, r)
    for j in 0 ..< SceCols:
      result -= float64(t[r * SceCols + j]) * float64(l[r * SceCols + j])
  result /= float64(SceRows)

proc statsTerm64(l: openArray[float32], dRM, dRSE: openArray[float32]): float64 =
  ## Exported statistics' direct terms `sum d_rowMax*rowMax + d_rowSumexp*rowSumexp`.
  for r in 0 ..< SceRows:
    result += float64(dRM[r]) * rowMax64(l, r)
    result += float64(dRSE[r]) * rowSumexp64(l, r)

proc analyticDLogits(l, t: openArray[float32], dLoss: float64, r, j: int): float64 =
  ## dL/dl_j through the loss only. The closed form of the total derivative
  ## (the max path is already folded in): `(p_j - t_j) * d_loss / rows`.
  let p = shiftedExp64(l, r, j) / rowSumexp64(l, r)
  result = (p - float64(t[r * SceCols + j])) * dLoss / float64(SceRows)

proc analyticDLogitsStats(l, t: openArray[float32], dLoss: float64, dRM, dRSE: openArray[float32], r, j: int): float64 =
  ## dL/dl_j of the augmented loss: the loss term plus the exports' paths.
  ## `d(sumexp)/dl_j = exp(l_j - m) - sumexp` on the argmax column, since
  ## `d m/dl_j = 1` there shifts every shifted exponential.
  result = analyticDLogits(l, t, dLoss, r, j)
  result += float64(dRSE[r]) * shiftedExp64(l, r, j)
  if j == argmaxColumn(l, r):
    result += float64(dRM[r]) - float64(dRSE[r]) * rowSumexp64(l, r)

proc analyticDTarget(l: openArray[float32], dLoss: float64, r, j: int): float64 =
  ## dL/dt_j: the target enters only through the dot with the logits.
  -float64(l[r * SceCols + j]) * dLoss / float64(SceRows)

proc relClose(got, want, rtol, atol: float64): bool =
  abs(got - want) <= atol + rtol * abs(want)

proc centralFd64(l: array[SceN, float32], t: array[SceN, float32], dLoss: float32, dRM, dRSE: array[SceRows, float32], i: int): float64 =
  ## Central finite difference of the differentiated loss against one logit.
  ## The loss term scales by the incoming `d_loss`, the statistics' direct terms carry their own cotangents.
  var lp = l
  var lm = l
  lp[i] = lp[i] + H
  lm[i] = lm[i] - H
  let lossFd = (loss64(lp, t) - loss64(lm, t)) / (2.0 * float64(H))
  let statFd = (statsTerm64(lp, dRM, dRSE) - statsTerm64(lm, dRM, dRSE)) /
               (2.0 * float64(H))
  result = float64(dLoss) * lossFd + statFd

# ── Test ──────────────────────────────────────────────────────────────────

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  let ad = engine.setupAD(softmaxCEPair)

  let lossDev = engine.allocDevice(float32, 1)
  let logitsDev = engine.allocDevice(float32, SceN)
  let targetDev = engine.allocDevice(float32, SceN)
  let rowMaxDev = engine.allocDevice(float32, SceRows)
  let rowSumexpDev = engine.allocDevice(float32, SceRows)
  let dLossDev = engine.allocDevice(float32, 1)
  let dRowMaxDev = engine.allocDevice(float32, SceRows)
  let dRowSumexpDev = engine.allocDevice(float32, SceRows)

  var zerosN: array[SceN, float32]   # zero-initialized
  var zerosR: array[SceRows, float32]

  proc loadFixture(l: array[SceN, float32], t: array[SceN, float32], dLoss: float32) =
    ## Uploads one fixture and zeroes the incoming stats cotangents.
    engine.copyToDeviceFromHost(logitsDev, addr l[0], SceN)
    engine.copyToDeviceFromHost(targetDev, addr t[0], SceN)
    engine.copyToDeviceFromHost(dLossDev, addr dLoss, 1)
    engine.copyToDeviceFromHost(dRowMaxDev, addr zerosR[0], SceRows)
    engine.copyToDeviceFromHost(dRowSumexpDev, addr zerosR[0], SceRows)

  proc forwardLoss: tuple[loss, rm0, rm1, se0, se1: float32] =
    ad.forward(lossDev, logitsDev, targetDev, rowMaxDev, rowSumexpDev)
    var lossGot: array[1, float32]
    var rmGot: array[SceRows, float32]
    var seGot: array[SceRows, float32]
    engine.copyToHostFromDevice(addr lossGot[0], lossDev, 1)
    engine.copyToHostFromDevice(addr rmGot[0], rowMaxDev, SceRows)
    engine.copyToHostFromDevice(addr seGot[0], rowSumexpDev, SceRows)
    (lossGot[0], rmGot[0], rmGot[1], seGot[0], seGot[1])

  proc convenienceDLogits(dRM, dRSE: array[SceRows, float32]): array[SceN, float32] =
    ## Convenience backward with the given incoming stats cotangents,
    ## returns the freshly allocated d_logits gradient.
    engine.copyToDeviceFromHost(dRowMaxDev, addr dRM[0], SceRows)
    engine.copyToDeviceFromHost(dRowSumexpDev, addr dRSE[0], SceRows)
    let grads = ad.backward(dLossDev, dRowMaxDev, dRowSumexpDev,
      lossDev,
      SceBuf[typeof(logitsDev)](p: logitsDev, n: SceN),
      SceBuf[typeof(targetDev)](p: targetDev, n: SceN),
      rowMaxDev, rowSumexpDev)
    engine.copyToHostFromDevice(addr result[0], grads.d_logits, SceN)

  proc convenienceDTarget: array[SceN, float32] =
    let grads = ad.backward(dLossDev, dRowMaxDev, dRowSumexpDev,
      lossDev,
      SceBuf[typeof(logitsDev)](p: logitsDev, n: SceN),
      SceBuf[typeof(targetDev)](p: targetDev, n: SceN),
      rowMaxDev, rowSumexpDev)
    engine.copyToHostFromDevice(addr result[0], grads.d_target, SceN)

  proc checkDLogits(l: array[SceN, float32], t: array[SceN, float32],
                    dLoss: float32, got: array[SceN, float32],
                    dRM, dRSE: array[SceRows, float32]) =
    ## d_logits against the analytic cotangent at rtol 1e-5 and the central
    ## finite difference of the differentiated loss at rtol 1e-3.
    for r in 0 ..< SceRows:
      for j in 0 ..< SceCols:
        let i = r * SceCols + j
        let wantA = if dRM == zerosR and dRSE == zerosR:
          analyticDLogits(l, t, float64(dLoss), r, j)
        else:
          analyticDLogitsStats(l, t, float64(dLoss), dRM, dRSE, r, j)
        doAssert relClose(float64(got[i]), wantA, 1.0e-5, 1.0e-7),
          "d_logits analytic: i " & $i & " got " & $got[i] & " want " & $wantA
        let wantFd = centralFd64(l, t, dLoss, dRM, dRSE, i)
        doAssert relClose(float64(got[i]), wantFd, 1.0e-3, 1.0e-5),
          "d_logits finite diff: i " & $i & " got " & $got[i] & " want " & $wantFd

  block:
    # Moderate fixture: forward against the closed form.
    let l = moderateLogits()
    let t = targetHost(5, 2)
    loadFixture(l, t, DLoss)
    let f = forwardLoss()
    doAssert relClose(float64(f.loss), loss64(l, t), 1.0e-6, 1.0e-7),
      "loss: got " & $f.loss & " want " & $loss64(l, t)
    for r in 0 ..< SceRows:
      let rm = if r == 0: f.rm0 else: f.rm1
      let se = if r == 0: f.se0 else: f.se1
      doAssert relClose(float64(rm), rowMax64(l, r), 1.0e-6, 1.0e-7),
        "rowMax: row " & $r & " got " & $rm & " want " & $rowMax64(l, r)
      doAssert relClose(float64(se), rowSumexp64(l, r), 1.0e-6, 1.0e-7),
        "rowSumexp: row " & $r & " got " & $se & " want " & $rowSumexp64(l, r)
    echo "[OK] forward: loss, rowMax, rowSumexp match the closed form"

    # Convenience backward, stats cotangents zeroed: d_logits vs analytic + FD, d_target vs analytic.
    let dLogits = convenienceDLogits(zerosR, zerosR)
    checkDLogits(l, t, DLoss, dLogits, zerosR, zerosR)
    let dTarget = convenienceDTarget()
    for r in 0 ..< SceRows:
      for j in 0 ..< SceCols:
        let i = r * SceCols + j
        let wantT = analyticDTarget(l, float64(DLoss), r, j)
        doAssert relClose(float64(dTarget[i]), wantT, 1.0e-5, 1.0e-7),
          "d_target: i " & $i & " got " & $dTarget[i] & " want " & $wantT
    echo "[OK] convenience backward: d_logits and d_target match analytic and finite differences"

    # Broadcast exactness: doubling d_loss doubles d_logits exactly.
    loadFixture(l, t, 2.0'f32 * DLoss)
    let dLogits2x = convenienceDLogits(zerosR, zerosR)
    for i in 0 ..< SceN:
      doAssert dLogits2x[i] == 2.0'f32 * dLogits[i],
        "broadcast: i " & $i & " got " & $dLogits2x[i] & " want " & $(2.0'f32 * dLogits[i])
    loadFixture(l, t, DLoss)
    echo "[OK] broadcast: d_loss doubled, d_logits exactly doubled"

    # Stats cotangent path: nonzero incoming d_rowMax/d_rowSumexp vs the augmented-loss finite differences.
    var dRM: array[SceRows, float32]
    var dRSE: array[SceRows, float32]
    dRM = [0.4'f32, -0.3'f32]
    dRSE = [0.2'f32, 0.1'f32]
    let dLogitsStats = convenienceDLogits(dRM, dRSE)
    checkDLogits(l, t, DLoss, dLogitsStats, dRM, dRSE)
    engine.copyToDeviceFromHost(dRowMaxDev, addr zerosR[0], SceRows)
    engine.copyToDeviceFromHost(dRowSumexpDev, addr zerosR[0], SceRows)
    echo "[OK] stats cotangent path: nonzero d_rowMax/d_rowSumexp match the augmented-loss finite differences"

    # Large-magnitude fixture: forward and backward stay exact at +-90.
    let le = extremeLogits()
    let te = targetHost(2, 5)
    loadFixture(le, te, DLoss)
    let fe = forwardLoss()
    doAssert relClose(float64(fe.loss), loss64(le, te), 1.0e-6, 1.0e-7),
      "extreme loss: got " & $fe.loss & " want " & $loss64(le, te)
    for r in 0 ..< SceRows:
      let rm = if r == 0: fe.rm0 else: fe.rm1
      let se = if r == 0: fe.se0 else: fe.se1
      doAssert relClose(float64(rm), rowMax64(le, r), 1.0e-6, 1.0e-7),
        "extreme rowMax: row " & $r & " got " & $rm & " want " & $rowMax64(le, r)
      doAssert relClose(float64(se), rowSumexp64(le, r), 1.0e-6, 1.0e-7),
        "extreme rowSumexp: row " & $r & " got " & $se & " want " & $rowSumexp64(le, r)
    let dLogitsExtreme = convenienceDLogits(zerosR, zerosR)
    checkDLogits(le, te, DLoss, dLogitsExtreme, zerosR, zerosR)
    echo "[OK] extreme fixture (+-90 logits): forward and backward match closed form and finite differences"

    # Primitive backward: pre-zeroed caller buffers, two calls, exactly 2x.
    loadFixture(l, t, DLoss)
    let dLogitsDev = engine.allocDevice(float32, SceN)
    let dTargetDev = engine.allocDevice(float32, SceN)
    engine.copyToDeviceFromHost(dLogitsDev, addr zerosN[0], SceN)
    engine.copyToDeviceFromHost(dTargetDev, addr zerosN[0], SceN)
    ad.backward(dLossDev, dRowMaxDev, dRowSumexpDev,
      lossDev, logitsDev, targetDev, rowMaxDev, rowSumexpDev,
      dLogitsDev, dTargetDev)
    var primOnce: array[SceN, float32]
    engine.copyToHostFromDevice(addr primOnce[0], dLogitsDev, SceN)
    ad.backward(dLossDev, dRowMaxDev, dRowSumexpDev,
      lossDev, logitsDev, targetDev, rowMaxDev, rowSumexpDev,
      dLogitsDev, dTargetDev)
    var primTwice: array[SceN, float32]
    engine.copyToHostFromDevice(addr primTwice[0], dLogitsDev, SceN)
    for i in 0 ..< SceN:
      doAssert primOnce[i] == dLogits[i],
        "primitive single call: i " & $i & " got " & $primOnce[i] &
        " want " & $dLogits[i]
      doAssert primTwice[i] == 2.0'f32 * dLogits[i],
        "primitive accumulate: i " & $i & " got " & $primTwice[i] &
        " want " & $(2.0'f32 * dLogits[i])
    echo "[OK] primitive backward: accumulates, two calls exactly twice one"

when isMainModule:
  runTest()
