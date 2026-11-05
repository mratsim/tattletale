# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Microgpt forward op set: gather, layernorm, gelu, softmax, cross entropy, where, sgd, adam.
##
## Test that every op matches an independent f64 host reference on CPU and Metal.
## Test that the causal-mask composition (where + softmax) matches the f64 reference.
## Test that adam keeps m and v across steps and moves the parameters by the f64 recurrence.
## Test that a mixed op record drains in ONE Metal submission.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_ops.nim

import std/math

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery,
  workspace/tensorcery/src/backends/cpu,
  workspace/tensorcery/src/backends/metal

func frand(i: int): float32 =
  ## Deterministic host input in [0, 1) from the golden-ratio low-discrepancy sequence.
  let x = float64(i) * 0.6180339887498949
  float32(x mod 1.0)

func frandSigned(i: int): float32 =
  ## Deterministic host input in [-1, 1) from the golden-ratio low-discrepancy sequence.
  2.0 * frand(i) - 1.0

proc checkBand(actual: seq[float32], expected: seq[float64], label: string) =
  ## Elementwise band check of device f32 output against the f64 host reference.
  doAssert actual.len == expected.len, label & ": length mismatch"
  for i in 0 ..< expected.len:
    let band = 1e-5 * abs(expected[i]) + 1e-5
    doAssert abs(float64(actual[i]) - expected[i]) <= band,
        label & ": element " & $i & " got " & $actual[i] & ", expected " & $expected[i]

proc refGather(w: seq[float64], v, e: int, idx: seq[int]): seq[float64] =
  ## Host reference of the embedding lookup, one embedding vector per index.
  for row in 0 ..< idx.len:
    for col in 0 ..< e:
      result.add w[idx[row] * e + col]

proc refLayernorm(x: seq[float64], rows, d: int, w, b: seq[float64], eps: float64): seq[float64] =
  ## Host reference of the last-axis layernorm.
  for r in 0 ..< rows:
    var mean = 0.0
    for c in 0 ..< d:
      mean += x[r * d + c]
    mean /= float64(d)
    var vr = 0.0
    for c in 0 ..< d:
      let dv = x[r * d + c] - mean
      vr += dv * dv
    vr /= float64(d)
    let scale = 1.0 / sqrt(vr + eps)
    for c in 0 ..< d:
      result.add (x[r * d + c] - mean) * scale * w[c] + b[c]

proc refGelu(x: seq[float64]): seq[float64] =
  ## Host reference of the tanh-approximate gelu.
  for v in x:
    let u = 0.7978845608028654 * (v + 0.044715 * v * v * v)
    result.add 0.5 * v * (1.0 + tanh(u))

proc refSoftmax(x: seq[float64], rows, d: int): seq[float64] =
  ## Host reference of the stable last-axis softmax.
  for r in 0 ..< rows:
    var m = x[r * d]
    for c in 1 ..< d:
      m = max(m, x[r * d + c])
    var s = 0.0
    var e: seq[float64]
    for c in 0 ..< d:
      let ex = exp(x[r * d + c] - m)
      e.add ex
      s += ex
    for c in 0 ..< d:
      result.add e[c] / s

proc refCrossEntropy(logits: seq[float64], rows, v: int, targets: seq[int]): seq[float64] =
  ## Host reference of the cross entropy, logsumexp + target gather, one loss per logit set.
  for r in 0 ..< rows:
    var m = logits[r * v]
    for c in 1 ..< v:
      m = max(m, logits[r * v + c])
    var s = 0.0
    for c in 0 ..< v:
      s += exp(logits[r * v + c] - m)
    let lse = m + ln(s)
    result.add lse - logits[r * v + targets[r]]

proc refAdam(p, g, m, v: seq[float64], lr, beta1, beta2, eps: float64, step: int): seq[float64] =
  ## Host reference of the adam recurrence with the folded bias corrections.
  let bc1 = 1.0 / (1.0 - pow(beta1, float64(step)))
  let bc2 = 1.0 / (1.0 - pow(beta2, float64(step)))
  for i in 0 ..< p.len:
    let mi = beta1 * m[i] + (1.0 - beta1) * g[i]
    let vi = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]
    result.add p[i] - lr * bc1 * mi / (sqrt(bc2 * vi) + eps)

proc runLeg(Hw: static BackendKind) =
  ## Forward op checks on one backend.
  let v, e = 3
  var wdata: seq[float32]
  for i in 0 ..< v * e:
    wdata.add frandSigned(i + 3)

  block gather:
    let ctx = Ctx()
    let w = toTensor(ctx, wdata, Hw, v, e)
    var idxData: seq[int32]
    idxData.add 2'i32
    idxData.add 0'i32
    idxData.add 2'i32
    let idx = toTensor(ctx, idxData, Hw, 3)
    let outT = gather(w, idx)
    ctx.eval()
    var wref: seq[float64]
    for x in wdata:
      wref.add float64(x)
    checkBand(toSeq[float32](outT), refGather(wref, v, e, @[2, 0, 2]), $Hw & " gather")

  block layernorm:
    let ctx = Ctx()
    let rows, d = 4
    var xdata, ws, bs: seq[float32]
    for i in 0 ..< rows * d:
      xdata.add frandSigned(i + 11)
    for i in 0 ..< d:
      ws.add frand(i + 51)
      bs.add frand(i + 57)
    let x = toTensor(ctx, xdata, Hw, rows, d)
    let w = toTensor(ctx, ws, Hw, d)
    let b = toTensor(ctx, bs, Hw, d)
    let outT = layernorm(x, w, b, 1e-5)
    ctx.eval()
    var x64, w64, b64: seq[float64]
    for x in xdata:
      x64.add float64(x)
    for x in ws:
      w64.add float64(x)
    for x in bs:
      b64.add float64(x)
    checkBand(toSeq[float32](outT),
        refLayernorm(x64, rows, d, w64, b64, 1e-5), $Hw & " layernorm")

  block gelu:
    let ctx = Ctx()
    let n = 8
    var xdata: seq[float32]
    for i in 0 ..< n:
      xdata.add frandSigned(i + 23)
    let outT = gelu(toTensor(ctx, xdata, Hw, n))
    ctx.eval()
    var x64: seq[float64]
    for x in xdata:
      x64.add float64(x)
    checkBand(toSeq[float32](outT), refGelu(x64), $Hw & " gelu")

  block softmaxCausalMask:
    # microgpt composition: scores masked by where, then softmax
    let ctx = Ctx()
    let s0, s1 = 4
    var scores, mask, negInf: seq[float32]
    for i in 0 ..< s0 * s1:
      scores.add frandSigned(i + 31)
      let keep = (i mod s1) <= (i div s1)
      mask.add if keep: 1.0'f32 else: 0.0'f32
      negInf.add if keep: 0.0'f32 else: -Inf.float32
    let sc = toTensor(ctx, scores, Hw, s0, s1)
    let m = toTensor(ctx, mask, Hw, s0, s1)
    let ni = toTensor(ctx, negInf, Hw, s0, s1)
    let masked = where(m, sc, ni)
    let outT = softmax(masked)
    ctx.eval()
    var refIn: seq[float64]
    for i in 0 ..< s0 * s1:
      refIn.add if mask[i] != 0.0'f32: float64(scores[i]) else: -Inf.float64
    checkBand(toSeq[float32](outT), refSoftmax(refIn, s0, s1), $Hw & " softmax+mask")

  block crossEntropy:
    let ctx = Ctx()
    let rows, v = 3
    var logits: seq[float32]
    for i in 0 ..< rows * v:
      logits.add frandSigned(i + 43)
    let lg = toTensor(ctx, logits, Hw, rows, v)
    var tgData: seq[int32]
    tgData.add 1'i32
    tgData.add 2'i32
    tgData.add 0'i32
    let tg = toTensor(ctx, tgData, Hw, rows)
    let loss = crossEntropy(lg, tg)
    ctx.eval()
    var lg64: seq[float64]
    for x in logits:
      lg64.add float64(x)
    checkBand(toSeq[float32](loss),
        refCrossEntropy(lg64, rows, v, @[1, 2, 0]), $Hw & " cross entropy")

  block sgdTwoSteps:
    let ctx = Ctx()
    let n = 4
    var pdata, g1data, g2data: seq[float32]
    for i in 0 ..< n:
      pdata.add frand(i + 61)
      g1data.add frandSigned(i + 71)
      g2data.add frandSigned(i + 81)
    let p = toTensor(ctx, pdata, Hw, n)
    let lr = 0.1
    sgdUpdate(p, toTensor(ctx, g1data, Hw, n), lr)
    ctx.eval()
    var expected: seq[float64]
    for i in 0 ..< n:
      expected.add float64(pdata[i]) - lr * float64(g1data[i])
    checkBand(toSeq[float32](p), expected, $Hw & " sgd step 1")
    sgdUpdate(p, toTensor(ctx, g2data, Hw, n), lr)
    ctx.eval()
    for i in 0 ..< n:
      expected[i] -= lr * float64(g2data[i])
    checkBand(toSeq[float32](p), expected, $Hw & " sgd step 2")

  block adamTwoSteps:
    let ctx = Ctx()
    let n = 4
    var pdata, g1data, g2data: seq[float32]
    for i in 0 ..< n:
      pdata.add frand(i + 91)
      g1data.add frandSigned(i + 101)
      g2data.add frandSigned(i + 111)
    let zero = newSeq[float32](n)
    let p = toTensor(ctx, pdata, Hw, n)
    let m = toTensor(ctx, zero, Hw, n)
    let v = toTensor(ctx, zero, Hw, n)
    const lr = 0.05
    const beta1 = 0.9
    const beta2 = 0.999
    const eps = 1e-8

    var m64 = newSeq[float64](n)
    var v64 = newSeq[float64](n)
    var p64: seq[float64]
    for x in pdata:
      p64.add float64(x)

    for step in 1 .. 2:
      let gdata = if step == 1: g1data else: g2data
      adamUpdate(p, toTensor(ctx, gdata, Hw, n), m, v, lr, beta1, beta2, eps, step)
      ctx.eval()
      var g64: seq[float64]
      for x in gdata:
        g64.add float64(x)
      # refAdam applies the moment recurrence itself, so it reads the pre-step moments
      let expectedP = refAdam(p64, g64, m64, v64, lr, beta1, beta2, eps, step)
      for i in 0 ..< n:
        let g = float64(gdata[i])
        m64[i] = beta1 * m64[i] + (1.0 - beta1) * g
        v64[i] = beta2 * v64[i] + (1.0 - beta2) * g * g
      checkBand(toSeq[float32](p), expectedP, $Hw & " adam step " & $step)
      # m and v persist and carry the recurrence
      checkBand(toSeq[float32](m), m64, $Hw & " adam m step " & $step)
      checkBand(toSeq[float32](v), v64, $Hw & " adam v step " & $step)
      p64 = expectedP

proc main() =
  runLeg(bkCpu)
  runLeg(bkMetal)

  block oneSubmissionMixedRecord:
    # gather + layernorm + gelu in one record drains as ONE Metal submission
    let ctx = Ctx()
    let v, e = 3
    var wdata: seq[float32]
    for i in 0 ..< v * e:
      wdata.add frandSigned(i + 3)
    let w = toTensor(ctx, wdata, bkMetal, v, e)
    var idx2Data: seq[int32]
    idx2Data.add 1'i32
    idx2Data.add 2'i32
    let idx = toTensor(ctx, idx2Data, bkMetal, 2)
    let emb = gather(w, idx)
    var ws, bs: seq[float32]
    for i in 0 ..< e:
      ws.add frand(i + 51)
      bs.add frand(i + 57)
    let outT = gelu(layernorm(emb, toTensor(ctx, ws, bkMetal, e), toTensor(ctx, bs, bkMetal, e), 1e-5))
    doAssert ctx.getPendingCount() == 3, "three ops wait in one record"
    resetSubmissionCount()
    ctx.eval()
    doAssert submissionCount == 1,
        "a mixed op record must drain as ONE submission, got " & $submissionCount
    doAssert toSeq[float32](outT).len == 2 * e, "the chain computed both rows"

  stdout.writeLine "t_ops: OK"

main()
