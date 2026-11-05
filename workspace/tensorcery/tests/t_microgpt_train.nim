# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Microgpt model end to end: forward vs host reference, training, one submission per step.
##
## Test that the microgpt forward matches an independent f64 host implementation
## of the same math on CPU and Metal.
## Test that training on a cyclic token pattern lowers the mean loss on CPU and Metal.
## Test that a Metal training step drains forward, backward and adam in ONE submission.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_microgpt_train.nim

import std/[math, random, sequtils]

import workspace/data_structures/src/small_seqs
import workspace/tensorcery
import workspace/tensorcery/src/backends/cpu
import workspace/tensorcery/src/backends/metal
import workspace/tensorcery/src/tape
import workspace/tensorcery/examples/microgpt_model

const
  B = 2
  T = 4
  D = 8
  H = 2
  HD = D div H
  V = 7
  L = 1
  Eps = 1e-5
  TrainSteps = 12

type HostParams = object
  wte, wpe, lmHead: seq[float64]
  wq, wk, wv, wo, fc1, fc2: seq[float64]
  ln1w, ln1b, ln2w, ln2b, lnfw, lnfb: seq[float64]

proc layernormRef(x: seq[float64], n, d: int, w, b: seq[float64]): seq[float64] =
  result = newSeq[float64](x.len)
  for r in 0 ..< n:
    var mean = 0.0
    for c in 0 ..< d:
      mean += x[r*d + c]
    mean /= float64(d)
    var vr = 0.0
    for c in 0 ..< d:
      let dv = x[r*d + c] - mean
      vr += dv * dv
    vr /= float64(d)
    let scale = 1.0 / sqrt(vr + Eps)
    for c in 0 ..< d:
      result[r*d + c] = (x[r*d + c] - mean) * scale * w[c] + b[c]

proc linearRef(x: seq[float64], rows, inD, outD: int, w: seq[float64]): seq[float64] =
  ## y[r, o] = sum_i x[r, i] * w[o, i]
  result = newSeq[float64](rows * outD)
  for r in 0 ..< rows:
    for o in 0 ..< outD:
      var s = 0.0
      for i in 0 ..< inD:
        s += x[r*inD + i] * w[o*inD + i]
      result[r*outD + o] = s

proc geluRef(x: float64): float64 =
  let inner = sqrt(2.0 / PI) * (x + 0.044715 * x * x * x)
  0.5 * x * (1.0 + tanh(inner))

proc hostLoss(p: HostParams, tokens, targets: seq[int32]): seq[float64] =
  ## Per-position losses of the one-layer model, f64 host math.
  var h = newSeq[float64](B * T * D)
  for b in 0 ..< B:
    for t in 0 ..< T:
      for d in 0 ..< D:
        h[b*T*D + t*D + d] = p.wte[int(tokens[b*T + t])*D + d] + p.wpe[t*D + d]
  let ln1 = layernormRef(h, B*T, D, p.ln1w, p.ln1b)
  let q = linearRef(ln1, B*T, D, D, p.wq)
  let k = linearRef(ln1, B*T, D, D, p.wk)
  let v = linearRef(ln1, B*T, D, D, p.wv)
  var attnOut = newSeq[float64](B * T * D)
  for b in 0 ..< B:
    for head in 0 ..< H:
      for i in 0 ..< T:
        var scores: array[8, float64]
        var maxScore = -1e30
        for j in 0 .. i:
          var s = 0.0
          for d in 0 ..< HD:
            s += q[b*T*D + i*D + head*HD + d] * k[b*T*D + j*D + head*HD + d]
          scores[j] = s / sqrt(float64(HD))
          if scores[j] > maxScore:
            maxScore = scores[j]
        var denom = 0.0
        for j in 0 .. i:
          scores[j] = exp(scores[j] - maxScore)
          denom += scores[j]
        for d in 0 ..< HD:
          var acc = 0.0
          for j in 0 .. i:
            acc += scores[j] / denom * v[b*T*D + j*D + head*HD + d]
          attnOut[b*T*D + i*D + head*HD + d] = acc
  let proj = linearRef(attnOut, B*T, D, D, p.wo)
  var h2 = newSeq[float64](B * T * D)
  for i in 0 ..< B*T*D:
    h2[i] = h[i] + proj[i]
  let ln2 = layernormRef(h2, B*T, D, p.ln2w, p.ln2b)
  let fc1 = linearRef(ln2, B*T, D, 4*D, p.fc1)
  var act = newSeq[float64](fc1.len)
  for i in 0 ..< fc1.len:
    act[i] = geluRef(fc1[i])
  let fc2 = linearRef(act, B*T, 4*D, D, p.fc2)
  var h3 = newSeq[float64](B * T * D)
  for i in 0 ..< B*T*D:
    h3[i] = h2[i] + fc2[i]
  let lnf = layernormRef(h3, B*T, D, p.lnfw, p.lnfb)
  let logits = linearRef(lnf, B*T, D, V, p.lmHead)
  result = newSeq[float64](B * T)
  for b in 0 ..< B:
    for t in 0 ..< T:
      let row = b*T + t
      var maxLogit = -1e30
      for c in 0 ..< V:
        if logits[row*V + c] > maxLogit:
          maxLogit = logits[row*V + c]
      var logSum = 0.0
      for c in 0 ..< V:
        logSum += exp(logits[row*V + c] - maxLogit)
      result[row] = -(logits[row*V + int(targets[row])] - maxLogit - ln(logSum))

proc runLeg(Hw: static BackendKind) =
  randomize(11)
  let cfg = GptConfig(vocabSize: V, dModel: D, nHead: H, nLayer: L, blockSize: T, batchSize: B)
  let ctx = Ctx()
  when Hw == bkMetal:
    ensureMetal(ctx)
    metal.resetSubmissionCount()
  else:
    ensureCpu(ctx)

  # cyclic token pattern: the next token is (current + 1) mod V, learnable structure
  var tokens, targets: seq[int32]
  for i in 0 ..< B*T:
    tokens.add int32((i * 3 + 1) mod V)
    targets.add int32((int(tokens[i]) + 1) mod V)
  let xDev = toTensor(ctx, tokens, Hw, B, T)
  let yDev = toTensor(ctx, targets, Hw, B, T)

  let params = initGptParams(ctx, Hw, cfg, 0.2)
  let consts = buildConsts(ctx, Hw, cfg)
  reRegister(ctx, params, consts, cfg)

  # forward vs the independent host reference
  block:
    let lossMean = gptForward(params, consts, cfg, xDev, yDev)
    ctx.eval()
    let got = toSeq[float32](lossMean)
    var p: HostParams
    p.wte = toSeq[float32](params.wte).mapIt(float64(it))
    p.wpe = toSeq[float32](params.wpe).mapIt(float64(it))
    p.lmHead = toSeq[float32](params.lmHead).mapIt(float64(it))
    p.wq = toSeq[float32](params.layerWq[0]).mapIt(float64(it))
    p.wk = toSeq[float32](params.layerWk[0]).mapIt(float64(it))
    p.wv = toSeq[float32](params.layerWv[0]).mapIt(float64(it))
    p.wo = toSeq[float32](params.layerWo[0]).mapIt(float64(it))
    p.fc1 = toSeq[float32](params.layerFc1[0]).mapIt(float64(it))
    p.fc2 = toSeq[float32](params.layerFc2[0]).mapIt(float64(it))
    p.ln1w = toSeq[float32](params.layerLn1W[0]).mapIt(float64(it))
    p.ln1b = toSeq[float32](params.layerLn1B[0]).mapIt(float64(it))
    p.ln2w = toSeq[float32](params.layerLn2W[0]).mapIt(float64(it))
    p.ln2b = toSeq[float32](params.layerLn2B[0]).mapIt(float64(it))
    p.lnfw = toSeq[float32](params.lnfW).mapIt(float64(it))
    p.lnfb = toSeq[float32](params.lnfB).mapIt(float64(it))
    let expected = hostLoss(p, tokens, targets)
    for row in 0 ..< B*T:
      let expRow = expected[row] / float64(B*T)
      doAssert abs(float64(got[row]) - expRow) <= 1e-4 * abs(expRow) + 1e-4,
          "loss mismatch at position " & $row & ": " & $got[row] & " vs host " & $expRow
    resetTape(ctx)
    reRegister(ctx, params, consts, cfg)
    tape.register(ctx, xDev.getImpl().getTid(), xDev.getImpl())
    tape.register(ctx, yDev.getImpl().getTid(), yDev.getImpl())

  # training: the cyclic pattern is learnable, the loss must come down
  when Hw == bkMetal:
    metal.resetSubmissionCount()
  var firstLoss, lastLoss: float64
  for step in 1 .. TrainSteps:
    let lossMean = gptForward(params, consts, cfg, xDev, yDev)
    backward(lossMean)
    adamStepAll(params, cfg, step, 0.01, 0.9, 0.99, 1e-8)
    ctx.eval()
    when Hw == bkMetal:
      doAssert metal.submissionCount == 1,
          "step " & $step & ": forward + backward + adam must drain in ONE submission, got " &
          $metal.submissionCount
      metal.resetSubmissionCount()
    var s = 0.0
    for v in toSeq[float32](lossMean):
      s += float64(v)
    if step == 1:
      firstLoss = s
    if step == TrainSteps:
      lastLoss = s
    resetTape(ctx)
    reRegister(ctx, params, consts, cfg)
    tape.register(ctx, xDev.getImpl().getTid(), xDev.getImpl())
    tape.register(ctx, yDev.getImpl().getTid(), yDev.getImpl())

  doAssert lastLoss < firstLoss,
      $Hw & ": loss did not decrease over " & $TrainSteps & " steps: " &
      $firstLoss & " -> " & $lastLoss

runLeg(bkCpu)
runLeg(bkMetal)
