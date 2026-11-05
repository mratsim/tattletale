# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Microgpt end to end on tensorcery: train a small GPT on the Shakespeare
## corpus and generate text, on CPU or Metal, with a Vulkan forward leg.
##
## One training step = one record = one backend submission:
## the forward ops, the backward grad segment and the in-place adam updates
## enqueue into a single record and drain in ONE eval.
##
## Run from the worktree root (backend is the Backend const, one line):
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests \
##     --nimcache:nimcache/microgpt workspace/tensorcery/examples/microgpt.nim

import std/[math, os, osproc, random, sequtils, strutils, tables]

import workspace/data_structures/src/small_seqs
import workspace/tensorcery
import workspace/tensorcery/src/tape
import workspace/tensorcery/src/backends/cpu
import workspace/tensorcery/src/backends/metal
import workspace/tensorcery/src/backends/vulkan
import workspace/tensorcery/examples/microgpt_model

const
  # Backend switch: bkCpu or bkMetal for the full train + generate pipeline.
  # bkVulkan has no backward or optimizer kernels yet; it runs the forward leg
  # at the end (VulkanLeg) and is wired below as the generation demo.
  Backend = bkMetal
  VulkanLeg = true

  Seed = 42
  Steps = 60
  Lr = 0.003
  Beta1 = 0.9
  Beta2 = 0.99
  EpsAdam = 1e-8
  InitStd = 0.02
  GenChars = 400
  Temperature = 0.8

  Cfg = GptConfig(
    vocabSize: 0,  # filled from the corpus below
    dModel: 64,
    nHead: 4,
    nLayer: 2,
    blockSize: 16,
    batchSize: 8)

  DefaultCorpus = currentSourcePath().parentDir() /
    "../../../../../../_references_nim_libs/arraymancer/examples/ex06_shakespeare_input.txt"

proc hostSoftmax(x: seq[float64]): seq[float64] =
  var m = -1e30
  for v in x:
    if v > m:
      m = v
  var s = 0.0
  result = newSeq[float64](x.len)
  for i in 0 ..< x.len:
    result[i] = exp(x[i] - m)
    s += result[i]
  for i in 0 ..< x.len:
    result[i] /= s

proc sampleFrom(probs: seq[float64]): int =
  let r = rand(1.0)
  var acc = 0.0
  for i, p in probs:
    acc += p
    if r <= acc:
      return i
  probs.len - 1

proc rssKb(): int =
  ## Rough cross-step memory probe: resident set size in KB via ps.
  let outp = execProcess("ps -o rss= -p " & $getCurrentProcessId())
  parseInt(outp.strip())

proc trainStep(ctx: Ctx, params: GptParams, consts: GptConsts, cfg: GptConfig, tokens, targets: Tensor, step, totalSteps: int): float64 =
  ## One training step: forward, backward and adam enqueue into one record,
  ## one eval drains it, one submission per backend.
  let lossMean = gptForward(params, consts, cfg, tokens, targets)
  backward(lossMean)
  let lr = Lr * (1.0 - float64(step - 1) / float64(totalSteps))
  adamStepAll(params, cfg, step, lr, Beta1, Beta2, EpsAdam)
  ctx.eval()
  when Backend == bkMetal:
    doAssert metal.submissionCount == 1,
        "step " & $step & ": forward + backward + adam must drain in ONE submission, got " &
        $metal.submissionCount
    metal.resetSubmissionCount()
  var s = 0.0
  for v in toSeq[float32](lossMean):
    s += float64(v)
  resetTape(ctx)
  reRegister(ctx, params, consts, cfg)
  s

proc generateText(ctx: Ctx, Hw: static BackendKind, params: GptParams,
    consts: GptConsts, cfg: GptConfig, vocab: seq[char], prompt: string,
    nNew: int, temperature: float64): string =
  ## Samples nNew characters from the trained model. Each position runs one
  ## forward record and one eval. Sampling happens on the host.
  let idOf = block:
    var t = initTable[char, int32]()
    for i, c in vocab:
      t[c] = int32(i)
    t
  var window = newSeq[int32](cfg.blockSize)
  for t in 0 ..< cfg.blockSize:
    window[t] = idOf[prompt[t]]
  var text = prompt
  for iter in 0 ..< nNew:
    let xDev = toTensor(ctx, window, Hw, 1, cfg.blockSize)
    let logits = gptLogits(params, consts, cfg, xDev)
    ctx.eval()
    when Hw == bkMetal:
      doAssert metal.submissionCount == 1,
          "generation step " & $iter & " must drain in ONE submission, got " &
          $metal.submissionCount
      metal.resetSubmissionCount()
    let lastRow = slice(logits, 0 .. 0, cfg.blockSize - 1 .. cfg.blockSize - 1)
    var scaled = newSeq[float64](cfg.vocabSize)
    let raw = toSeq[float32](lastRow)
    for c in 0 ..< cfg.vocabSize:
      scaled[c] = float64(raw[c]) / temperature
    let nextId = sampleFrom(hostSoftmax(scaled))
    text.add vocab[nextId]
    for t in 0 ..< cfg.blockSize - 1:
      window[t] = window[t + 1]
    window[cfg.blockSize - 1] = int32(nextId)
    resetTape(ctx)
    reRegister(ctx, params, consts, cfg)
  text

proc main =
  randomize(Seed)
  let corpusPath = if paramCount() > 0: paramStr(1) else: DefaultCorpus
  let steps = if paramCount() > 1: parseInt(paramStr(2)) else: Steps
  let genChars = if paramCount() > 2: parseInt(paramStr(3)) else: GenChars
  doAssert fileExists(corpusPath), "corpus not found: " & corpusPath
  let corpus = readFile(corpusPath)

  # char-level tokenizer (microgpt.py's scheme, see README for the BPE ruling)
  var vocab: seq[char]
  var idOf = initTable[char, int32]()
  for c in corpus:
    if c notin idOf:
      idOf[c] = int32(vocab.len)
      vocab.add c
  var tokens = newSeq[int32](corpus.len)
  for i, c in corpus:
    tokens[i] = idOf[c]

  var cfg = Cfg
  cfg.vocabSize = vocab.len
  echo "corpus chars: ", corpus.len, "  vocab: ", cfg.vocabSize,
      "  dModel: ", cfg.dModel, "  layers: ", cfg.nLayer,
      "  block: ", cfg.blockSize, "  batch: ", cfg.batchSize

  let ctx = Ctx()
  when Backend == bkMetal:
    ensureMetal(ctx)
    metal.resetSubmissionCount()
  else:
    ensureCpu(ctx)

  let params = initGptParams(ctx, Backend, cfg, InitStd)
  let consts = buildConsts(ctx, Backend, cfg)
  reRegister(ctx, params, consts, cfg)

  # training: one eval per step, loss trajectory printed
  var first5: seq[float64]
  var last5: seq[float64]
  for step in 1 .. steps:
    let o = rand(corpus.len - cfg.batchSize * cfg.blockSize - 2)
    var x, y: seq[int32]
    for i in 0 ..< cfg.batchSize * cfg.blockSize:
      x.add tokens[o + i]
      y.add tokens[o + i + 1]
    let xDev = toTensor(ctx, x, Backend, cfg.batchSize, cfg.blockSize)
    let yDev = toTensor(ctx, y, Backend, cfg.batchSize, cfg.blockSize)
    let loss = trainStep(ctx, params, consts, cfg, xDev, yDev, step, steps)
    doAssert loss == loss and abs(loss) < 1e6, "step " & $step & ": loss is not finite: " & $loss
    if step <= 5:
      first5.add loss
    if step > steps - 5:
      last5.add loss
    echo "step ", step, " / ", steps, "  loss ", loss.formatFloat(ffDecimal, 4),
        "  rss_kb ", rssKb()

  let firstAvg = first5.sum() / 5.0
  let lastAvg = last5.sum() / 5.0
  echo "loss trajectory: first5 avg ", firstAvg.formatFloat(ffDecimal, 4),
      "  last5 avg ", lastAvg.formatFloat(ffDecimal, 4)
  doAssert lastAvg < firstAvg,
      "loss did not decrease: first5 " & $firstAvg & " last5 " & $lastAvg

  # generation on the training backend: one sequence, its own constant set
  let genCfg = GptConfig(
    vocabSize: cfg.vocabSize, dModel: cfg.dModel, nHead: cfg.nHead,
    nLayer: cfg.nLayer, blockSize: cfg.blockSize, batchSize: 1)
  let genConsts = buildConsts(ctx, Backend, genCfg)
  reRegister(ctx, params, genConsts, genCfg)
  let p0 = rand(corpus.len - cfg.blockSize - 1)
  let prompt = corpus[p0 ..< p0 + cfg.blockSize]
  echo "\n--- generation (", $Backend, ", prompt: ", prompt.replace("\n", "\\n"), ") ---"
  echo generateText(ctx, Backend, params, genConsts, genCfg, vocab, prompt, genChars, Temperature)

  when VulkanLeg:
    # Vulkan forward-only pass: no backward or optimizer kernels exist for
    # Vulkan yet, so this pass copies the trained weights and runs forward work.
    ensureVulkan(ctx)
    vulkan.resetSubmissionCount()
    proc zeros(n: int): Tensor =
      toTensor(ctx, newSeqWith(n, 0'f32), bkVulkan, n)
    proc zeroLike(t: Tensor): Tensor =
      let sh = t.getShape()
      let rank = int(sh.len)
      if rank == 1:
        zeros(t.getNumel())
      else:
        zeros(t.getNumel()).reshape(int(sh[0]), int(sh[1]))
    var vp: GptParams
    vp.wte = params.wte.to(bkVulkan)
    vp.wpe = params.wpe.to(bkVulkan)
    vp.lmHead = params.lmHead.to(bkVulkan)
    vp.lnfW = params.lnfW.to(bkVulkan)
    vp.lnfB = params.lnfB.to(bkVulkan)
    vp.mWte = zeroLike(vp.wte)
    vp.vWte = zeroLike(vp.wte)
    vp.mWpe = zeroLike(vp.wpe)
    vp.vWpe = zeroLike(vp.wpe)
    vp.mLmHead = zeroLike(vp.lmHead)
    vp.vLmHead = zeroLike(vp.lmHead)
    vp.mLnfW = zeroLike(vp.lnfW)
    vp.vLnfW = zeroLike(vp.lnfW)
    vp.mLnfB = zeroLike(vp.lnfB)
    vp.vLnfB = zeroLike(vp.lnfB)
    for layer in 0 ..< cfg.nLayer:
      vp.layerWq.add params.layerWq[layer].to(bkVulkan)
      vp.layerWk.add params.layerWk[layer].to(bkVulkan)
      vp.layerWv.add params.layerWv[layer].to(bkVulkan)
      vp.layerWo.add params.layerWo[layer].to(bkVulkan)
      vp.layerFc1.add params.layerFc1[layer].to(bkVulkan)
      vp.layerFc2.add params.layerFc2[layer].to(bkVulkan)
      vp.layerLn1W.add params.layerLn1W[layer].to(bkVulkan)
      vp.layerLn1B.add params.layerLn1B[layer].to(bkVulkan)
      vp.layerLn2W.add params.layerLn2W[layer].to(bkVulkan)
      vp.layerLn2B.add params.layerLn2B[layer].to(bkVulkan)
      var m, v: array[10, Tensor]
      for slot in 0 ..< 10:
        m[slot] = zeros(cfg.dModel * cfg.dModel)
        v[slot] = zeros(cfg.dModel * cfg.dModel)
      vp.layerM.add m
      vp.layerV.add v
    let vConsts = buildConsts(ctx, bkVulkan, genCfg)
    reRegister(ctx, vp, vConsts, genCfg)
    let vConstsTrain = buildConsts(ctx, bkVulkan, cfg)

    # forward loss on Vulkan must match the training backend on the same batch
    var x, y: seq[int32]
    for i in 0 ..< cfg.batchSize * cfg.blockSize:
      x.add tokens[i]
      y.add tokens[i + 1]
    let xv = toTensor(ctx, x, bkVulkan, cfg.batchSize, cfg.blockSize)
    let yv = toTensor(ctx, y, bkVulkan, cfg.batchSize, cfg.blockSize)
    let vLoss = gptForward(vp, vConstsTrain, cfg, xv, yv)
    ctx.eval()
    doAssert vulkan.submissionCount == 1,
        "the Vulkan forward eval must submit once, got " & $vulkan.submissionCount
    vulkan.resetSubmissionCount()
    var s = 0.0
    for v in toSeq[float32](vLoss):
      s += float64(v)
    echo "\nVulkan forward loss on the first batch: ",
        s.formatFloat(ffDecimal, 4), " (one submission)"

    let vPrompt = corpus[p0 ..< p0 + cfg.blockSize]
    echo "--- generation (bkVulkan forward leg) ---"
    echo generateText(ctx, bkVulkan, vp, vConsts, genCfg, vocab, vPrompt, genChars, Temperature)

main()
