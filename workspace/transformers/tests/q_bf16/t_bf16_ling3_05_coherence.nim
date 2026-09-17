# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fixture-free coherence rung for Ling-3.0-tiny. The answer position of each
## prompt is scored by ranking, so no recorded frame stands between the stack
## and the verdict.
##
## - the answer must be top-1, or inside top-8 while clearing every fixed rival
##   by the stated margin
## - accepted answers and rivals come from world knowledge, thresholds come
##   from the logit scale
## - secondary, one greedy chain checked for finiteness, its accepted surface
##   form and its EOS
##
## A split answer form is skipped aloud. A run that skips every prompt fails.
##
## Requires the local checkpoint at tests/hf_models/Ling-3.0-tiny (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_ling3_05_coherence.nim

import
  std/math,
  std/monotimes,
  std/os,
  std/sequtils,
  std/strformat,
  std/strutils,
  std/times,
  workspace/libtorch as F,
  workspace/toktoktok,
  workspace/transformers/src/models,
  workspace/transformers/src/models/ling3 {.all.},
  workspace/transformers/src/samplers,
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/harness/select_device

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" /
    "Ling-3.0-tiny"

type Prompt = object
  ## One world-knowledge question with its accepted answer forms and its fixed
  ## rival answers, which must rank below the accepted answer.
  prompt: string
  answer: string
  accepted, rivals: seq[string]

proc main() =
  let model = loadModel($ModelPath, testDevice())
  let device = model.getDeviceKind()
  if device != testDevice():
    raise newException(ValueError, # noqa, a CPU fallback ends the run here
      "the checkpoint landed on " & deviceName(device) &
      ", the suite device is " & deviceName(testDevice()))

  let tok = model.getTokenizer()
  let cfg = model.getConfig()
  let dtype = F.kBFloat16
  var failures: seq[string] = @[]

  # Ranking contract, uniform across prompts.
  #
  # - top-1 is the decision a greedy chain acts upon, nothing else is checked
  # - otherwise the answer must sit inside top-8 and clear every rival
  # - the margin floor is 32 bf16 ulps at logit 11, the scale these rows
  #   answer at, so honest per-op drift cannot reach it while a miswired rope
  #   or cache puts a rival above the answer
  let rankCap = 8
  let marginFloor = 2.0'f32
  # Decoding budget of the secondary chain, the measured EOS lands well inside.
  let maxNewTokens = 256
  let maxContextLen = 1024

  let prompts = [
    Prompt(prompt: "What is the capital of France?", answer: "Paris",
      accepted: @["Paris", " Paris"],
      rivals: @["London", " London", "Berlin", " Berlin", "Rome", " Rome"]),
    Prompt(prompt: "The capital of France is", answer: "Paris",
      accepted: @["Paris", " Paris"],
      rivals: @["London", " London", "Berlin", " Berlin", "Rome", " Rome"]),
    Prompt(prompt: "The largest planet in the solar system is", answer: "Jupiter",
      accepted: @["Jupiter", " Jupiter"],
      rivals: @["Saturn", " Saturn", "Neptune", " Neptune", "Mars", " Mars",
        "Earth", " Earth"]),
    Prompt(prompt: "The planet known as the Red Planet is", answer: "Mars",
      accepted: @["Mars", " Mars"],
      rivals: @["Jupiter", " Jupiter", "Saturn", " Saturn", "Venus", " Venus",
        "Neptune", " Neptune", "Earth", " Earth"]),
    Prompt(prompt: "The chemical symbol for gold is", answer: "Au",
      accepted: @["Au", " Au"],
      rivals: @["Ag", " Ag", "Cu", " Cu", "Fe", " Fe", "Pb", " Pb"])]

  var ranked = 0
  var skipped = 0
  for item in prompts:
    var acceptedIds: seq[int] = @[]
    for form in item.accepted:
      let e = tok.encode(form)
      if e.len == 1:
        acceptedIds.add e[0]
    var rivalIds: seq[int] = @[]
    for form in item.rivals:
      let e = tok.encode(form)
      if e.len == 1:
        rivalIds.add e[0]
    if acceptedIds.len == 0 or rivalIds.len == 0:
      echo "SKIP, no single-token answer form for prompt ", item.prompt
      skipped += 1
      continue

    let numPoolPages = computeNumPages(maxContextLen, concurrentRequests = 1)
    var orc = Orchestrator.init(
      cfg.num_hidden_layers, 1,
      1, cfg.mlaKvLoraRank,
      1, cfg.mlaKpeWidth,
      maxContextLen, numPoolPages, dtype, device)
    defer: orc.endSequence()

    let promptIds = tok.encode(item.prompt)
    orc.startSequence(promptIds.mapIt(it.uint32))
    let logits = model.forward(orc.getInferenceContextMut(),
      F.toTensor([promptIds]).to(device))
    orc.setKvPosition(promptIds.len)

    let row = nextStepRow(logits, promptIds.len - 1)
      .contiguous().to(F.kCPU).to(F.kFloat32).contiguous()
    let vocab = row.numel()
    let raw = cast[ptr UncheckedArray[float32]](row.data_ptr(float32))
    for i in 0 ..< vocab:
      if classify(raw[i]) in {fcNaN, fcInf, fcNegInf}:
        failures.add "prompt " & item.prompt &
          " has a non-finite logit at vocab id " & $i

    var answerId = -1
    var answerLogit = NegInf.float32
    for id in acceptedIds:
      if raw[id] > answerLogit:
        answerLogit = raw[id]
        answerId = id
    var rivalId = -1
    var rivalLogit = NegInf.float32
    for id in rivalIds:
      if raw[id] > rivalLogit:
        rivalLogit = raw[id]
        rivalId = id
    var above = 0
    for i in 0 ..< vocab:
      if raw[i] > answerLogit:
        above += 1
    let rank = above + 1
    let margin = answerLogit - rivalLogit

    echo &"  {item.answer}: id={answerId} logit={answerLogit:.4f} " &
      &"rank={rank} bestRival={rivalId}({rivalLogit:.4f}) margin={margin:.4f}"
    if rank != 1 and not (rank <= rankCap and margin >= marginFloor):
      failures.add "prompt " & item.prompt & " answer " & item.answer &
        " landed at rank " & $rank & " with margin " & $margin &
        " over rival id " & $rivalId & ", the contract is top-1 or inside top-" &
        $rankCap & " by " & $marginFloor
    ranked += 1

  if ranked == 0:
    failures.add "every prompt skipped, " & $skipped &
      " prompt(s) named and none decoded as a single-token answer, so the rung" &
      " checked nothing"

  # Secondary, clearly labelled, one greedy chain from the France cloze,
  # free-running argmax, checked for its accepted surface form and its EOS.
  block:
    let prompt = "The capital of France is"
    let numPoolPages = computeNumPages(maxContextLen, concurrentRequests = 1)
    var orc = Orchestrator.init(
      cfg.num_hidden_layers, 1,
      1, cfg.mlaKvLoraRank,
      1, cfg.mlaKpeWidth,
      maxContextLen, numPoolPages, dtype, device)
    defer: orc.endSequence()

    var ids = tok.encode(prompt)
    orc.startSequence(ids.mapIt(it.uint32))
    let logits = model.forward(orc.getInferenceContextMut(),
      F.toTensor([ids]).to(device))
    orc.setKvPosition(ids.len)
    var row = nextStepRow(logits, ids.len - 1)

    let chainStart = getMonoTime()
    var generated: seq[int] = @[]
    var sawEos = false
    while generated.len < maxNewTokens and ids.len < maxContextLen:
      let stepRow = row.contiguous().to(F.kCPU).to(F.kFloat32).contiguous()
      let vocab = stepRow.numel()
      let raw = cast[ptr UncheckedArray[float32]](stepRow.data_ptr(float32))
      for i in 0 ..< vocab:
        if classify(raw[i]) in {fcNaN, fcInf, fcNegInf}:
          failures.add "greedy step " & $generated.len &
            " has a non-finite logit at vocab id " & $i
      let chosen = sample(row, 0.0f)
      generated.add chosen
      ids.add chosen
      if cfg.eosTokenIds.len > 0 and chosen in cfg.eosTokenIds:
        sawEos = true
        break
      orc.appendToken(ids.len - 1, chosen.uint32, device)
      let stepLogits = model.forward(orc.getInferenceContextMut(),
        F.toTensor([[chosen]]).to(device))
      orc.setKvPosition(ids.len)
      row = nextStepRow(stepLogits, 0)

    let chainWall = (getMonoTime() - chainStart).inNanoseconds.float64 * 1e-9
    let text = tok.decodeToString(generated).replace("\n", "|")
    echo "  greedy chain ", generated.len, " tokens in ",
      formatFloat(chainWall, ffDecimal, precision = 3), " s (",
      formatFloat(generated.len.float64 / chainWall, ffDecimal, precision = 2),
      " tok/s)"
    echo "  greedy text ", text
    if not sawEos:
      failures.add "greedy chain reached no EOS inside " & $maxNewTokens &
        " steps, text " & text
    if "Paris" notin text:
      failures.add "greedy chain never produced the accepted answer surface" &
        " form Paris, text " & text

  if failures.len > 0:
    raise newException(ValueError, # noqa, one failure site names every check
      $failures.len & " coherence check(s) failed, " & failures.join(" | "))

when isMainModule:
  main()
