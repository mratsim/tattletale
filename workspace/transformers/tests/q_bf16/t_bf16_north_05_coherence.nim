# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fixture-free coherence tier for North-Mini-Code-1.0. Each prompt's answer
## position is scored by ranking, no recorded frame stands between the stack and the verdict.
##
## - the answer must be top-1, or inside top-8 while clearing every fixed rival by the stated margin
## - accepted answers and rivals come from code syntax and world knowledge, thresholds come from the logit scale
##
## Secondary, one greedy chain over the bounded budget:
##
## - every step's logits stay finite and the accepted surface form must appear among the generated ids
## - the base checkpoint free-runs without an EOS, termination is reported aloud, not asserted
## - prompts and candidates enter as checkpoint token ids, each id is BOS plus the surface form
##
## The checkpoint tokenizer has no converter. A split answer form is skipped aloud, a run that skips every prompt fails.
##
## Requires the local checkpoint at tests/hf_models/North-Mini-Code-1.0 (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_north_05_coherence.nim

import
  std/math,
  std/monotimes,
  std/os,
  std/sequtils,
  std/strformat,
  std/strutils,
  std/times,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/transformers/src/models,
  workspace/transformers/src/samplers,
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils,
  workspace/transformers/tests/harness/select_device

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" /
    "North-Mini-Code-1.0"

type Prompt = object
  ## One cloze prompt as checkpoint token ids.
  ##
  ## - promptIds, BOS followed by the prompt's surface form
  ## - accepted, the answer's surface forms, one id each
  ## - rivals, the natural spaced rival continuations, each must rank below every accepted answer
  prompt: string
  answer: string
  promptIds: seq[int]
  accepted: seq[int]
  rivals: seq[int]

proc main(): bool =
  let model = loadModel($ModelPath, testDevice())
  let device = model.getDeviceKind()
  if device != testDevice():
    echo "the checkpoint landed on " & deviceName(device) &
      ", the suite device is " & deviceName(testDevice())
    return false

  let cfg = model.getConfig()
  let dtype = F.kBFloat16
  var failures: seq[string] = @[]

  # Ranking contract, uniform across prompts.
  #
  # - top-1 is the decision a greedy chain acts upon, otherwise the answer
  #   must sit inside top-8 and clear every rival
  # - the margin floor is 2 logits, 16 bf16 ulps at logit 16, the scale these answer rows occupy
  # - honest per-op drift cannot reach the floor, a miswired rope or cache puts a rival above the answer
  let rankCap = 8
  let marginFloor = 2.0'f32
  # Secondary-chain budget. The checkpoint free-runs past any affordable horizon,
  # 256-step greedy chains measured from both cloze families never emitted EOS (id 255001).
  #
  # - the chain checks finiteness and its accepted surface form
  # - termination is reported aloud, not asserted
  let maxNewTokens = 256
  let maxContextLen = 1024

  # Token ids verified against the checkpoint tokenizer.json:
  # an id is BOS (2) followed by the surface form's single BPE token.
  let prompts = [
    Prompt(prompt: "The capital of France is", answer: "Paris",
      promptIds: @[2, 669, 6345, 302, 8987, 341],
      accepted: @[63533, 12071],
      rivals: @[7318, 18544, 19454]),
    Prompt(prompt: "def add(a, b):\n    return a", answer: "+",
      promptIds: @[2, 1847, 919, 6439, 16, 290, 2407, 298, 758, 265],
      accepted: @[15, 692],
      rivals: @[506, 535]),
    Prompt(prompt: "def max_value(a, b):\n    return a if a > b else", answer: "b",
      promptIds: @[2, 1847, 2687, 7325, 6439, 16, 290, 2407, 298, 758, 265,
        592, 265, 1518, 290, 1625],
      accepted: @[70, 290],
      rivals: @[265, 274, 1150]),
    Prompt(prompt: "for i in range(10", answer: ")",
      promptIds: @[2, 1943, 700, 297, 2783, 12, 622],
      accepted: @[13],
      rivals: @[30, 97, 12015]),
    Prompt(prompt: "The largest planet in the solar system is", answer: "Jupiter",
      promptIds: @[2, 669, 8331, 12422, 297, 277, 10118, 1181, 341],
      accepted: @[203349, 50332],
      rivals: @[54796, 23656, 9165])]

  var ranked = 0
  var skipped = 0
  for item in prompts:
    if item.accepted.len == 0 or item.rivals.len == 0:
      echo "SKIP, no single-token answer form for prompt ", item.prompt
      skipped += 1
      continue

    var orc = newOrchestrator(model, dtype = dtype, maxContextLen = maxContextLen)
    defer: orc.endSequence()

    orc.startSequence(item.promptIds.mapIt(it.uint32))
    let logits = model.forward(orc.getInferenceContextMut(),
      F.toTensor([item.promptIds]).to(device))
    orc.setKvPosition(item.promptIds.len)

    let row = nextStepRow(logits, item.promptIds.len - 1)
      .contiguous().to(F.kCPU).to(F.kFloat32).contiguous()
    let vocab = row.numel()
    let raw = cast[ptr UncheckedArray[float32]](row.data_ptr(float32))
    for i in 0 ..< vocab:
      if classify(raw[i]) in {fcNaN, fcInf, fcNegInf}:
        failures.add "prompt " & item.prompt &
          " has a non-finite logit at vocab id " & $i

    var answerId = -1
    var answerLogit = NegInf.float32
    for id in item.accepted:
      if raw[id] > answerLogit:
        answerLogit = raw[id]
        answerId = id
    var rivalId = -1
    var rivalLogit = NegInf.float32
    for id in item.rivals:
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
      " prompt(s) named and none decoded as a single-token answer, so the tier" &
      " checked nothing"

  # Secondary, clearly labelled, one greedy chain from the France cloze,
  # free-running argmax, checked for its accepted surface form and its EOS.
  block:
    let france = prompts[0]
    var orc = newOrchestrator(model, dtype = dtype, maxContextLen = maxContextLen)
    defer: orc.endSequence()

    var ids = france.promptIds
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
      if (cfg.eosTokenIds.len > 0 and chosen in cfg.eosTokenIds) or
          chosen == cfg.eosTokenId:
        sawEos = true
        break

      orc.appendToken(ids.len - 1, chosen.uint32, device)
      let stepLogits = model.forward(orc.getInferenceContextMut(),
        F.toTensor([[chosen]]).to(device))
      orc.setKvPosition(ids.len)
      row = nextStepRow(stepLogits, 0)

    let chainWall = (getMonoTime() - chainStart).inNanoseconds.float64 * 1e-9
    echo "  greedy chain ", generated.len, " tokens in ",
      formatFloat(chainWall, ffDecimal, precision = 3), " s (",
      formatFloat(generated.len.float64 / chainWall, ffDecimal, precision = 2),
      " tok/s)"
    echo "  greedy ids ", generated[0 ..< min(generated.len, 32)]
    echo "  greedy chain reached EOS: ", sawEos
    if not france.accepted.anyIt(it in generated):
      failures.add "greedy chain never produced the accepted answer surface" &
        " form id " & $france.accepted & ", ids " & $generated

  if failures.len > 0:
    raise newException(ValueError, # noqa, one failure site names every check
      $failures.len & " coherence check(s) failed, " & failures.join(" | "))

  result = true

when isMainModule:
  runCppTest("north coherence", main)
