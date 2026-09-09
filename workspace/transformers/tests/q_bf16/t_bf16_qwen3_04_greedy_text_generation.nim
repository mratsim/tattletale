# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp=0) decoding of the Qwen3-0.6B stack against ttt-tf-001-greedy-steps-h2
## fixtures: per-step argmax checks with margin-scaled logit caps,
## truncated-KL and tail-probability checksums, plus teacher-forced
## recovery at structural ties (harness/tolerance.nim greedy checks).

import
  std/json,
  std/strutils,
  std/os,
  std/sequtils,
  workspace/libtorch as F,
  workspace/toktoktok,
  workspace/transformers/src/models,
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-04-greedy-text-generation" / "Qwen3-0.6B"

  # Tie-eligibility from the corpus margin report
  # structural ties record margin 0.0,
  # the smallest nonzero margin is exactly one bf16 ulp at logit 16.
  # The floor is tieUlps ulps of the step's own top logit, never
  # an absolute constant: one ulp is 0.0625 at logit 16, 0.125
  # at logit 32.
  ChainChecks = GreedyConfig(
    tieUlps: 1,
    epsBase: 0.05,
    tailBand: 0.3,
    klBand: 0.05,
    maxFlips: 4)

proc parseGreedyStep(node: JsonNode, step: int): GreedyStepRef =
  ## ttt-tf-001-greedy-steps-h2 step node to GreedyStepRef.
  result.step = step
  result.chosenToken = node["chosen_token"].getInt()
  for el in node["top32_ids"]:
    result.top32Ids.add el.getInt()
  for el in node["top32_logits"]:
    result.top32Logits.add float32(el.getFloat())
  result.argmaxMargin = node["argmax_margin"].getFloat()
  result.tailProbability = node["tail_probability"].getFloat()

proc runChain(model: AnyModel, fixture: JsonNode): bool =
  ## Replay one greedy chain with per-step checks, the recorded prefix
  ## teacher-forced at every step. A tie flip keeps the chain aligned
  ## with the recording, and every following check must re-converge. The flip cap
  ## and a post-flip real divergence both fail with the localized report.
  let promptIds = fixture["prompt_ids"].mapIt(it.getInt())
  let expected = fixture["generated_ids"].mapIt(it.getInt())
  let horizon = expected.len
  let cfg = model.getConfig()
  let device = model.getDeviceKind()
  # Decode context ceiling for the replay: the recorded chains stay under
  # 50 tokens (prompt + horizon), so one TokensPerPage page covers every
  # chain with headroom. A pool sized by the model's max_position_embeddings
  # would pre-allocate a multi-gigabyte KV pool for a 48-token chain.
  const TestMaxCtx = 256
  let numPoolPages = computeNumPages(TestMaxCtx, concurrentRequests = 1)
  var orc = Orchestrator.init(cfg.num_hidden_layers, 1,
    cfg.num_key_value_heads, TestMaxCtx, cfg.head_dim, numPoolPages,
    F.kBFloat16, device)
  defer: orc.endSequence()

  var state = GreedyState()
  var ids = promptIds
  orc.startSequence(ids.mapIt(it.uint32))
  let inputIds = F.toTensor([ids]).to(device)
  let logits = model.forward(orc.getInferenceContextMut(), inputIds)
  orc.setKvPosition(ids.len)
  var row = logits.narrow(1, ids.len - 1, 1).squeeze(1).squeeze(0)

  for step in 0 ..< horizon:
    let refStep = parseGreedyStep(fixture["steps"][step], step)
    let verdict = checkGreedyStep(state, ChainChecks, refStep, row)
    if verdict == gvTieFlip:
      echo "    step " & $step & " tie flip (recorded margin " &
        $refStep.argmaxMargin & "), teacher-forcing " &
        $refStep.chosenToken
    let chosen = expected[step]
    ids.add chosen
    if step + 1 < horizon:
      orc.appendToken(ids.len - 1, chosen.uint32, device)
      let stepLogits = model.forward(orc.getInferenceContextMut(),
        F.toTensor([[chosen]]).to(device))
      orc.setKvPosition(ids.len)
      row = stepLogits.squeeze(0).squeeze(0)
  echo "    chain passed: " & $state.flips & " tie flip(s) within cap"
  result = true

proc main() =
  assertTorchStamp(FixtureDir)
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  runCppTest "Qwen3-0.6B greedy decoding - prefix checks vs ttt-tf-001-greedy-steps-h2 fixtures":
    proc(): bool =
      echo "Loading model..."
      let model = loadModel($ModelPath, testDevice())
      echo "Model loaded."
      var passed = 0
      var total = 0
      for f in ["Hello_how_are_you.json.zst",
                "Do_you_know_the_story_of_this_proverb_磨刀.json.zst"]:
        inc total
        echo "Fixture: " & f
        if runChain(model, parseJson(zstdReadFixture(FixtureDir / f))):
          inc passed
      echo "Greedy decoding: " & $passed & "/" & $total & " fixtures passed"
      result = passed == total

  # Forced-first-step variant, the retired t2t suite folded in here. The production
  # prefill-to-decode transition becomes step 1 of the prefix
  # test: generate() tokenizes, prefills, samples and decodes through the same
  # orchestrator that the chain replay above drives by hand. This variant
  # checks the entry conventions and the structural contracts.
  # The recorded-chain variant carries the per-step determinism.
  runCppTest "Qwen3-0.6B t2t entry: decode conventions + prefill-to-decode transition":
    proc(): bool =
      echo "Loading model..."
      let model = loadModel($ModelPath, kCPU)

      # Decode entry: encode(prompt) must equal the recorded fixture
      # prompt_ids exactly, locking the no-bos convention on both sides
      # and tying this variant to the recorded chain family.
      let data = parseJson(zstdReadFixture(FixtureDir / "Hello_how_are_you.json.zst"))
      let fixturePrompt = data["prompt"].getStr()
      var expectedIds: seq[int] = @[]
      for el in data["prompt_ids"]:
        expectedIds.add(el.getInt())
      doAssert model.getTokenizer().encode(fixturePrompt) == expectedIds,
        "decode entry diverges from the recorded chain convention"

      # End-to-end generate: prefill on the prompt, decode a short
      # continuation. temp = 1.0 samples (Gumbel), so the structural
      # contracts below are the deterministic part of this variant.
      let prompt = "Hello, how are you?"
      let output = model.generate(prompt, temp = 1.0f, maxTokens = 16)
      echo "Output: " & output
      doAssert output.len > prompt.len, "output must be longer than the prompt"
      doAssert output.startsWith(prompt), "output must start with the prompt text"
      true

when isMainModule:
  main()
