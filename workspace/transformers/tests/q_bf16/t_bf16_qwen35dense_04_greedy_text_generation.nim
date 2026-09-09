# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35dense-greedy \
##   --nimcache:nimcache/tests/qwen35dense-greedy \
##   workspace/transformers/tests/q_bf16/t_bf16_qwen35dense_04_greedy_text_generation.nim

import
  std/json,
  std/os,
  std/sequtils,
  std/strutils,
  workspace/libtorch as F,
  workspace/toktoktok,
  workspace/transformers/src/models,
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-04-greedy-text-generation" / "Qwen3.5-0.8B"

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
  ## tt-greedy-2 step node to GreedyStepRef.
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
  ## teacher-forced throughout. Tie flips stay recorded, later checks
  ## must re-converge, and the flip cap bounds flips.
  let promptIds = fixture["prompt_ids"].mapIt(it.getInt())
  let expected = fixture["generated_ids"].mapIt(it.getInt())
  let horizon = expected.len
  let cfg = model.getConfig()
  let device = model.getDeviceKind()
  # Decode context ceiling for the replay: the recorded chains stay under
  # 20 tokens (prompt + horizon), so one TokensPerPage page covers every
  # chain with headroom. A pool sized by the model's max_position_embeddings
  # (262144) would pre-allocate a six-gigabyte KV pool for a 15-token chain.
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
  # The chain replays on the device testDevice() resolves, Metal by the auto
  # default on macOS, with a PyTorch fallback where kernels are missing.
  # TTT_TEST_ON=metal|cpu|cuda flips the device.
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  runCppTest "Qwen3.5-0.8B greedy decoding - prefix checks vs tt-greedy-2 fixtures":
    proc(): bool =
      echo "Loading model..."
      let model = loadModel($ModelPath, testDevice())
      echo "Model loaded."
      var passed = 0
      var total = 0
      for f in ["Hello_how_are_you.json.zst",
                "The_resume_is_ready.json.zst",
                "What_is_the_capital_of_France.json.zst"]:
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
  runCppTest "Qwen3.5-0.8B t2t entry: decode conventions + prefill-to-decode transition":
    proc(): bool =
      echo "Loading model..."
      let model = loadModel($(ModelPath), kCPU)

      # The generate() stop condition uses the config eos_token_id (248044).
      doAssert model.getConfig().eosTokenId == 248044,
        "generate() must stop at config eos 248044, not the tokenizer im_end 248046"

      # Decode entry: encode(prompt) must equal the vendored fixture
      # prompt_ids exactly, proving that neither side prepends a bos
      # token. The two NFC-clean fixture prompts lock the convention.
      for f in ["Hello_how_are_you.json.zst", "What_is_the_capital_of_France.json.zst"]:
        let data = parseJson(zstdReadFixture(FixtureDir / f))
        let prompt = data["prompt"].getStr()
        var expectedIds: seq[int] = @[]
        for el in data["prompt_ids"]:
          expectedIds.add(el.getInt())
        doAssert model.getTokenizer().encode(prompt) == expectedIds,
          "decode entry diverges from the vendored convention in " & f

      # Combining marks: the pre-tokenizer regex includes \p{M}. The resume
      # fixture prompt is decomposed (e + U+0301). Tokenize then untokenize
      # must reproduce the prompt text byte for byte. toktoktok does not
      # implement the tokenizer.json NFC normalizer, so a decomposed prompt
      # tokenizes differently from the vendored tokenizer, and the greedy
      # test uses the precomposed form. The tokenizer carries the base vocab
      # plus 26 added specials: 248070 ids, distinct from the 248320 embed
      # width.
      let resumePrompt = "The re\u0301sume\u0301 is ready"
      let markTokens = model.getTokenizer().encode(resumePrompt)
      doAssert model.getTokenizer().decodeToString(markTokens) == resumePrompt,
        "combining-mark prompt must round-trip through tokenize/untokenize"
      let precomposedTokens = model.getTokenizer().encode("The résumé is ready")
      doAssert markTokens != precomposedTokens,
        "decomposed and precomposed forms must tokenize differently (NFC gap locked)"
      let tokenCount = model.getTokenizer().tokenCount()
      doAssert tokenCount == 248070,
        "tokenizer vocab must be 248044 + 26 added specials, got " & $tokenCount

      # End-to-end generate: prefill on the prompt, decode a short
      # continuation. maxContextLen is bounded so the page pool stays
      # small: the default maxContextLen = -1 would size the pool to the full
      # max_position_embeddings 262144.
      let output = model.generate(resumePrompt, temp = 1.0f, maxTokens = 16,
                                  maxContextLen = 512)
      echo "Output: " & output
      doAssert output.len > resumePrompt.len, "output must be longer than the prompt"
      doAssert output.startsWith(resumePrompt), "output must start with the prompt text"
      true

when isMainModule:
  main()
