# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   TTT_TEST_ON=cpu nim test_tf_exl3_qwen3_04_greedy_text_generation

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
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-04-greedy-text-generation"

  # Tie-eligibility and step caps of the dense chain rows, the ulp unit
  # fp16. The floor takes the deep-stack row: the mac replay against the
  # CUDA recording measured up to ~47 fp16 ulps of support drift at the
  # depth-29 chain, the same scale the 35B bf16 stack measured on MPS and
  # covered with 64 (GreedyConfig.ulpFloorUlps derivation).
  ChainChecks = GreedyConfig(
    tieUlps: 1,
    epsBase: 0.05,
    tailBand: 0.3,
    klBand: 0.05,
    maxFlips: 4,
    ulpFloorUlps: 64,
    ulpUnitF16: true)

proc parseGreedyStep(node: JsonNode, step: int): GreedyStepRef =
  ## One recorded step node to GreedyStepRef. The support is whatever the
  ## recording carries (top-32 of the ttt-tf-001-greedy-steps-h2 shape, the top-10 of the
  ## recordings that predate it); the argmax margin derives from the support
  ## pair when the node does not state it, and the tail checksum runs only
  ## when the recording carries the tail probability.
  result.step = step
  result.chosenToken = node["chosen_token"].getInt()
  let idsNode = if node.hasKey("top32_ids"): node["top32_ids"]
    else: node["top10_tokens"]
  let logitsNode = if node.hasKey("top32_logits"): node["top32_logits"]
    else: node["top10_logits"]
  for el in idsNode:
    result.top32Ids.add el.getInt()
  for el in logitsNode:
    result.top32Logits.add float32(el.getFloat())
  if node.hasKey("argmax_margin"):
    result.argmaxMargin = node["argmax_margin"].getFloat()
  else:
    result.argmaxMargin = result.top32Logits[0].float64 - result.top32Logits[1].float64
  if node.hasKey("tail_probability"):
    result.tailProbability = node["tail_probability"].getFloat()
  else:
    result.tailRecorded = false

proc runChain(model: AnyModel, fixture: JsonNode): bool =
  ## Replay one greedy chain with per-step checks, the recorded prefix
  ## teacher-forced at every step. A tie flip keeps the chain aligned with
  ## the recording, and every following check must re-converge.
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
    F.kFloat16, device)
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
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  runCppTest "Qwen3-0.6B-EXL3-5bpw greedy decoding - prefix checks vs recorded chains":
    proc(): bool =
      echo "Loading model..."
      let model = loadModel($ModelPath, testDevice())
      echo "Model loaded."
      var passed = 0
      var total = 0
      for fixture in walkPattern($FixtureDir & "/*.json.zst"):
        inc total
        echo "Fixture: " & fixture
        if runChain(model, parseJson(zstdReadFixture(fixture))):
          inc passed
      echo "Greedy decoding: " & $passed & "/" & $total & " fixtures passed"
      result = passed == total

when isMainModule:
  main()
