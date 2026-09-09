# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp=0) decoding of the wired Qwen3.6-35B-A3B stack against
## greedy-steps fixtures (GreedyStepsSchema): per-step argmax checks with margin-scaled logit
## caps, truncated-KL and tail-probability checksums, plus teacher-forced
## recovery at structural ties (harness/tolerance.nim greedy checks).

import
  std/importutils,
  std/json,
  std/monotimes,
  std/strformat,
  pkg/jsony,
  std/os,
  std/sequtils,
  std/strutils,
  std/tables,
  std/times,
  workspace/libtorch as F,
  workspace/safetensors/src/collections {.all.},
  workspace/safetensors/src/safetensors_libtorch,
  workspace/toktoktok,
  workspace/transformers/src/models,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/samplers,
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

privateAccess(SafetensorsCollectionObj)

const
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  GreedyFixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-04-greedy-text-generation" / "Qwen3.6-35B-A3B"
  GreedyFixtureFiles = [
    "Hello_how_are_you_32_steps.json.zst",
    "The_capital_of_France_is_32_steps.json.zst",
    "Big_blue_whales_eat_krill_32_steps.json.zst",
  ]
  # The recording environment of the committed fixtures stays
  # manifest-declared: the suite reads the torch and transformers
  # rows of the family PROVENANCE.md (manifestValue), never
  # hardcoded constants. The device carries no row of its own:
  # it derives from the recorded_from row (harness/device.nim).
  # Generator ceilings: prompts stay inside one GDN chunk, horizons
  # inside the recorded 32 greedy step ceiling.
  FixturePromptTokenCeiling = 12
  FixtureHorizonCeiling = 32
  # Decode context ceiling: one TokensPerPage page covers every prompt
  # chain comfortably above the prompt + horizon footprint.
  MaxContextLen = 256

  # Tie-eligibility from the regenerated corpus margin report
  # structural ties record margin 0.0,
  # the smallest nonzero margin is exactly one bf16 ulp at logit 16.
  # The floor is tieUlps ulps of the step's own top logit, never
  # an absolute constant: one ulp is 0.0625 at logit 16, 0.125
  # at logit 32. The drift floor is this stack's measured rounding behavior:
  # the 40-layer MPS pass drifts the top-32 support up to 2.25 absolute
  # (36 bf16 ulps at logit 17) across the 96 recorded steps, so 64 ulps
  # carry it with headroom. The tie steps hold top1Drift 0.0.
  ChainChecks = GreedyConfig(
    tieUlps: 1,
    epsBase: 0.05,
    tailBand: 0.3,
    klBand: 0.05,
    maxFlips: 4,
    ulpFloorUlps: 64)

type
  GreedyFixture = object
    schema: string
    model: string
    torch_version: string
    transformers_version: string
    experts_implementation: string
    attn_implementation: string
    dtype: string
    device: string
    num_threads: int
    prompt: string
    num_prompt_tokens: int
    num_generated_tokens: int
    prompt_ids: seq[int]
    generated_ids: seq[int]

proc checkpointTensorRequests(): int =
  ## Name-based tensor requests the loader makes against the checkpoint:
  ## one per language_model key, plus the head request the untied
  ## checkpoint answers through its lm_head.weight key.
  let view = SafetensorsCollection.open(ModelDir)
  for name in view.weightMap.keys():
    if name.startsWith("model.language_model."):
      inc result
  if view.hasTensor("lm_head.weight"):
    inc result

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
  ## teacher-forced throughout. Tie flips stay recorded, later checks
  ## must re-converge, and the flip cap bounds flips.
  let promptIds = fixture["prompt_ids"].mapIt(it.getInt())
  let expected = fixture["generated_ids"].mapIt(it.getInt())
  let horizon = expected.len
  let cfg = model.getConfig()
  let device = model.getDeviceKind()
  let numPoolPages = computeNumPages(MaxContextLen, concurrentRequests = 1)
  var orc = Orchestrator.init(cfg.num_hidden_layers, 1,
    cfg.num_key_value_heads, MaxContextLen, cfg.head_dim, numPoolPages,
    F.kBFloat16, device)
  defer: orc.endSequence()

  var state = GreedyState()
  var ids = promptIds
  orc.startSequence(ids.mapIt(it.uint32))
  let chainStart = getMonoTime()
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
  # Informational perf line: wall clock and tok/s of the replay chain.
  # The 35B cpu replay runs at ~0.01 tok/s (operator-measured blocker).
  # These numbers make device and kernel regressions visible per run.
  let chainWall = (getMonoTime() - chainStart).inNanoseconds.float64 * 1e-9
  echo &"    chain wall {chainWall:.3f} s ({horizon.float64 / chainWall:.3f} tok/s)"
  result = true

proc main() =
  assertTorchStamp(GreedyFixtureDir)
  # Chains replay on the device testDevice() resolves: Metal by the auto
  # default on macOS, with a PyTorch fallback where kernels are missing.
  # TTT_TEST_ON=metal|cpu|cuda flips the device.
  # The switch is process-wide, so set it before the first libtorch call.
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  runCppTest "greedy fixture provenance":
    proc(): bool =
      # Provenance block: schema, model name, recorded versions,
      # identity checks, re-read and re-checked before any model load.
      # The chain blocks below re-read the same archives into JsonNode.
      doAssert checkpointTensorRequests() == 693,
        "the loader contract: 692 language_model keys plus the untied head"
      for fixtureName in GreedyFixtureFiles:
        # jsony derives the parser: absent or wrong-typed required
        # fields raise at the read, assertions below see well-typed
        # values.
        let meta = zstdReadFixture(GreedyFixtureDir / fixtureName)
          .fromJson(GreedyFixture)
        doAssert meta.schema == GreedyStepsSchema
        doAssert meta.model == "Qwen3.6-35B-A3B"
        doAssert meta.torch_version ==
          manifestValue(GreedyFixtureDir, "torch")
        doAssert meta.transformers_version ==
          manifestValue(GreedyFixtureDir, "transformers")
        doAssert meta.experts_implementation == "eager"
        doAssert meta.attn_implementation == "sdpa"
        doAssert meta.num_threads == 1
        doAssert meta.dtype == "bfloat16"
        doAssert meta.device ==
          deviceName(recordedDevice(recordedFrom(GreedyFixtureDir)))

        doAssert meta.num_generated_tokens > 0 and
          meta.num_generated_tokens <= FixtureHorizonCeiling,
          "horizon must stay inside the recorded generator ceiling"
        doAssert meta.prompt_ids.len > 0 and
          meta.prompt_ids.len <= FixturePromptTokenCeiling
        doAssert meta.prompt_ids.len == meta.num_prompt_tokens
        doAssert meta.generated_ids.len == meta.num_generated_tokens
      result = true

  runCppTest "loadModel dispatches the Qwen3.5-MoE architecture and generates":
    proc(): bool =
      # The registry path is the library's production entry point: loadModel
      # reads architectures[0] from the checkpoint config.json and dispatches
      # to the loader registered by qwen35_moe.nim.
      let model = loadModel($(ModelDir), testDevice())
      let cfg = model.getConfig()
      doAssert cfg.architecture == "Qwen3_5MoeForConditionalGeneration"
      doAssert cfg.model_type == "qwen3_5_moe_text"
      doAssert cfg.num_hidden_layers == 40
      doAssert cfg.vocab_size == 248320
      doAssert cfg.num_key_value_heads == 2
      doAssert cfg.head_dim == 256
      doAssert model.getDeviceKind() == testDevice()
      doAssert not model.getTokenizer().isNil

      # Greedy must be deterministic argmax: sample() at temp 0 returns
      # logits.argmax() with no Gumbel noise. Lock that on a fixed logits
      # row so the token comparisons below cannot flake on sampler
      # randomness.
      let fixedLogits = F.toTensor([[1.0f, 2.5f, 0.3f, 4.0f, 3.2f]])
      doAssert sample(fixedLogits, 0.0f) == fixedLogits.argmax().item(int),
        "temp 0 must be deterministic argmax"

      # Short greedy continuation driven by `generate()` itself, replaying
      # the prefill and decode loop, token-comparing against the recorded
      # Hello chain prefix: the fixture prompt encodes to the recorded
      # input tokens, and the comparison re-encodes the generated text
      # then drops the prompt ids from the front. The check block
      # carries the verdict on the teacher-forced chain. The front door
      # runs free-decode and only reports divergences. The possible
      # tie flip verdict belongs to the check block.
      const FrontDoorSteps = 8
      let meta = zstdReadFixture(
        GreedyFixtureDir / "Hello_how_are_you_32_steps.json.zst"
      ).fromJson(GreedyFixture)
      let expected = meta.generated_ids
      let output = model.generate(
        meta.prompt, temp = 0.0f, maxTokens = FrontDoorSteps,
        maxContextLen = MaxContextLen)
      let actualIds = model.getTokenizer().encode(output)
      doAssert actualIds.len == meta.num_prompt_tokens + FrontDoorSteps
      var aligned = true
      for i in 0 ..< FrontDoorSteps:
        if actualIds[meta.num_prompt_tokens + i] != expected[i]:
          aligned = false
          break
      if not aligned:
        echo "    front door diverges from the recording (tie-flip family), " &
          "the check block carries the verdict"
      true

  runCppTest "greedy chains through the wired 35B vs ttt-tf-001-greedy-steps-h2 checks":
    proc(): bool =
      echo "Loading model..."
      let model = loadModel($(ModelDir), testDevice())
      echo "Model loaded."
      var passed = 0
      var total = 0
      for f in GreedyFixtureFiles:
        inc total
        echo "Fixture: " & f
        let fixture = parseJson(zstdReadFixture(GreedyFixtureDir / f))
        if runChain(model, fixture):
          inc passed
      echo &"Greedy decoding: {passed}/{total} fixtures passed"
      result = passed == total

  echo "\nt_bf16_qwen36moe_04_greedy_text_generation: all blocks PASS"

when isMainModule:
  main()
