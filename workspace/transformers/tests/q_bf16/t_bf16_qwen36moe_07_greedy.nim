# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --verbosity:0 --hints:off --warnings:off
##   --outdir:build/tests/t_bf16_qwen36moe_07_greedy
##   --nimcache:nimcache/tests/t_bf16_qwen36moe_07_greedy
##   workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_07_greedy.nim
#
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/os,
  std/math,
  std/importutils,
  std/strformat,
  std/sequtils,
  std/strutils,
  std/monotimes,
  std/times,
  pkg/jsony,
  workspace/libtorch as F,
  workspace/toktoktok,
  workspace/transformers/src/models,
  workspace/transformers/src/samplers,
  std/tables,
  workspace/safetensors/src/collections {.all.},
  workspace/transformers/tests/transformers_testutils,
  workspace/safetensors/src/safetensors_libtorch,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/stateful/orchestrator,
  workspace/libtorch_testutils

privateAccess(SafetensorsCollectionObj)

const
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  GreedyFixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "greedy-decoding" / "Qwen3.6-35B-A3B"
  GreedyFixtureFiles = [
    "Hello_how_are_you_32_steps.json.zip",
    "The_capital_of_France_is_32_steps.json.zip",
    "Big_blue_whales_eat_krill_32_steps.json.zip",
  ]
  RecordingSha = "36deb0b53ed0863f4b4dfdea23dcaec7f3df3701"
  # Recording environment of the committed fixtures. A regenerated artifact
  # from a foreign environment fails the provenance block, not the token
  # chains under the same filenames.
  TorchRecordingVersion = "2.11.0"
  TransformersRecordingVersion = "5.16.0.dev0"
  # Generator ceilings: prompts stay inside one GDN chunk, horizons
  # inside the recorded 32 greedy step ceiling.
  FixturePromptTokenCeiling = 12
  FixtureHorizonCeiling = 32
  # Decode context ceiling: one TokensPerPage page covers every prompt
  # chain comfortably above the prompt + horizon footprint.
  MaxContextLen = 256

type
  GreedyFixture = object
    schema: string
    model: string
    vendored_sha: string
    torch_version: string
    transformers_version: string
    experts_implementation: string
    attn_implementation: string
    dtype: string
    device: string
    num_threads: int
    prompt: string
    num_input_tokens: int
    max_new_tokens: int
    input_tokens: seq[int]
    generated_ids: seq[int]
    step_logits_checksum: seq[float64]

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

proc runGreedyChains*(device: DeviceKind = kCPU): int =
  ## Replay the greedy fixture chains on `device` through the wired 35B
  ## stack, one fresh Orchestrator per chain, comparing every token
  ## against the fixture. Returns the count of token-exact fixtures.
  ## Diverging chains raise `AssertionError`.
  let loadStart = getMonoTime()
  let model = loadQwen35MoeModelRaw(ModelDir, device)
  doAssert checkpointTensorRequests() == 693
  doAssert model.config.numHiddenLayers == 40
  echo &"    load wall {(getMonoTime() - loadStart).inNanoseconds.float64 * 1e-9:.3f} s"

  var tokenExactPrompts = 0
  for fixtureName in GreedyFixtureFiles:
    let meta = readFixture(GreedyFixtureDir / fixtureName).fromJson(GreedyFixture)
    let promptIds = meta.input_tokens
    let expected = meta.generated_ids
    let horizon = meta.max_new_tokens
    doAssert expected.len == horizon
    doAssert promptIds.len == meta.num_input_tokens
    echo &"Fixture: {fixtureName} ({promptIds.len} prompt tokens, {horizon} steps)"

    # Fresh decode state per prompt: the Orchestrator owns the paged KV
    # pool plus the GDN conv and SSM slots, so a new one carries no
    # residue of the previous chain.
    var orc = Orchestrator.init(
      model.config.numHiddenLayers, 1, model.config.numKeyValueHeads,
      MaxContextLen, model.config.headDim,
      computeNumPages(MaxContextLen, 1), F.kBFloat16, device)
    orc.startSequence(promptIds.mapIt(it.uint32))
    let inputIds = F.toTensor([promptIds]).to(device)
    let logits = model.forward(orc.getInferenceContextMut(), inputIds)
    # Greedy contract: argmax of the deciding logits row, no sampler
    # between. Same write-skips as generate(): kv_position advances
    # after the forward pass that consumed it.
    orc.setKvPosition(promptIds.len)
    # Chain: the prefill argmax is fixture generated_ids[0]. Every decode
    # step feeds back the previous pick and appends its own argmax.
    var nextToken =
      logits.narrow(1, promptIds.len - 1, 1).squeeze(1).squeeze(0).argmax().item(int)
    var nimTokens: seq[int] = @[]
    var ids = promptIds
    ids.add nextToken
    nimTokens.add nextToken
    let prefillChecksum = logits.narrow(1, promptIds.len - 1, 1)
      .squeeze(1).squeeze(0).to(F.kFloat32).sum().item(float64)
    echo "    step 0 logits f32 checksum: recorded " & $meta.step_logits_checksum[0] &
      ", observed " & $prefillChecksum & " (diagnostic, not asserted)"
    var finalChecksum = -1.0'f64
    let armStart = getMonoTime()
    while nimTokens.len < horizon:
      orc.appendToken(ids.len - 1, nextToken.uint32, device)
      let stepLogits = model.forward(
        orc.getInferenceContextMut(), F.toTensor([[nextToken]]).to(device))
      orc.setKvPosition(ids.len)
      let flatLogits = stepLogits.squeeze(0).squeeze(0)
      let nextTokenNew = flatLogits.argmax().item(int)
      if nimTokens.len == horizon - 1:
        finalChecksum = flatLogits.to(F.kFloat32).sum().item(float64)
      nimTokens.add nextTokenNew
      ids.add nextTokenNew
      nextToken = nextTokenNew
    let armWall = (getMonoTime() - armStart).inNanoseconds.float64 * 1e-9
    echo "    final logits f32 checksum: recorded " & $meta.step_logits_checksum[horizon - 1] &
      ", observed " & $finalChecksum & " (diagnostic, not asserted)"
    echo &"    arm wall {armWall:.3f} s ({horizon.float64 / armWall:.2f} tok/s)"

    var firstDiff = -1
    for i in 0 ..< horizon:
      if nimTokens[i] != expected[i]:
        firstDiff = i
        break
    if firstDiff < 0:
      inc tokenExactPrompts
      echo &"  ✅ token-exact: all {horizon} greedy steps match the fixture"
    else:
      echo &"  ❌ tokens diverge: step {firstDiff}: expected {expected[firstDiff]}, " &
        &"got {nimTokens[firstDiff]}"
      raise newException(AssertionError,
        "[greedy-test] gross token-chain divergence for " & fixtureName & " at step " &
        $firstDiff)
    echo ""

  result = tokenExactPrompts

proc main() =
  # Chains replay on the Metal Performance Shaders device, with a PyTorch
  # fallback for the kernels Metal does not implement.
  # The switch is process-wide, so set it before the first libtorch call.
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  runCppTest "greedy fixture provenance":
    proc(): bool =
      # Provenance block: schema, model name, recorded versions and checksums,
      # re-read and re-checked before any model load.
      for fixtureName in GreedyFixtureFiles:
        # jsony derives the parser: absent or wrong-typed required
        # fields raise at the read, assertions below see well-typed
        # values.
        let meta = readFixture(GreedyFixtureDir / fixtureName)
          .fromJson(GreedyFixture)
        doAssert meta.schema == "tt-qwen36-greedy-1"
        doAssert meta.model == "Qwen3.6-35B-A3B"
        doAssert meta.vendored_sha == RecordingSha
        doAssert meta.torch_version == TorchRecordingVersion
        doAssert meta.transformers_version == TransformersRecordingVersion
        doAssert meta.experts_implementation == "eager"
        doAssert meta.attn_implementation == "sdpa"
        doAssert meta.num_threads == 1
        doAssert meta.dtype == "bfloat16"
        doAssert meta.device == "cpu"

        let horizon = meta.max_new_tokens
        doAssert horizon > 0 and horizon <= FixtureHorizonCeiling,
          "horizon must stay inside the recorded generator ceiling"
        doAssert meta.input_tokens.len > 0 and
          meta.input_tokens.len <= FixturePromptTokenCeiling
        doAssert meta.input_tokens.len == meta.num_input_tokens
        doAssert meta.generated_ids.len == horizon
        doAssert meta.step_logits_checksum.len == horizon
      result = true

  runCppTest "loadModel dispatches the Qwen3.5-MoE architecture and generates":
    proc(): bool =
      # The registry path is the library's production entry point: loadModel
      # reads architectures[0] from the checkpoint config.json and dispatches
      # to the loader registered by qwen35_moe.nim.
      let model = loadModel($(ModelDir), kMPS)
      let cfg = model.getConfig()
      doAssert cfg.architecture == "Qwen3_5MoeForConditionalGeneration"
      doAssert cfg.model_type == "qwen3_5_moe_text"
      doAssert cfg.num_hidden_layers == 40
      doAssert cfg.vocab_size == 248320
      doAssert cfg.num_key_value_heads == 2
      doAssert cfg.head_dim == 256
      doAssert model.getDeviceKind() == kMPS
      doAssert not model.getTokenizer().isNil

      # Greedy must be deterministic argmax: sample() at temp 0 returns
      # logits.argmax() with no Gumbel noise. Lock that on a fixed logits
      # row so the token-exact comparison below cannot flake on sampler
      # randomness.
      let fixedLogits = F.toTensor([[1.0f, 2.5f, 0.3f, 4.0f, 3.2f]])
      doAssert sample(fixedLogits, 0.0f) == fixedLogits.argmax().item(int),
        "temp 0 must be deterministic argmax"

      # Short greedy continuation driven by `generate()` itself, replaying
      # the prefill and decode loop, token-exact against the recorded Hello
      # chain, because the fixture prompt encodes to the recorded input
      # tokens. The comparison re-encodes the generated text, then drops
      # the prompt ids from the front.
      const FrontDoorSteps = 8
      let meta = readFixture(
        GreedyFixtureDir / "Hello_how_are_you_32_steps.json.zip"
      ).fromJson(GreedyFixture)
      let prompt = meta.prompt
      let numPrompt = meta.num_input_tokens
      let expected = meta.generated_ids
      let output = model.generate(
        prompt, temp = 0.0f, maxTokens = FrontDoorSteps,
        maxContextLen = MaxContextLen)
      let actualIds = model.getTokenizer().encode(output)
      doAssert actualIds.len == numPrompt + FrontDoorSteps
      doAssert actualIds[numPrompt ..< actualIds.len] == expected[0 ..< FrontDoorSteps]
      true

  runCppTest "greedy chains through the wired 35B vs fixtures":
    proc(): bool =
      let tokenExactPrompts = runGreedyChains(kMPS)
      doAssert tokenExactPrompts == GreedyFixtureFiles.len,
        "every fixture chain must be token-exact"
      echo &"Greedy e2e: {tokenExactPrompts}/{GreedyFixtureFiles.len} fixtures token-exact"
      result = true

  echo "\nt_bf16_qwen36moe_07_greedy: all blocks PASS"

when isMainModule:
  main()
