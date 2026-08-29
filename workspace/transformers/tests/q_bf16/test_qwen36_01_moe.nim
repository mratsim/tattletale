# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --verbosity:0 --hints:off --warnings:off \
##   --outdir:build/tests/test_qwen36_01_moe --nimcache:nimcache/tests/test_qwen36_01_moe \
##   workspace/transformers/tests/q_bf16/test_qwen36_01_moe.nim
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/importutils,
  std/memfiles,
  std/options,
  std/os,
  std/strutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/mixtures_of_experts,
  workspace/transformers/src/layers/mlp,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/model/loading/layer_kinds,
  workspace/transformers/src/model/loading/generation_config,
  workspace/transformers/src/safetensors/collection,
  workspace/transformers/tests/transformers_testutils,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen35MoeModel)
privateAccess(LMHead)
privateAccess(Embedding)

const ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
const Shard1 = ModelDir / "model-00001-of-00026.safetensors"
const Shard2 = ModelDir / "model-00002-of-00026.safetensors"
const FixturePath = currentSourcePath().parentDir() / ".." / "fixtures" / "qwen36-moe" /
  "moe_layer0_fixture.json"

const Tokens = 6
const TopK = 8
const NumExperts = 256
const RoutedWidth = 512
const Hidden = 2048

proc matrixFromJson(node: JsonNode): Tensor =
  ## [rows, cols] fp32 tensor from a JSON matrix: a list of equal-length
  ## numeric rows. A missing key arrives as JNull, whose zero rows fail
  ## the shape guards downstream.
  let rows = node.len
  var flat = newSeq[float32]()
  var cols = 0
  for i in 0 ..< rows:
    let row = node[i]
    cols = row.len
    for j in 0 ..< cols:
      let cell = row[j]
      flat.add cell.getFloat().float32
  flat.toTensor().reshape(rows, cols)

proc indicesFromJson(node: JsonNode): Tensor =
  ## [rows, cols] int64 tensor of expert ids from a JSON matrix of ints.
  let rows = node.len
  var flat = newSeq[int64]()
  var cols = 0
  for i in 0 ..< rows:
    let row = node[i]
    cols = row.len
    for j in 0 ..< cols:
      let cell = row[j]
      flat.add cell.getBiggestInt()
  flat.toTensor().reshape(rows, cols)

proc vectorFromJson(node: JsonNode): Tensor =
  ## [n] fp32 tensor from a flat JSON array of numbers.
  var flat = newSeq[float32]()
  for i in 0 ..< node.len:
    let cell = node[i]
    flat.add cell.getFloat().float32
  flat.toTensor()

# The `text_config` keys the parser reads strictly, at the values the checkpoint
# spells. A synthetic config built from them reaches the gate a block names,
# instead of tripping an earlier absent-key read.
const StrictTextKeys = [
  ("vocab_size", "248320"),
  ("hidden_size", "2048"),
  ("num_hidden_layers", "2"),
  ("num_attention_heads", "16"),
  ("num_key_value_heads", "2"),
  ("head_dim", "256"),
  ("num_experts", "256"),
  ("num_experts_per_tok", "8"),
  ("moe_intermediate_size", "512"),
  ("shared_expert_intermediate_size", "512"),
  ("linear_num_key_heads", "16"),
  ("linear_key_head_dim", "128"),
  ("linear_num_value_heads", "32"),
  ("linear_value_head_dim", "128"),
  ("linear_conv_kernel_dim", "4"),
  ("rms_norm_eps", "1e-06"),
  ("max_position_embeddings", "262144"),
  ("eos_token_id", "248044"),
  ("layer_types", """["linear_attention", "full_attention"]"""),
]

const MlpOnlyLayers = @[0, 8]

proc appended(key, value: string): string =
  ## One `"key": value` entry for a name outside `StrictTextKeys`. An empty
  ## value becomes `null` rather than nothing after the colon: `"key": ` alone
  ## is not JSON, and a `JsonParsingError` would replace the gate a block
  ## targets with a message about the builder.
  '"' & key & "\": " & (if value.len > 0: value else: "null")

proc textConfigJson(overrideKey = "", overrideValue = "",
                    extraKey = "", extraValue = ""): string =
  ## `text_config` with every strictly-read key at its checkpoint value.
  ## `overrideKey` takes `overrideValue` instead, and an empty `overrideValue`
  ## drops a table key entirely. `extraKey` is spliced the same way, so one
  ## call can spell two keys. A name outside the table goes to `appended`.
  result = "{"
  var sep = ""
  var applied = overrideKey.len == 0
  var appliedExtra = extraKey.len == 0
  for (key, value) in StrictTextKeys:
    var kept = value
    if key == overrideKey:
      kept = overrideValue
      applied = true
    elif key == extraKey:
      kept = extraValue
      appliedExtra = true
    if kept.len > 0:
      result.add sep & '"' & key & "\": " & kept
      sep = ", "
  if not applied:
    result.add sep & appended(overrideKey, overrideValue)
    sep = ", "
  if not appliedExtra:
    result.add sep & appended(extraKey, extraValue)
  result.add "}"

proc configJson(textConfig: string): string =
  ## The `qwen3_5_moe` wrapper object around a `text_config` body.
  """{"architectures": ["Qwen3_5MoeForConditionalGeneration"], "text_config": """ &
    textConfig & "}"

proc configError(jsonText: string): string =
  ## The `ValueError` message a config parse produces, empty when the parse
  ## succeeds. A defect of another kind escapes, which fails the block.
  result = ""
  try:
    discard parseQwen35MoeConfig(parseJson(jsonText))
  except ValueError as e:
    result = e.msg

proc generationError(jsonText: string): string =
  ## The `ValueError` message a generation-config parse produces. Empty result
  ## means the parse succeeded.
  result = ""
  try:
    discard parseGenerationConfig(parseJson(jsonText))
  except ValueError as e:
    result = e.msg

proc expectValueError(msg, key, what: string) =
  ## A guard contract: the raise must happen, and its message must name `key`.
  ## An empty message means the bad input was accepted, which fails here.
  doAssert msg.len > 0, what & " was accepted"
  doAssert key in msg, what & " raised without naming " & key & ": " & msg

proc main() =


  runCppTest "Layer kinds: wrong length, unmapped and non-string spellings, zero count":
    proc(): bool =
      # Every strictly-read key comes from the config builder, so each parse
      # reaches the layer-types defect that the block targets.
      expectValueError configError(configJson(textConfigJson("num_hidden_layers", "40"))),
        "layer_types", "a 2-entry layer_types for 40 layers"
      let msg = configError(configJson(textConfigJson("num_hidden_layers", "40")))
      doAssert "40" in msg
      doAssert "layer_types has 2 entries, expected one attention kind per layer of the " &
        "40-layer stack" in msg

      # An unmapped spelling dies at the parse bound, naming the raw text
      # and its config path, with the mapped spellings in the raise message.
      let kindMsg = configError(configJson(
        textConfigJson("layer_types", """["linear_attention", "sliding_attention"]""")))
      expectValueError kindMsg, "sliding_attention", "a spelling outside the mapped table"
      doAssert "text_config.layer_types[1]" in kindMsg
      doAssert "linear_attention" in kindMsg and "full_attention" in kindMsg

      # Direct parse arms: exact roundtrip per mapped spelling, and the raise
      # naming which raw text, at which source path, is unknown.
      doAssert parseAttnFromHfTransformers("linear_attention", "pin") == alkGatedDeltaNet
      doAssert parseAttnFromHfTransformers("full_attention", "pin") == alkAttention
      var raw = ""
      try:
        discard parseAttnFromHfTransformers("sliding_attention", "text_config.layer_types[9]")
      except ValueError as e:
        raw = e.msg
      doAssert raw.startsWith("[ttt]")
      doAssert "sliding_attention" in raw
      doAssert "text_config.layer_types[9]" in raw
      doAssert "full_attention" in raw and "linear_attention" in raw

      # A non-string entry dies at the caller loop's JString gate, quoting
      # the array path with the entry index.
      let typeMsg = configError(configJson(
        textConfigJson("layer_types", """["linear_attention", 3]""")))
      expectValueError typeMsg, "expected a string", "a non-string layer_types entry"
      doAssert "text_config.layer_types[1]" in typeMsg

      # A zero layer count refuses at the positive-count
      # `num_hidden_layers` read, before the per-layer list is compared.
      let zeroMsg = configError(configJson(textConfigJson("num_hidden_layers", "0")))
      expectValueError zeroMsg, "num_hidden_layers", "a zero layer count"
      doAssert "expected a positive value, found 0" in zeroMsg

      true

  runCppTest "Rope reads: rope_theta prefers rope_parameters, factor text_config":
    proc(): bool =
      # No `rope_parameters`, only the text-level factor.
      let textLevel = parseQwen35MoeConfig(parseJson(
        configJson(textConfigJson("partial_rotary_factor", "0.5"))))
      doAssert textLevel.partialRotaryFactor == 0.5
      doAssert textLevel.ropeTheta == 10000000.0
      doAssert textLevel.ropeType == "default"

      # Neither level spells the keys, so the last-resort pair answers.
      let bare = parseQwen35MoeConfig(parseJson(configJson(textConfigJson())))
      doAssert bare.partialRotaryFactor == 0.25
      doAssert bare.ropeTheta == 10000000.0

      # Both levels spell both keys, and the levels differ. `rope_theta` keeps
      # the `rope_parameters` entry while the factor takes the `text_config`
      # one, the asymmetry of `modeling_rope_utils.py:786` and `:788`.
      # Qwen3.6-35B-A3B spells both levels `0.25`, equal values, so the two orders cannot diverge on this checkpoint.
      let both = parseQwen35MoeConfig(parseJson(configJson(textConfigJson(
        "partial_rotary_factor", "0.5", "rope_parameters",
        """{"partial_rotary_factor": 0.75, "rope_theta": 1234567.0}"""))))
      doAssert both.partialRotaryFactor == 0.5
      doAssert both.ropeTheta == 1234567.0

      true

  runCppTest "Absent, wrong-typed and non-positive keys raise, naming the key":
    proc(): bool =
      expectValueError configError(configJson(textConfigJson("hidden_size", ""))),
        "hidden_size", "a config with no hidden_size"
      expectValueError configError(configJson(textConfigJson("moe_intermediate_size", ""))),
        "moe_intermediate_size", "a config with no moe_intermediate_size"
      expectValueError configError(configJson(textConfigJson("rms_norm_eps", ""))),
        "rms_norm_eps", "a config with no rms_norm_eps"
      expectValueError configError(configJson(textConfigJson("num_key_value_heads", "0"))),
        "num_key_value_heads", "a config with zero KV heads"
      expectValueError configError(configJson(textConfigJson("hidden_size", "2048.0"))),
        "hidden_size", "a float-spelled hidden_size"
      expectValueError configError(configJson(textConfigJson("pad_token_id", "\"248044\""))),
        "pad_token_id", "a string-spelled pad_token_id"
      expectValueError configError(configJson(textConfigJson("max_position_embeddings", ""))),
        "max_position_embeddings", "a config with no max_position_embeddings"
      expectValueError configError(configJson(textConfigJson("eos_token_id", "null"))),
        "text_config.eos_token_id", "a null text-level eos"
      expectValueError configError("""{"architectures": [], "text_config": {}}"""),
        "architectures", "a config with an empty architectures list"
      expectValueError configError("""{"text_config": {}}"""),
        "architectures", "a config with no architectures"
      expectValueError configError(
        """{"architectures": ["Qwen3_5MoeForConditionalGeneration"]}"""),
        "text_config", "a config with no text_config"

      # A missing stop id is one of three spellings that reach the empty-set
      # gate, and the message names the key the file should carry.
      expectValueError generationError("{}"), "eos_token_id", "a generation file with no eos"
      expectValueError generationError("""{"eos_token_id": []}"""),
        "eos_token_id", "a generation file with an empty eos list"
      let nullElem = generationError("""{"eos_token_id": [248046, null]}""")
      expectValueError nullElem, "eos_token_id", "a null element inside the eos list"
      doAssert "[1]" in nullElem
      expectValueError generationError(
        """{"eos_token_id": [248046], "pad_token_id": "0"}"""),
        "pad_token_id", "a string-spelled generation pad_token_id"

      # The scalar spelling stays the documented one-element stop set.
      let scalar = parseGenerationConfig(parseJson("""{"eos_token_id": 248046}"""))
      doAssert scalar.eosTokenIds.len == 1
      doAssert scalar.eosTokenIds[0] == 248046

      # A float-spelled token id is a wrong type, not an absence: `optInt`
      # refuses it instead of answering 0.
      expectValueError configError(configJson(textConfigJson("bos_token_id", "1.0"))),
        "bos_token_id", "a float-spelled bos_token_id"

      # A null pad token stays absence in a generation file, never the id 0.
      let nullPad = parseGenerationConfig(
        parseJson("""{"eos_token_id": 248046, "pad_token_id": null}"""))
      doAssert nullPad.padTokenId.isNone

      # `mlp_only_layers` holds ints, so an int list arrives as ints rather
      # than as empty strings.
      let mlpOnly = parseQwen35MoeConfig(parseJson(
        configJson(textConfigJson("mlp_only_layers", "[0, 8]"))))
      doAssert mlpOnly.mlpOnlyLayers == MlpOnlyLayers

      true

  # Shard-by-name access below is a suite-only convenience: the upcoming
  # loader work replaces it with the index-backed weight_map view.

  runCppTest "Real layer-0 MoE tensors: ranks and shapes":
    proc(): bool =
      var memFile1 = memFiles.open(Shard1, mode = fmRead)
      var memFile2 = memFiles.open(Shard2, mode = fmRead)
      defer:
        close(memFile1)
        close(memFile2)
      let st1 = safetensors.load(memFile1)
      let st2 = safetensors.load(memFile2)

      let gateUpProj = st1.getTensorOwned(
        "model.language_model.layers.0.mlp.experts.gate_up_proj", kCPU)
      let downProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.experts.down_proj", kCPU)
      let routerWeight = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.gate.weight", kCPU)
      let sharedGateWeight = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert_gate.weight", kCPU)
      let sharedGateProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", kCPU)
      let sharedUpProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", kCPU)
      let sharedDownProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight", kCPU)

      # Fused gate/up weight of the routed bodies: gate rows 0:I, up rows I:2I
      doAssert gateUpProj.dim() == 3
      doAssert gateUpProj.size(0) == NumExperts
      doAssert gateUpProj.size(1) == 2 * RoutedWidth
      doAssert gateUpProj.size(2) == Hidden
      doAssert downProj.dim() == 3
      doAssert downProj.size(0) == NumExperts
      doAssert downProj.size(1) == Hidden
      doAssert downProj.size(2) == RoutedWidth
      doAssert routerWeight.dim() == 2
      doAssert routerWeight.size(0) == NumExperts
      doAssert routerWeight.size(1) == Hidden
      doAssert sharedGateWeight.dim() == 2
      doAssert sharedGateWeight.size(0) == 1
      doAssert sharedGateWeight.size(1) == Hidden
      doAssert sharedGateProj.dim() == 2
      doAssert sharedGateProj.size(0) == RoutedWidth
      doAssert sharedGateProj.size(1) == Hidden
      doAssert sharedUpProj.dim() == 2
      doAssert sharedUpProj.size(0) == RoutedWidth
      doAssert sharedUpProj.size(1) == Hidden
      doAssert sharedDownProj.dim() == 2
      doAssert sharedDownProj.size(0) == Hidden
      doAssert sharedDownProj.size(1) == RoutedWidth
      true

  runCppTest "Routed block forward vs fixture within recorded bands":
    proc(): bool =
      var memFile1 = memFiles.open(Shard1, mode = fmRead)
      var memFile2 = memFiles.open(Shard2, mode = fmRead)
      defer:
        close(memFile1)
        close(memFile2)
      let st1 = safetensors.load(memFile1)
      let st2 = safetensors.load(memFile2)

      let gateUpProj = st1.getTensorOwned(
        "model.language_model.layers.0.mlp.experts.gate_up_proj", kCPU)
      let downProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.experts.down_proj", kCPU)
      let routerWeight = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.gate.weight", kCPU)
      let sharedGateWeight = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert_gate.weight", kCPU)
      let sharedGateProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", kCPU)
      let sharedUpProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", kCPU)
      let sharedDownProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight", kCPU)

      let experts = MixtureOfExperts.init(gateUpProj, downProj)
      let sharedExpert = GatedMLP.init(
        sharedGateProj, sharedUpProj, sharedDownProj)

      let fixture = parseFile(FixturePath)
      let cfg = new Qwen35MoeConfig
      cfg.numExpertsPerTok = fixture{"meta", "num_experts_per_tok"}.getInt()

      let h = matrixFromJson(fixture{"h"}).to(kBfloat16)
      let moe = sparseMoeForward(
        cfg.numExpertsPerTok, h, routerWeight, experts, sharedExpert, sharedGateWeight)

      # Bands the generator recorded from bf16 ulp arithmetic
      let outputBand = fixture{"bands", "output_band"}.getFloat()
      let routerBand = fixture{"bands", "router_logits_band"}.getFloat()
      let weightsBand = fixture{"bands", "routing_weights_band"}.getFloat()
      let gateBand = fixture{"bands", "shared_gate_band"}.getFloat()

      let fixtureOutput = matrixFromJson(fixture{"moe_output"})
      let fixtureLogits = matrixFromJson(fixture{"router_logits"})
      let fixtureWeights = matrixFromJson(fixture{"routing_weights"})
      let fixtureGate = vectorFromJson(fixture{"shared_gate"}).reshape(Tokens, 1)

      doAssert maxAbsDiff(moe.output, fixtureOutput) <= outputBand
      doAssert maxAbsDiff(moe.routerLogits, fixtureLogits) <= routerBand
      doAssert maxAbsDiff(moe.routingWeights, fixtureWeights) <= weightsBand
      doAssert maxAbsDiff(moe.sharedGate, fixtureGate) <= gateBand

      # The cast and the fp32 renorm are asserted separately, a bf16 renorm
      # is borderline within the post-cast band
      doAssert moe.routingWeights.scalarType() == kBfloat16,
        "routing weights must be cast to the hidden-state dtype"
      doAssert moe.renormValues.scalarType() == kFloat32,
        "renorm values must stay fp32 before the dtype cast"
      doAssert maxAbsDiff(moe.renormValues, matrixFromJson(fixture{"renorm_values"})) <=
        2.0 * weightsBand,
        "fp32 renorm outside recorded band"

      # Smallest top-8 margin and smallest adjacent gap are both positive:
      # the top-8 order is uniquely determined by value
      doAssert fixture{"margins", "topk_margin_min"}.getFloat() > 0.0
      doAssert fixture{"margins", "topk_inner_gap_min"}.getFloat() > 0.0

      let fixtureIndices = indicesFromJson(fixture{"topk_indices"})
      for tok in 0 ..< Tokens:
        for pos in 0 ..< TopK:
          let nimIdx = moe.topkIndices[tok, pos]
          let fixIdx = fixtureIndices[tok, pos]
          doAssert nimIdx.item(int64) == fixIdx.item(int64),
            "topk index mismatch at [" & $tok & ", " & $pos & "]"
      true

  runCppTest "tied synthetic checkpoint loads, head derives from the embedding":
    proc(): bool =
      # A tied checkpoint ships no lm_head entry: the loader builds a head
      # from the embedding tensor instead of demanding an absent file.
      # Committed fixture: tests/fixtures/qwen36-tied, emitted by gen_qwen36_tied_synthetic.py.
      let dir = currentSourcePath().parentDir() / ".." / "fixtures" / "qwen36-tied" / "tied"
      let model = loadQwen35MoeModelRaw(dir)

      # The loaded count squares for the tied lane: all 17 keys,
      # language_model only, no lm_head entry, no head request.
      var tied = openSafetensorsCollection(dir)
      doAssert tied.tensorCount == 17
      doAssert not tied.hasTensor(LmHeadKey)
      close(tied)
      doAssert model.loadedTensorCount == 17

      # Head derivation from the embedding tensor: the tie branch stores
      # the embedding reference itself, so both weights are one storage.
      doAssert model.lmHead.tied
      doAssert model.lmHead.tied_embedding == model.embedTokens

      # Bit-exact derivation: the head logits equal the embedding-matrix
      # logits over the same weight storage, on a bf16 input.
      let hidden = F.zeros(1, 1, model.config.hiddenSize).to(F.kBFloat16)
      let logits = model.lmHead.forward(hidden)
      let embeddingLogits = F.linear(hidden, model.embedTokens.weight)
      doAssert logits.equal(embeddingLogits)
      true

  runCppTest "tie declared false with no lm_head entry raises through the loader":
    proc(): bool =
      # The untied branch keeps its loud refusal checkpoint-wide: with no
      # lm_head entry plus tie_word_embeddings false, a head would silently
      # become the embedding, so the load refuses.
      let dir = currentSourcePath().parentDir() / ".." / "fixtures" / "qwen36-tied" / "untied"
      try:
        discard loadQwen35MoeModelRaw(dir)
        doAssert false, "tie declared false with no lm_head entry was accepted"
      except ValueError as err:
        doAssert err.msg.startsWith("[ttt]")
        doAssert "tie_word_embeddings" in err.msg
        doAssert "lm_head.weight" in err.msg
      true

when isMainModule:
  main()
