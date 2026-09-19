# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  std/os,
  std/strutils,
  std/tables,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/toktoktok,
  ../layers,
  ../deserialization,
  ../models/loading/config_json,
  ../models/loading/generation_config,
  ../models/loading/layer_kinds,
  ../stateful/inference_context,
  workspace/safetensors/src/collections,
  ./all_interfaces

{.experimental: "callOperator".}


type
  Qwen35MoeConfig* = ref object
    architecture*: string
    wrapperModelType*: string
    modelType*: string
    transformersVersion*: string

    vocabSize*: int
    hiddenSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int
    numKeyValueHeads*: int
    headDim*: int
    intermediateSize*: Option[int]
    mlpOnlyLayers*: seq[int]

    # Routed block
    numExperts*: int
    numExpertsPerTok*: int
    moeIntermediateSize*: int
    sharedExpertIntermediateSize*: int
    outputRouterLogits*: bool
    routerAuxLossCoef*: float64

    # Gated DeltaNet block
    linearNumKeyHeads*: int
    linearKeyHeadDim*: int
    linearNumValueHeads*: int
    linearValueHeadDim*: int
    linearConvKernelDim*: int
    mambaSsmDtype*: string

    # Full-attention layers
    fullAttentionInterval*: int
    attnOutputGate*: bool
    layerTypes*: seq[AttentionLayerKind]

    # Numerics and position handling
    rmsNormEps*: float64
    hiddenAct*: string
    maxPositionEmbeddings*: int
    dtype*: string
    tieWordEmbeddings*: bool
    ropeType*: string
    ropeTheta*: float64
    partialRotaryFactor*: float64
    mropeInterleaved*: bool
    mropeSection*: seq[int]

    # Draft-block footprint
    mtpNumHiddenLayers*: int
    mtpUseDedicatedEmbeddings*: bool

    # Token ids
    bosTokenId*: Option[int]
    textEosTokenId*: int
    padTokenId*: Option[int]
    imageTokenId*: int
    videoTokenId*: int

################################################################################
#                          Qwen3.5 MoE Parsing                                  #
################################################################################


proc parseQwen35MoeConfig(json: JsonNode): Qwen35MoeConfig =
  let textCfg = json{"text_config"}
  checkValue(textCfg.kind == JObject,
    "[ttt] Qwen35MoeConfig.parse: text_config is missing or not an object, found " &
    $textCfg.kind)
  let ropeParams = textCfg{"rope_parameters"}

  let archs = json{"architectures"}
  checkValue(archs.kind == JArray and archs.len != 0,
    "[ttt] No architectures found in config.json")

  result = new Qwen35MoeConfig
  result.architecture = archs[0].getStr()
  result.wrapperModelType = json{"model_type"}.getStr()
  result.transformersVersion = json{"transformers_version"}.getStr("")
  result.imageTokenId = json{"image_token_id"}.getInt().int
  result.videoTokenId = json{"video_token_id"}.getInt().int

  result.modelType = textCfg{"model_type"}.getStr()
  result.vocabSize = textCfg{"vocab_size"}.reqPosInt("vocab_size")
  result.hiddenSize = textCfg{"hidden_size"}.reqPosInt("hidden_size")
  result.numHiddenLayers = textCfg{"num_hidden_layers"}.reqPosInt("num_hidden_layers")
  result.numAttentionHeads = textCfg{"num_attention_heads"}.reqPosInt("num_attention_heads")
  result.numKeyValueHeads = textCfg{"num_key_value_heads"}.reqPosInt("num_key_value_heads")
  result.headDim = textCfg{"head_dim"}.reqPosInt("head_dim")
  result.intermediateSize = textCfg{"intermediate_size"}.optInt("intermediate_size")
  result.mlpOnlyLayers = textCfg{"mlp_only_layers"}.parseIntList("mlp_only_layers")

  result.numExperts = textCfg{"num_experts"}.reqPosInt("num_experts")
  result.numExpertsPerTok = textCfg{"num_experts_per_tok"}.reqPosInt("num_experts_per_tok")
  result.moeIntermediateSize =
    textCfg{"moe_intermediate_size"}.reqPosInt("moe_intermediate_size")
  result.sharedExpertIntermediateSize = textCfg.reqSharedExpertIntermediateSize()
  result.outputRouterLogits = textCfg{"output_router_logits"}.getBool()
  result.routerAuxLossCoef = textCfg{"router_aux_loss_coef"}.getFloat()

  result.linearNumKeyHeads =
    textCfg{"linear_num_key_heads"}.reqPosInt("linear_num_key_heads")
  result.linearKeyHeadDim =
    textCfg{"linear_key_head_dim"}.reqPosInt("linear_key_head_dim")
  result.linearNumValueHeads =
    textCfg{"linear_num_value_heads"}.reqPosInt("linear_num_value_heads")
  result.linearValueHeadDim =
    textCfg{"linear_value_head_dim"}.reqPosInt("linear_value_head_dim")
  result.linearConvKernelDim =
    textCfg{"linear_conv_kernel_dim"}.reqPosInt("linear_conv_kernel_dim")
  result.mambaSsmDtype = textCfg{"mamba_ssm_dtype"}.getStr("float32")

  result.fullAttentionInterval = textCfg{"full_attention_interval"}.getInt(4)
  result.attnOutputGate = textCfg{"attn_output_gate"}.getBool(false)
  result.layerTypes = newSeq[AttentionLayerKind]()
  let rawKinds = textCfg{"layer_types"}
  if rawKinds.kind == JArray:
    for i in 0 ..< rawKinds.len:
      let elem = rawKinds[i]
      checkValue(elem.kind == JString,
        "[ttt] text_config.layer_types[" & $i & "]: expected a string, found " & $elem.kind)
      result.layerTypes.add parseAttnFromHfTransformers(
        elem.getStr(), "text_config.layer_types[" & $i & "]")

  result.rmsNormEps = textCfg{"rms_norm_eps"}.reqPosFloat("rms_norm_eps")
  result.hiddenAct = textCfg{"hidden_act"}.getStr()
  result.maxPositionEmbeddings =
    textCfg{"max_position_embeddings"}.reqPosInt("max_position_embeddings")
  result.dtype = textCfg{"dtype"}.getStr("bfloat16")
  result.tieWordEmbeddings = textCfg{"tie_word_embeddings"}.getBool()

  result.ropeType = ropeParams{"rope_type"}.getStr("default")
  # Each key must be found at one of the two levels: a key absent or null
  # at both raises `ValueError` naming it, a wrong-typed or non-positive
  # value at the winning level raises too. `rope_theta` prefers `rope_parameters`,
  # `partial_rotary_factor` prefers `text_config`.
  let ropeThetaAtRope = ropeParams{"rope_theta"}
  result.ropeTheta =
    if ropeThetaAtRope.kind != JNull:
      ropeThetaAtRope.reqPosFloat("rope_theta")
    else:
      textCfg{"rope_theta"}.reqPosFloat("rope_theta")
  let rotaryFactorAtText = textCfg{"partial_rotary_factor"}
  result.partialRotaryFactor =
    if rotaryFactorAtText.kind != JNull:
      rotaryFactorAtText.reqPosFloat("partial_rotary_factor")
    else:
      ropeParams{"partial_rotary_factor"}.reqPosFloat("partial_rotary_factor")
  result.mropeInterleaved = ropeParams{"mrope_interleaved"}.getBool(false)
  result.mropeSection = ropeParams{"mrope_section"}.parseIntList("mrope_section")

  result.mtpNumHiddenLayers = textCfg{"mtp_num_hidden_layers"}.getInt()
  result.mtpUseDedicatedEmbeddings =
    textCfg{"mtp_use_dedicated_embeddings"}.getBool(false)

  result.bosTokenId = textCfg{"bos_token_id"}.optInt("text_config.bos_token_id")
  result.textEosTokenId = textCfg{"eos_token_id"}.reqInt(
    "text_config.eos_token_id",
    "the stop set a generator uses lives in generation_config.json")
  result.padTokenId = textCfg{"pad_token_id"}.optInt("text_config.pad_token_id")

  checkValue(result.layerTypes.len == result.numHiddenLayers,
    "[ttt] Qwen35MoeConfig: layer_types has " & $result.layerTypes.len &
    " entries, expected one attention kind per layer of the " &
    $result.numHiddenLayers & "-layer stack")


proc loadQwen35MoeConfig(path: string): Qwen35MoeConfig =
  parseFile(path).parseQwen35MoeConfig()


################################################################################
#                     Qwen3.5 MoE Model Assembly                               #
################################################################################

type
  Qwen35MoeGdnDecoderLayer = DecoderLayer[GatedDeltaNet[perHead, FullRankGateIn, GateForm.softplus], GatedBlockSparseFFN, RmsNormOne]

  Qwen35MoeAttnDecoderLayer = DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedBlockSparseFFN, RmsNormOne]

  Qwen35MoeModel* = ref object
    embedTokens: Embedding
    layers: seq[AnyDecoderLayer]
    norm: RmsNormOne
    lmHead: LMHead
    rotary: RotaryPositionEmbedding
    config*: Qwen35MoeConfig
    tokenizer*: BPETokenizer
    device*: DeviceKind
                             ## entry when the checkpoint is untied

proc forward*(self: Qwen35MoeModel, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  var x = self.embedTokens.forward(input_ids)

  ctx.setRopeForPositions(self.rotary)

  var residual: Option[Tensor]
  for layer in self.layers:
    let layerOut = layer(ctx, x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])

  let finalResidual = residual.get(x)
  let normed = self.norm.forward(x + finalResidual)
  self.lmHead.forward(normed)

proc getConfig(self: Qwen35MoeModel): ModelConfigBase =
  ## Minimal config behind the `generate()` entry point, consumed when it
  ## builds the InferenceContext and the Orchestrator.
  ## `intermediate_size` holds `moe_intermediate_size`, the per-expert width
  ## of the routed FFN, because this checkpoint family keeps the plain
  ## `text_config.intermediate_size` key null.
  ModelConfigBase(
    architecture: self.config.architecture,
    model_type: self.config.modelType,
    num_hidden_layers: self.config.numHiddenLayers,
    hidden_size: self.config.hiddenSize,
    vocab_size: self.config.vocabSize,
    rms_norm_eps: self.config.rmsNormEps,
    torch_dtype: self.config.dtype,
    num_attention_heads: self.config.numAttentionHeads,
    num_key_value_heads: self.config.numKeyValueHeads,
    head_dim: self.config.headDim,
    intermediate_size: self.config.moeIntermediateSize,
    max_position_embeddings: self.config.maxPositionEmbeddings,
    eosTokenId: self.config.textEosTokenId
  )

proc getTokenizer(self: Qwen35MoeModel): BPETokenizer =
  self.tokenizer

proc getDeviceKind(self: Qwen35MoeModel): DeviceKind =
  self.device

proc loadQwen35MoeModelRaw(modelPath: string, device: DeviceKind): Qwen35MoeModel =
  ## Weight scope: the model reads `model.language_model.*` only. Foreign
  ## `model.visual.*` (vision tower) plus `mtp.*` (draft block) tensors
  ## of the same checkpoint are never requested.
  let config = loadQwen35MoeConfig(modelPath / "config.json")
  checkValue(config.hiddenAct == "silu",
    "[ttt] loadQwen35MoeModelRaw: unsupported hidden_act \"" &
    config.hiddenAct & "\", the routed block implements SiLU")
  let view = SafetensorsCollection.open(modelPath)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  let embedTokens = Embedding.load(view, cfgJson, "model.language_model.embed_tokens", device)

  let rotary = RotaryPositionEmbedding.new(
    config.headDim,
    config.maxPositionEmbeddings,
    config.ropeTheta,
    actDtype,
    device,
    rotary_dim = int(config.headDim.float64 * config.partialRotaryFactor))

  var layers = newSeq[AnyDecoderLayer](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lp = "model.language_model.layers." & $i
    let mlpPrefix = lp & ".mlp"

    let inputLN = RmsNormOne.load(view, cfgJson, lp & ".input_layernorm", device)
    let postLN = RmsNormOne.load(view, cfgJson, lp & ".post_attention_layernorm", device)

    let ffn = GatedBlockSparseFFN.load(
      view, cfgJson, mlpPrefix, config.numExpertsPerTok, device)

    var gdn: GatedDeltaNet[perHead, FullRankGateIn, GateForm.softplus] = nil
    var attn: RopeElementWiseGatedAttention[RmsNormOne] = nil
    if config.layerTypes[i] == alkGatedDeltaNet:
      let gdnPrefix = lp & ".linear_attn"
      gdn = GatedDeltaNet[perHead, FullRankGateIn, GateForm.softplus].load(
        view, cfgJson, gdnPrefix, i,
        config.linearNumKeyHeads, config.linearNumValueHeads,
        config.linearKeyHeadDim, config.linearValueHeadDim,
        config.linearConvKernelDim, device)
    else:
      let attnPrefix = lp & ".self_attn"
      attn = RopeElementWiseGatedAttention[RmsNormOne].load(
        view, cfgJson, attnPrefix, i,
        config.numAttentionHeads, config.numKeyValueHeads, config.headDim,
        rotary, device)

    layers[i] =
      if config.layerTypes[i] == alkGatedDeltaNet:
        Qwen35MoeGdnDecoderLayer.init(
          input_layernorm = inputLN, sequence_mixer = gdn,
          post_attention_layernorm = postLN,
          hidden_mixer = ffn).to(AnyDecoderLayer)
      else:
        Qwen35MoeAttnDecoderLayer.init(
          input_layernorm = inputLN, sequence_mixer = attn,
          post_attention_layernorm = postLN,
          hidden_mixer = ffn).to(AnyDecoderLayer)

  let norm = RmsNormOne.load(view, cfgJson, "model.language_model.norm", device)

  let lmHead = LMHead.load(view, cfgJson, embedTokens, device)

  let tokenizer = loadHFTokenizer(modelPath / "tokenizer.json")

  result = Qwen35MoeModel(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    rotary: rotary,
    config: config,
    tokenizer: tokenizer,
    device: device,
  )

proc loadQwen35MoeModel*(modelPath: string, device: DeviceKind): AnyModel =
  ## Loads the Qwen3.5-MoE checkpoint directory from `modelPath` onto `device`,
  ## wrapped as AnyModel.
  let qwen35MoeModel = loadQwen35MoeModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  qwen35MoeModel.to(AnyModel)

static:
  # Register the Qwen3.5 MoE model in the registry
  ModelRegistry["Qwen3_5MoeForConditionalGeneration"] = loadQwen35MoeModel
