# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/importutils,
  std/math,
  std/options,
  std/os,
  std/strutils,
  std/tables,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/toktoktok,
  ../layers,
  workspace/transformers/src/layers/rope,
  ../layers/attn_ssm/multi_head_latent_attention,
  ../deserialization,
  ../models/loading/config_json,
  ../models/loading/generation_config,
  ../quantizations/datatypes,
  ../stateful/inference_context,
  workspace/safetensors/src/collections,
  workspace/safetensors/src/safetensors_libtorch,
  ./all_interfaces

{.experimental: "callOperator".}

type
  Glm47Config* = ref object
    architecture: string
    modelType: string
    transformersVersion: string

    vocabSize: int
    hiddenSize: int
    numHiddenLayers*: int
    numAttentionHeads: int

    # MLA shape: compressed-Q
    qLoraRank: int
    kvLoraRank*: int
    qkNopeHeadDim: int
    qkRopeHeadDim*: int
    vHeadDim: int

    # Feed-forward blocks
    firstKDenseReplace: int
    intermediateSize: int
    moeIntermediateSize: int
    nSharedExperts: int
    nRoutedExperts: int
    numExpertsPerTok: int
    routedScalingFactor: float64
    nGroup: int
    topkGroup: int
    normTopkProb: bool

    # Numerics and position handling
    rmsNormEps: float64
    ropeTheta: float64
    maxPositionEmbeddings: int
    dtype: string

    # Token ids: the checkpoint config spells eos_token_id as a list,
    # the generator stop set lives in generation_config.json
    eosTokenIds: seq[int]

type
  Glm47DenseLayer* = DecoderLayer[MLAttention[RmsNorm, FullRoPe], GatedDenseFFN, RmsNorm]
    ## Layer 0, first_k_dense_replace 1 keeps every later block routed.

  Glm47MoeLayer* = DecoderLayer[MLAttention[RmsNorm, FullRoPe], BlockSparseFFN, RmsNorm]
    ## Routed layers 1..46, noaux_tc sigmoid top-4 router on the plain
    ## shared-expert tail.

################################################################################
#                          GLM-4.7-Flash Parsing                               #
################################################################################

func parseGlm47Config(json: JsonNode): Glm47Config =
  let archs = json{"architectures"}
  checkValue(archs.kind == JArray and archs.len != 0,
    "[ttt] No architectures found in config.json")

  result = new Glm47Config
  result.architecture = archs[0].getStr()
  result.modelType = json{"model_type"}.getStr()
  result.transformersVersion = json{"transformers_version"}.getStr("")

  result.vocabSize = json{"vocab_size"}.reqPosInt("vocab_size")
  result.hiddenSize = json{"hidden_size"}.reqPosInt("hidden_size")
  result.numHiddenLayers = json{"num_hidden_layers"}.reqPosInt("num_hidden_layers")
  result.numAttentionHeads = json{"num_attention_heads"}.reqPosInt(
    "num_attention_heads")

  # Compressed-Q shape: q_lora_rank is required on this template,
  # the direct-Q form of the mixer stays out of its build.
  result.qLoraRank = json{"q_lora_rank"}.reqPosInt("q_lora_rank")
  result.kvLoraRank = json{"kv_lora_rank"}.reqPosInt("kv_lora_rank")
  result.qkNopeHeadDim = json{"qk_nope_head_dim"}.reqPosInt("qk_nope_head_dim")
  result.qkRopeHeadDim = json{"qk_rope_head_dim"}.reqPosInt("qk_rope_head_dim")
  result.vHeadDim = json{"v_head_dim"}.reqPosInt("v_head_dim")

  result.firstKDenseReplace = json{"first_k_dense_replace"}.reqInt(
    "first_k_dense_replace")
  result.intermediateSize = json{"intermediate_size"}.reqPosInt(
    "intermediate_size")
  result.moeIntermediateSize = json{"moe_intermediate_size"}.reqPosInt(
    "moe_intermediate_size")
  result.nSharedExperts = json{"n_shared_experts"}.reqInt("n_shared_experts")
  result.nRoutedExperts = json{"n_routed_experts"}.reqPosInt("n_routed_experts")
  result.numExpertsPerTok = json{"num_experts_per_tok"}.reqPosInt(
    "num_experts_per_tok")
  result.routedScalingFactor = json{"routed_scaling_factor"}.reqPosFloat(
    "routed_scaling_factor")
  result.nGroup = json{"n_group"}.reqInt("n_group")
  result.topkGroup = json{"topk_group"}.reqInt("topk_group")
  result.normTopkProb = json{"norm_topk_prob"}.getBool()

  result.rmsNormEps = json{"rms_norm_eps"}.reqPosFloat("rms_norm_eps")
  result.ropeTheta = json{"rope_theta"}.reqPosFloat("rope_theta")
  result.maxPositionEmbeddings = json{"max_position_embeddings"}.reqPosInt(
    "max_position_embeddings")
  result.dtype = json{"torch_dtype"}.getStr("bfloat16")

  # eos_token_id is a LIST on this checkpoint, one id per conversation
  # end the chat template emits; generation_config.json carries the
  # generator stop set, built at model build.
  result.eosTokenIds = json{"eos_token_id"}.parseIntList("eos_token_id")
  checkValue(result.eosTokenIds.len != 0,
    "[ttt] Glm47Config: config.json eos_token_id carries no stop id")

  checkValue(result.numExpertsPerTok <= result.nRoutedExperts,
    "[ttt] Glm47Config: num_experts_per_tok " & $result.numExpertsPerTok &
    " exceeds the routed expert count " & $result.nRoutedExperts)
  checkValue(0 <= result.firstKDenseReplace and
    result.firstKDenseReplace <= result.numHiddenLayers,
    "[ttt] Glm47Config: first_k_dense_replace " &
    $result.firstKDenseReplace & " outside 0.." & $result.numHiddenLayers)

proc loadGlm47Config(path: string): Glm47Config =
  parseFile(path).parseGlm47Config()

################################################################################
#                        GLM-4.7-Flash Model Assembly                          #
################################################################################

type
  Glm47Model* = ref object
    embedTokens: Embedding
    layers: seq[AnyDecoderLayer]
    norm: RmsNorm
    lmHead: LMHead
    rotary: MlaRotary
    config: Glm47Config
    tokenizer: BPETokenizer
    device: DeviceKind

proc forward*(self: Glm47Model, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  var x = self.embedTokens.forward(input_ids)

  ctx.setMlaRopeForPositions(self.rotary)

  var residual: Option[Tensor]
  for layer in self.layers:
    let layerOut = layer(ctx, x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])

  let finalResidual = residual.get(x)
  let normed = self.norm.forward(x + finalResidual)
  self.lmHead.forward(normed)

func getConfig(self: Glm47Model): ModelConfigBase =
  ModelConfigBase(
    architecture: self.config.architecture,
    model_type: self.config.modelType,
    num_hidden_layers: self.config.numHiddenLayers,
    hidden_size: self.config.hiddenSize,
    vocab_size: self.config.vocabSize,
    rms_norm_eps: self.config.rmsNormEps,
    torch_dtype: self.config.dtype,
    num_attention_heads: self.config.numAttentionHeads,
    num_key_value_heads: 1,
    head_dim: self.config.kvLoraRank,
    intermediate_size: self.config.moeIntermediateSize,
    max_position_embeddings: self.config.maxPositionEmbeddings,
    eosTokenId: self.config.eosTokenIds[0],
    eosTokenIds: self.config.eosTokenIds,
    mlaKvLoraRank: self.config.kvLoraRank,
    mlaKpeWidth: self.config.qkRopeHeadDim
  )

func getTokenizer(self: Glm47Model): BPETokenizer =
  self.tokenizer

func getDeviceKind(self: Glm47Model): DeviceKind =
  self.device

proc loadGlm47ModelRaw(modelPath: string, device: DeviceKind): Glm47Model =
  let config = loadGlm47Config(modelPath / "config.json")
  checkValue(config.modelType == "glm4_moe_lite",
    "[ttt] loadGlm47ModelRaw: model_type \"" & config.modelType &
    "\" is not the glm4_moe_lite template")

  # Generator stop set, the eos_token_id list of generation_config.json.
  let stopSet = loadGenerationConfig(modelPath / "generation_config.json").eosTokenIds
  checkValue(stopSet.len != 0,
    "[ttt] loadGlm47ModelRaw: generation_config.json carries no stop id")
  config.eosTokenIds = stopSet

  let view = SafetensorsCollection.open(modelPath)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  let embedTokens = Embedding.load(view, cfgJson, "model.embed_tokens", device)

  let rotary = MlaRotary.new(
    config.qkRopeHeadDim,
    config.maxPositionEmbeddings,
    config.ropeTheta,
    device,
    # Plain-rope checkpoint: yarnFactor 0.0 stays spelled explicitly,
    # the yarn blend and its beta parameters remain unread.
    yarnFactor = 0.0)

  let cache = MlaLatentCache.init(
    config.kvLoraRank, config.qkRopeHeadDim, config.maxPositionEmbeddings,
    actDtype, device)

  var layers = newSeq[AnyDecoderLayer](config.numHiddenLayers)
  var routers = newSeq[NoAuxTopCorr](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lp = "model.layers." & $i
    let mlpPrefix = lp & ".mlp"

    let inputLN = RmsNorm.load(view, cfgJson, lp & ".input_layernorm", device)
    let postLN = RmsNorm.load(view, cfgJson, lp & ".post_attention_layernorm", device)

    let attnPrefix = lp & ".self_attn"
    let attn = MLAttention[RmsNorm, FullRoPe].init(
      i, attnPrefix,
      q_a_proj = Linear.load(view, cfgJson, attnPrefix & ".q_a_proj", device),
      q_b_proj = Linear.load(view, cfgJson, attnPrefix & ".q_b_proj", device),
      q_a_norm = RmsNorm.init(
        view.getTensorOwned(attnPrefix & ".q_a_layernorm.weight", device),
        qBF16,
        eps = BottleneckNormEps),
      kv_a_proj_with_mqa = Linear.load(
        view, cfgJson, attnPrefix & ".kv_a_proj_with_mqa", device),
      kv_a_layernorm = RmsNorm.init(
        view.getTensorOwned(attnPrefix & ".kv_a_layernorm.weight", device),
        qBF16,
        eps = BottleneckNormEps),
      kv_b_proj = Linear.load(view, cfgJson, attnPrefix & ".kv_b_proj", device),
      o_proj = Linear.load(view, cfgJson, attnPrefix & ".o_proj", device),
      numHeads = config.numAttentionHeads,
      qkNopeHeadDim = config.qkNopeHeadDim,
      kvLoraRank = config.kvLoraRank,
      vHeadDim = config.vHeadDim,
      softmaxScale = mlaSoftmaxScale(config.qkNopeHeadDim,
        config.qkRopeHeadDim),
      cache = cache)

    if i < config.firstKDenseReplace:
      let dense = GatedDenseFFN.load(view, cfgJson, mlpPrefix, device)
      layers[i] = Glm47DenseLayer.init(
        input_layernorm = inputLN, sequence_mixer = attn,
        post_attention_layernorm = postLN,
        hidden_mixer = dense).to(AnyDecoderLayer)
      routers[i] = nil
    else:
      let routerWeight = view.getTensorOwned(mlpPrefix & ".gate.weight", device)
      let expertBias = view.getTensorOwned(
        mlpPrefix & ".gate.e_score_correction_bias", device)
      routers[i] = NoAuxTopCorr.init(
        routerWeight, expertBias,
        config.numExpertsPerTok, config.nGroup, config.topkGroup,
        config.routedScalingFactor, config.normTopkProb)
      let ffn = BlockSparseFFN.load(view, cfgJson, mlpPrefix, routers[i], device)
      layers[i] = Glm47MoeLayer.init(
        input_layernorm = inputLN, sequence_mixer = attn,
        post_attention_layernorm = postLN,
        hidden_mixer = ffn).to(AnyDecoderLayer)

  let norm = RmsNorm.load(view, cfgJson, "model.norm", device)

  let lmHead = LMHead.load(view, cfgJson, embedTokens, device)

  let tokenizer = loadHFTokenizer(modelPath / "tokenizer.json")

  result = Glm47Model(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    rotary: rotary,
    config: config,
    tokenizer: tokenizer,
    device: device,
  )

proc loadGlm47Model*(modelPath: string, device: DeviceKind): AnyModel =
  let glm47FlashModel = loadGlm47ModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  glm47FlashModel.to(AnyModel)

static:
  # Register the GLM-4.7-Flash (glm4_moe_lite) model in the registry
  ModelRegistry["Glm4MoeLiteForCausalLM"] = loadGlm47Model
