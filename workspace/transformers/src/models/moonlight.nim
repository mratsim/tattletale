# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/algorithm,
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
  ../quantizations/datatypes,
  ../models/loading/generation_config,
  ../stateful/inference_context,
  workspace/safetensors/src/collections,
  workspace/safetensors/src/safetensors_libtorch,
  ./all_interfaces

{.experimental: "callOperator".}

type
  MoonlightConfig* = ref object
    architecture*: string
    modelType*: string
    transformersVersion*: string

    vocabSize*: int
    hiddenSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int

    # MLA shape
    kvLoraRank*: int
    qkNopeHeadDim*: int
    qkRopeHeadDim*: int
    vHeadDim*: int

    # Feed-forward blocks
    firstKDenseReplace*: int
    intermediateSize*: int
    moeIntermediateSize*: int
    nSharedExperts*: int
    nRoutedExperts*: int
    numExpertsPerTok*: int
    routedScalingFactor*: float64
    nGroup*: int
    topkGroup*: int
    normTopkProb*: bool

    # Numerics and position handling
    rmsNormEps*: float64
    ropeTheta*: float64
    maxPositionEmbeddings*: int
    dtype*: string

    # Token ids
    textEosTokenId*: int

type
  MoonlightDenseLayer* = DecoderLayer[MLAttention[void, FullRoPe], GatedDenseFFN, RmsNorm]

  MoonlightMoeLayer* = DecoderLayer[MLAttention[void, FullRoPe], BlockSparseFFN, RmsNorm]

################################################################################
#                          Moonlight Parsing                                   #
################################################################################

proc parseMoonlightConfig(json: JsonNode): MoonlightConfig =
  ## Flat DeepseekV3-style config layout; torch_dtype falls back to the
  ## bf16 deploy default.
  let archs = json{"architectures"}
  checkValue(archs.kind == JArray and archs.len != 0,
    "[ttt] No architectures found in config.json")

  result = new MoonlightConfig
  result.architecture = archs[0].getStr()
  result.modelType = json{"model_type"}.getStr()
  result.transformersVersion = json{"transformers_version"}.getStr("")

  result.vocabSize = json{"vocab_size"}.reqPosInt("vocab_size")
  result.hiddenSize = json{"hidden_size"}.reqPosInt("hidden_size")
  result.numHiddenLayers = json{"num_hidden_layers"}.reqPosInt("num_hidden_layers")
  result.numAttentionHeads = json{"num_attention_heads"}.reqPosInt("num_attention_heads")

  result.kvLoraRank = json{"kv_lora_rank"}.reqPosInt("kv_lora_rank")
  result.qkNopeHeadDim = json{"qk_nope_head_dim"}.reqPosInt("qk_nope_head_dim")
  result.qkRopeHeadDim = json{"qk_rope_head_dim"}.reqPosInt("qk_rope_head_dim")
  result.vHeadDim = json{"v_head_dim"}.reqPosInt("v_head_dim")

  result.firstKDenseReplace = json{"first_k_dense_replace"}.reqInt(
    "first_k_dense_replace")
  result.intermediateSize = json{"intermediate_size"}.reqPosInt("intermediate_size")
  result.moeIntermediateSize = json{"moe_intermediate_size"}.reqPosInt(
    "moe_intermediate_size")
  result.nSharedExperts = json{"n_shared_experts"}.reqInt("n_shared_experts")
  result.nRoutedExperts = json{"n_routed_experts"}.reqPosInt("n_routed_experts")
  result.numExpertsPerTok = json{"num_experts_per_tok"}.reqPosInt("num_experts_per_tok")
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

  result.textEosTokenId = json{"eos_token_id"}.reqInt(
    "eos_token_id",
    "the stop set a generator uses lives in generation_config.json")

  checkValue(result.numExpertsPerTok <= result.nRoutedExperts,
    "[ttt] MoonlightConfig: num_experts_per_tok " & $result.numExpertsPerTok &
    " exceeds the routed expert count " & $result.nRoutedExperts)
  checkValue(0 <= result.firstKDenseReplace and
    result.firstKDenseReplace <= result.numHiddenLayers,
    "[ttt] MoonlightConfig: first_k_dense_replace " &
    $result.firstKDenseReplace & " outside 0.." & $result.numHiddenLayers)

proc loadMoonlightConfig(path: string): MoonlightConfig =
  parseFile(path).parseMoonlightConfig()

################################################################################
#                        Moonlight Model Assembly                              #
################################################################################

type
  MoonlightModel* = ref object
    embedTokens: Embedding
    layers: seq[AnyDecoderLayer]
    norm: RmsNorm
    lmHead: LMHead
    rotary: MlaRotary
    config*: MoonlightConfig
    tokenizer*: BPETokenizer
    device*: DeviceKind

proc forward*(self: MoonlightModel, ctx: var InferenceContext, input_ids: Tensor): Tensor =
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

proc getConfig(self: MoonlightModel): ModelConfigBase =
  ## Minimal config behind the `generate()` entry point. The MLA fields
  ## size the per-buffer pool: K the compressed latent, V the kpe plane,
  ## both single-head. The shared fields carry the same K width
  ## so context bookkeeping stays meaningful on a latent-cache checkpoint.
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
    eosTokenId: self.config.textEosTokenId,
    mlaKvLoraRank: self.config.kvLoraRank,
    mlaKpeWidth: self.config.qkRopeHeadDim
  )

proc getTokenizer(self: MoonlightModel): BPETokenizer =
  self.tokenizer

proc getDeviceKind(self: MoonlightModel): DeviceKind =
  self.device

proc loadMoonlightTokenizer(modelPath: string): BPETokenizer =
  ## Moonlight ships a tiktoken rank file and carries its special tokens
  ## in the added_tokens_decoder block of tokenizer_config.json, no
  ## tokenizer.json exists on the checkpoint side. MoonshotPatStrRegexp
  ## translates the checkpoint pat_str to PCRE2; the token-identity
  ## referee is the q_bf16 end-to-end suite for this model.
  let tokConfig = (modelPath / "tokenizer_config.json").parseFile()
  let decoder = tokConfig{"added_tokens_decoder"}
  checkValue(decoder.kind == JObject and decoder.len != 0,
    "[ttt] Moonlight tokenizer: no added_tokens_decoder block in tokenizer_config.json")
  var specials = newSeq[(int, string)](decoder.len)
  var at = 0
  for idText, token in decoder:
    checkValue(token.kind == JObject,
      "[ttt] Moonlight tokenizer: added_tokens_decoder entry " & idText &
      " is not an object")
    specials[at] = (parseInt(idText), token{"content"}.getStr())
    inc at
  specials.sort()
  # Decoder-uncovered slots inside the 258-slot special range take
  # the <|reserved_token_N|> fillers.
  let rankContent = readFile(modelPath / "tiktoken.model")
  let numBaseTokens = rankContent.count('\n') +
    (if rankContent.len != 0 and rankContent[^1] != '\n': 1 else: 0)
  var merged = newSeq[(int, string)]()
  var decIdx = 0
  for tokenId in numBaseTokens ..< numBaseTokens + 258:
    if decIdx < specials.len and specials[decIdx][0] == tokenId:
      merged.add specials[decIdx]
      inc decIdx
    else:
      merged.add (tokenId, "<|reserved_token_" & $tokenId & "|>")
  checkValue(decIdx == specials.len,
    "[ttt] Moonlight tokenizer: decoder special id outside the reserved slot range")
  var specialTokens = initOrderedTable[string, int]()
  for at in 0 ..< merged.len:
    let (tokenId, tokenText) = merged[at]
    checkValue(tokenText.len != 0,
      "[ttt] Moonlight tokenizer: empty special-token text at id " & $tokenId)
    specialTokens[tokenText] = tokenId
  loadTiktokenizer(modelPath / "tiktoken.model", MoonshotPatStrRegexp, specialTokens)

proc loadMoonlightModelRaw(modelPath: string, device = kCPU): MoonlightModel =
  ## Weight scope: `model.*` plus the untied `lm_head.weight`; no
  ## vision tower and no draft block on this checkpoint.
  let config = loadMoonlightConfig(modelPath / "config.json")
  checkValue(config.modelType == "deepseek_v3",
    "[ttt] loadMoonlightModelRaw: model_type \"" & config.modelType &
    "\" is not the deepseek_v3 template")
  let view = SafetensorsCollection.open(modelPath)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  let embedTokens = Embedding.load(view, cfgJson, "model.embed_tokens", device)

  let rotary = MlaRotary.new(
    config.qkRopeHeadDim,
    config.maxPositionEmbeddings,
    config.ropeTheta,
    device,
    # Default-rope checkpoint: yarnFactor 0.0 stays spelled explicitly,
    # the yarn blend and its beta parameters remain unread.
    yarnFactor = 0.0)

  let cache = MlaLatentCache.init(
    config.kvLoraRank, config.qkRopeHeadDim, config.maxPositionEmbeddings,
    actDtype, device)

  var layers = newSeq[AnyDecoderLayer](config.numHiddenLayers)
  var routers = newSeq[NoauxTcRouter](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lp = "model.layers." & $i
    let mlpPrefix = lp & ".mlp"

    let inputLN = RmsNorm.load(view, cfgJson, lp & ".input_layernorm", device)
    let postLN = RmsNorm.load(view, cfgJson, lp & ".post_attention_layernorm", device)

    let attnPrefix = lp & ".self_attn"
    let attn = MLAttention[void, FullRoPe].init(
      i, attnPrefix,
      Linear.load(view, cfgJson, attnPrefix & ".q_proj", device),
      Linear.load(view, cfgJson, attnPrefix & ".o_proj", device),
      Linear.load(view, cfgJson, attnPrefix & ".kv_a_proj_with_mqa", device),
      RmsNorm.init(
        view.getTensorOwned(attnPrefix & ".kv_a_layernorm.weight", device),
        qBF16,
        eps = BottleneckNormEps),
      Linear.load(view, cfgJson, attnPrefix & ".kv_b_proj", device),
      numHeads = config.numAttentionHeads,
      qkNopeHeadDim = config.qkNopeHeadDim,
      kvLoraRank = config.kvLoraRank,
      vHeadDim = config.vHeadDim,
      softmaxScale = mlaSoftmaxScale(config.qkNopeHeadDim,
        config.qkRopeHeadDim),
      cache = cache)

    if i < config.firstKDenseReplace:
      let dense = GatedDenseFFN.load(view, cfgJson, mlpPrefix, device)
      layers[i] = MoonlightDenseLayer.init(
        input_layernorm = inputLN, sequence_mixer = attn,
        post_attention_layernorm = postLN,
        hidden_mixer = dense).to(AnyDecoderLayer)
      routers[i] = nil
    else:
      let routerWeight = view.getTensorOwned(mlpPrefix & ".gate.weight", device)
      let expertBias = view.getTensorOwned(
        mlpPrefix & ".gate.e_score_correction_bias", device)
      routers[i] = NoauxTcRouter.init(
        routerWeight, expertBias,
        config.numExpertsPerTok, config.nGroup, config.topkGroup,
        config.routedScalingFactor, config.normTopkProb)
      let ffn = BlockSparseFFN.load(view, cfgJson, mlpPrefix, routers[i], device)
      layers[i] = MoonlightMoeLayer.init(
        input_layernorm = inputLN, sequence_mixer = attn,
        post_attention_layernorm = postLN,
        hidden_mixer = ffn).to(AnyDecoderLayer)

  let norm = RmsNorm.load(view, cfgJson, "model.norm", device)

  let lmHead = LMHead.load(view, cfgJson, embedTokens, device)

  let tokenizer = loadMoonlightTokenizer(modelPath)

  result = MoonlightModel(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    rotary: rotary,
    config: config,
    tokenizer: tokenizer,
    device: device,
  )

proc loadMoonlightModel*(modelPath: string, device = kCPU): AnyModel =
  let moonlightModel = loadMoonlightModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  moonlightModel.to(AnyModel)

static:
  # Register the Moonlight (DeepseekV3) model in the registry
  ModelRegistry["DeepseekV3ForCausalLM"] = loadMoonlightModel
