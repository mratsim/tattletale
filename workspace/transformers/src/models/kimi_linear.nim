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
  ../deserialization,
  ../layers/attn_ssm/gated_delta_net,
  workspace/transformers/src/layers/rope,
  ../layers/attn_ssm/multi_head_latent_attention,
  ../instrumentation,
  ../models/loading/config_json,
  ../models/loading/layer_kinds,
  ../models/loading/generation_config,
  ../quantizations/datatypes,
  ../stateful/inference_context,
  workspace/safetensors/src/collections,
  workspace/safetensors/src/safetensors_libtorch,
  ./all_interfaces

## Kimi-Linear 48B A3B inference model.
##
## Parses the kimi_linear config.json and converts the hybrid MLA and KDA
## schedule to per-layer kinds, then loads the decoder stack and tokenizer
## into a KimiModel.

{.experimental: "callOperator".}

type
  ## Config parsed from the kimi_linear config.json.
  KimiConfig* = ref object
    architecture*: string
    modelType*: string
    transformersVersion*: string

    vocabSize*: int
    hiddenSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int

    # MLA shape, direct-Q (q_lora_rank null on this template),
    # unrotated kpe plane
    qLoraRankIsNone*: bool
    kvLoraRank*: int
    qkNopeHeadDim*: int
    qkRopeHeadDim*: int
    vHeadDim*: int

    # Hybrid schedule, converted at parse time from the 1-indexed config
    # list to 0-based per-layer kinds
    fullAttnLayers1Indexed*: seq[int]
    kdaLayers1Indexed*: seq[int]
    layerKinds*: seq[AttentionLayerKind]

    # Feed-forward blocks and the routed MoE
    firstKDenseReplace*: int
    intermediateSize*: int
    moeIntermediateSize*: int
    nSharedExperts*: int
    nRoutedExperts*: int
    numExpertsPerTok*: int
    routedScalingFactor*: float64
    numExpertGroup*: int
    topkGroup*: int
    moeRenormalize*: bool

    # KDA mixer shape, low-rank gated derivation
    kdaNumHeads*: int
    kdaHeadDim*: int
    shortConvKernelSize*: int

    # Numerics and position handling
    rmsNormEps*: float64
    ropeTheta*: float64
    maxPositionEmbeddings*: int
    dtype*: string

    # Token ids:
    #   the stop set comes from generation_config.json
    # eos_token_id, a bare int on this checkpoint
    eosTokenIds*: seq[int]
    bosTokenId*: int
    padTokenId*: int

type
  KimiMlaLayer* = DecoderLayer[MLAttention[void, NoPe],
      BlockSparseFFN, RmsNorm]
    ## Routed MLA layers at the converted 0-based full-attention indices
    ## 3, 7, 11, 15, 19, 23, 26.
    ## - Direct-Q (q_lora_rank null), the kpe plane cached unrotated.
    ## - The reference deletes the rotary embedding entirely.
    ## - The 64 plane channels still cache and join all head-shared scores.
    ## - Noaux_tc sigmoid top-8 router, the degenerate single-group selection.

  KimiKdaLayer* = DecoderLayer[GatedDeltaNet[perChannel, LowRankGateIn, GateForm.softplus],
      BlockSparseFFN, RmsNorm]
    ## Routed KDA layers, the 20 remaining blocks, low-rank gated derivation
    ## and the double-rounded sigmoid output norm with the config eps.

  KimiDenseLayer* = DecoderLayer[GatedDeltaNet[perChannel, LowRankGateIn, GateForm.softplus],
      GatedDenseFFN, RmsNorm]
    ## Layer 0, first_k_dense_replace 1 keeps every later block routed.

# Kimi-Linear config parsing

proc parseHybridScheduleExplicit(numLayers: int,
    fullAttnLayers1Indexed, kdaLayers1Indexed: seq[int]): seq[AttentionLayerKind] =
  ## Kimi's hybrid schedule from the 1-indexed checkpoint layer lists.
  ##
  ## Returns the 0-based per-layer kind sequence.
  ## - The full_attn_layers list marks the MLA layers.
  ## - The kda_layers list cross-checks as its complement.
  ## - Both lists cover the stack exactly, the LAST layer is an MLA layer.
  ## Raises ValueError naming the violated invariant and the 1-indexed entry.
  checkValue(numLayers > 0,
    "[ttt] KimiSchedule: the layer count must be positive, found " & $numLayers)
  result = newSeq[AttentionLayerKind](numLayers)
  for i in 0 ..< numLayers:
    result[i] = alkGatedDeltaNet
  var mlaCount = 0
  for oneIndexed in fullAttnLayers1Indexed:
    checkValue(oneIndexed >= 1 and oneIndexed <= numLayers,
      "[ttt] KimiSchedule: full_attn_layers entry " & $oneIndexed &
      " outside the 1-indexed layer range 1.." & $numLayers)
    checkValue(result[oneIndexed - 1] == alkGatedDeltaNet,
      "[ttt] KimiSchedule: full_attn_layers entry " & $oneIndexed &
      " repeats an MLA layer")
    result[oneIndexed - 1] = alkMla
    inc mlaCount
  checkValue(mlaCount != 0,
    "[ttt] KimiSchedule: the converted full-attention set is empty")
  checkValue(result[^1] == alkMla,
    "[ttt] KimiSchedule: the LAST layer must be an MLA layer")
  checkValue(fullAttnLayers1Indexed.len + kdaLayers1Indexed.len == numLayers,
    "[ttt] KimiSchedule: " & $fullAttnLayers1Indexed.len &
    " full-attention and " & $kdaLayers1Indexed.len &
    " KDA entries do not cover the " & $numLayers & " layer stack")
  var kdaConverted = newSeq[int]()
  for oneIndexed in kdaLayers1Indexed:
    checkValue(oneIndexed >= 1 and oneIndexed <= numLayers,
      "[ttt] KimiSchedule: kda_layers entry " & $oneIndexed &
      " outside the 1-indexed layer range 1.." & $numLayers)
    kdaConverted.add oneIndexed - 1
  var kdaComplement = newSeq[int]()
  for i in 0 ..< numLayers:
    if result[i] == alkGatedDeltaNet:
      kdaComplement.add i
  checkValue(kdaComplement == kdaConverted.sorted,
    "[ttt] KimiSchedule: the kda_layers set disagrees with the" &
    " complement of the full_attn_layers set")

proc parseKimiConfig(json: JsonNode): KimiConfig =
  let archs = json{"architectures"}
  checkValue(archs.kind == JArray and archs.len != 0,
    "[ttt] No architectures found in config.json")
  result = KimiConfig()
  result.architecture = archs[0].getStr()
  result.modelType = json{"model_type"}.getStr()
  result.transformersVersion = json{"transformers_version"}.getStr("")

  result.vocabSize = json{"vocab_size"}.reqPosInt("vocab_size")
  result.hiddenSize = json{"hidden_size"}.reqPosInt("hidden_size")
  result.numHiddenLayers = json{"num_hidden_layers"}.reqPosInt(
    "num_hidden_layers")
  result.numAttentionHeads = json{"num_attention_heads"}.reqPosInt(
    "num_attention_heads")

  # Direct-Q is the only built MLA form:
  #   q_lora_rank null maps
  # to the compressed-Q-free overload, any concrete rank refuses.
  result.qLoraRankIsNone = json{"q_lora_rank"}.kind == JNull
  checkValue(result.qLoraRankIsNone,
    "[ttt] KimiConfig: q_lora_rank is not null, direct-Q is the" &
    " only built form on this template")
  checkValue(json{"mla_use_nope"}.getBool(false),
    "[ttt] KimiConfig: mla_use_nope is false, only the unrotated-plane" &
    " form is built on this template")
  result.kvLoraRank = json{"kv_lora_rank"}.reqPosInt("kv_lora_rank")
  result.qkNopeHeadDim = json{"qk_nope_head_dim"}.reqPosInt(
    "qk_nope_head_dim")
  result.qkRopeHeadDim = json{"qk_rope_head_dim"}.reqPosInt(
    "qk_rope_head_dim")
  result.vHeadDim = json{"v_head_dim"}.reqPosInt("v_head_dim")

  # Hybrid schedule:
  #   the config spells BOTH layer lists 1-INDEXED,
  # the shared explicit-list parser converts to the 0-based per-layer kinds
  # and cross-checks the two lists against each other.
  let lac = json{"linear_attn_config"}
  checkValue(lac.kind == JObject,
    "[ttt] KimiConfig: linear_attn_config is missing, the hybrid schedule" &
    " has no source")
  result.kdaNumHeads = lac{"num_heads"}.reqPosInt(
    "linear_attn_config.num_heads")
  result.kdaHeadDim = lac{"head_dim"}.reqPosInt(
    "linear_attn_config.head_dim")
  result.shortConvKernelSize = lac{"short_conv_kernel_size"}.reqPosInt(
    "linear_attn_config.short_conv_kernel_size")
  for node in lac{"full_attn_layers"}.items():
    result.fullAttnLayers1Indexed.add node.reqInt(
      "linear_attn_config.full_attn_layers")
  for node in lac{"kda_layers"}.items():
    result.kdaLayers1Indexed.add node.reqInt(
      "linear_attn_config.kda_layers")
  result.layerKinds = parseHybridScheduleExplicit(
    result.numHiddenLayers, result.fullAttnLayers1Indexed,
    result.kdaLayers1Indexed)

  result.firstKDenseReplace = json{"first_k_dense_replace"}.reqInt(
    "first_k_dense_replace")
  # Parse-time invariant on the schedule:
  #   the dense region carries
  # no attention match, every layer under first_k_dense_replace
  # must be a KDA layer.
  for i in 0 ..< result.firstKDenseReplace:
    checkValue(result.layerKinds[i] == alkGatedDeltaNet,
      "[ttt] KimiConfig: layer " & $i &
      " sits below first_k_dense_replace but converts to an MLA layer," &
      " the dense vocabulary carries no attention match")
  result.intermediateSize = json{"intermediate_size"}.reqPosInt(
    "intermediate_size")
  result.moeIntermediateSize = json{"moe_intermediate_size"}.reqPosInt(
    "moe_intermediate_size")
  result.nSharedExperts = json{"num_shared_experts"}.reqInt(
    "num_shared_experts")
  result.nRoutedExperts = json{"num_experts"}.reqPosInt("num_experts")
  result.numExpertsPerTok = json{"num_experts_per_token"}.reqPosInt(
    "num_experts_per_token")
  result.routedScalingFactor = json{"routed_scaling_factor"}.reqPosFloat(
    "routed_scaling_factor")
  result.numExpertGroup = json{"num_expert_group"}.reqInt("num_expert_group")
  result.topkGroup = json{"topk_group"}.reqInt("topk_group")
  result.moeRenormalize = json{"moe_renormalize"}.getBool()

  # NoauxTc derives from the checkpoint config keys.
  # No topk_method field exists, the router class follows from the sigmoid
  # scoring and the grouped-topk flag. num_expert_group 1 leaves the group
  # limiting degenerate (one group of all experts, the mask stays all-ones).
  checkValue(json{"moe_router_activation_func"}.getStr("") == "sigmoid",
    "[ttt] KimiConfig: moe_router_activation_func is \"" &
    json{"moe_router_activation_func"}.getStr("") &
    "\", only the sigmoid scoring derives a NoauxTcRouter on this template")
  checkValue(json{"use_grouped_topk"}.getBool(false),
    "[ttt] KimiConfig: use_grouped_topk is false, the noaux_tc derivation" &
    " does not apply")
  checkValue(result.numExpertGroup == 1 and result.topkGroup == 1,
    "[ttt] KimiConfig: num_expert_group " & $result.numExpertGroup &
    " / topk_group " & $result.topkGroup &
    " is not the degenerate single-group wiring this template is built for")

  # Top-level head_dim is a dense-MLP value this template leaves unread.
  # The KDA head dims come from linear_attn_config exclusively.

  result.rmsNormEps = json{"rms_norm_eps"}.reqPosFloat("rms_norm_eps")
  result.ropeTheta = json{"rope_theta"}.reqPosFloat("rope_theta")
  # Position budget. This checkpoint's config.json carries NO
  # max_position_embeddings key, the budget comes from model_max_length
  # (what the HF KimiLinearConfig reports as 1048576).
  # A checkpoint carrying the generic key instead is accepted, one
  # carrying neither is refused.
  if json{"model_max_length"}.kind == JInt:
    result.maxPositionEmbeddings = json{"model_max_length"}.reqPosInt(
      "model_max_length")
  else:
    result.maxPositionEmbeddings = json{"max_position_embeddings"}.reqPosInt(
      "max_position_embeddings")
  result.dtype = json{"dtype"}.getStr("bfloat16")
  checkValue(result.dtype == "bfloat16",
    "[ttt] KimiConfig: dtype is \"" & result.dtype &
    "\", only the bf16 deploy is built on this template")
  checkValue(result.numExpertsPerTok <= result.nRoutedExperts,
    "[ttt] KimiConfig: num_experts_per_token " & $result.numExpertsPerTok &
    " exceeds the routed expert count " & $result.nRoutedExperts)
  checkValue(0 <= result.firstKDenseReplace and
    result.firstKDenseReplace <= result.numHiddenLayers,
    "[ttt] KimiConfig: first_k_dense_replace " &
    $result.firstKDenseReplace & " outside 0.." & $result.numHiddenLayers)
  checkValue(result.nSharedExperts == 1,
    "[ttt] KimiConfig: num_shared_experts is " & $result.nSharedExperts &
    ", the ungated shared tail serves exactly one shared expert")

proc loadKimiConfig*(path: string): KimiConfig =
  ## Config reader for the fixture suites.
  ##
  ## Returns the KimiConfig parsed from `path` plus the generation config beside it.
  ## - `path` is the config.json path, the stop set comes from the generation_config.json.
  ## - eos_token_id is a bare int on this checkpoint, the list reader's
  ##   bare-int form handles the type.
  ## - Special ids stay optional fields of the generation config and fall
  ##   back to the config.json values.
  let raw = parseFile(path)
  result = raw.parseKimiConfig()
  let gen = loadGenerationConfig(parentDir(path) / "generation_config.json")
  result.eosTokenIds = gen.eosTokenIds
  if gen.bosTokenId.isSome:
    result.bosTokenId = gen.bosTokenId.get()
  else:
    result.bosTokenId = raw{"bos_token_id"}.reqInt("bos_token_id")
  if gen.padTokenId.isSome:
    result.padTokenId = gen.padTokenId.get()
  else:
    result.padTokenId = raw{"pad_token_id"}.reqInt("pad_token_id")
  checkValue(result.bosTokenId != result.eosTokenIds[0] and
      result.padTokenId != result.eosTokenIds[0],
    "[ttt] KimiConfig: the bos or pad id collides with the stop id")

# Kimi-Linear model assembly

type
  ## Assembled Kimi-Linear model with the decoder stack, norm, head, config, tokenizer, and device.
  KimiModel* = ref object
    embedTokens: Embedding
    layers: seq[AnyDecoderLayer]
    norm: RmsNorm
    lmHead: LMHead
    config*: KimiConfig
    tokenizer*: BPETokenizer
    device*: DeviceKind

proc forward*(self: KimiModel, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Returns the logits after a forward pass through the decoder stack.
  var x = self.embedTokens.forward(input_ids)

  # No rotation anywhere on this template:
  #   the reference deletes the rotary
  # embedding entirely and the position_ids flow through untouched, so no
  # rope tables enter the context and no per-forward rope call exists
  # (the NoPe mixers never read ctx.cos).

  var residual: Option[Tensor]
  for layer in self.layers:
    let layerOut = layer(ctx, x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])

  let finalResidual = residual.get(x)
  let normed = self.norm.forward(x + finalResidual)
  self.lmHead.forward(normed)

proc getConfig(self: KimiModel): ModelConfigBase =
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
    mlaKpeWidth: self.config.qkRopeHeadDim,
    layerKinds: self.config.layerKinds
  )

proc getTokenizer(self: KimiModel): BPETokenizer =
  self.tokenizer

proc getDeviceKind(self: KimiModel): DeviceKind =
  self.device

proc loadKimiTokenizer*(modelPath: string): BPETokenizer =
  ## Returns the BPETokenizer built from the checkpoint's tiktoken rank
  ## file and the added_tokens_decoder block of tokenizer_config.json.
  ## - Checkpoint pat_str is byte-identical to the Moonlight one, so
  ##   MoonshotPatStrRegexp is reused directly.
  ## - The special_tokens map spans 258 slots past the 163584 mergeable
  ##   ranks and fills every empty decoder-block slot with the synthetic
  ##   literal "<|reserved_token_N|>", ids ascending.
  let tokConfig = (modelPath / "tokenizer_config.json").parseFile()
  let decoder = tokConfig{"added_tokens_decoder"}
  checkValue(decoder.kind == JObject and decoder.len != 0,
    "[ttt] Kimi tokenizer: no added_tokens_decoder block in tokenizer_config.json")
  var specials = newSeq[(int, string)](decoder.len)
  var at = 0
  for idText, token in decoder:
    checkValue(token.kind == JObject,
      "[ttt] Kimi tokenizer: added_tokens_decoder entry " & idText &
      " is not an object")
    specials[at] = (parseInt(idText), token{"content"}.getStr())
    inc at
  specials.sort()
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
    "[ttt] Kimi tokenizer: decoder special id outside the reserved slot range")
  var specialTokens = initOrderedTable[string, int]()
  for at in 0 ..< merged.len:
    let (tokenId, tokenText) = merged[at]
    checkValue(tokenText.len != 0,
      "[ttt] Kimi tokenizer: empty special-token text at id " & $tokenId)
    specialTokens[tokenText] = tokenId
  loadTiktokenizer(modelPath / "tiktoken.model", MoonshotPatStrRegexp, specialTokens)

proc loadKimiLinearModelRaw(modelPath: string, device: DeviceKind): KimiModel =
  let config = loadKimiConfig(modelPath / "config.json")
  checkValue(config.modelType == "kimi_linear",
    "[ttt] loadKimiLinearModelRaw: model_type \"" & config.modelType &
    "\" is not the kimi_linear template")

  let view = SafetensorsCollection.open(modelPath)
  let cfgJson = (modelPath / "config.json").parseFile()

  let embedTokens = Embedding.load(view, cfgJson, "model.embed_tokens", device)

  let cache = MlaLatentCache.init(
    config.kvLoraRank, config.qkRopeHeadDim, config.maxPositionEmbeddings,
    kBFloat16, device)

  var layers = newSeq[AnyDecoderLayer](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lp = "model.layers." & $i
    let attnPrefix = lp & ".self_attn"

    let inputLN = RmsNorm.load(view, cfgJson, lp & ".input_layernorm", device)
    let postLN = RmsNorm.load(view, cfgJson, lp & ".post_attention_layernorm", device)

    if i < config.firstKDenseReplace:
      let attn = GatedDeltaNet[perChannel, LowRankGateIn, GateForm.softplus].load(
        view, cfgJson, attnPrefix, i,
        config.kdaNumHeads, config.kdaNumHeads,
        config.kdaHeadDim, config.kdaHeadDim,
        config.shortConvKernelSize, device)
      let dense = GatedDenseFFN.load(view, cfgJson, lp & ".mlp", device)
      layers[i] = KimiDenseLayer.init(
        input_layernorm = inputLN, sequence_mixer = attn,
        post_attention_layernorm = postLN,
        hidden_mixer = dense).to(AnyDecoderLayer)
    else:
      let mlpPrefix = lp & ".block_sparse_moe"
      # Noaux_tc router on the degenerate single-group config.
      # The checkpoint stores e_score_correction_bias bf16, the module holds
      # it f32. bf16 -> f32 is exact, so the added bias equals the checkpoint
      # bit-for-bit. Routing weights gather from the UNBIASED f32 scores,
      # the bias only affects selection.
      let router = NoauxTcRouter.init(
        view.getTensorOwned(mlpPrefix & ".gate.weight", device),
        view.getTensorOwned(
          mlpPrefix & ".gate.e_score_correction_bias", device).to(kFloat32),
        config.numExpertsPerTok, config.numExpertGroup, config.topkGroup,
        config.routedScalingFactor, config.moeRenormalize)
      # Routed experts load through the shared BlockSparseFFN loader,
      # the w1/w3/w2 key form, the shared-expert tail inside.
      let ffn = BlockSparseFFN.load(view, cfgJson, mlpPrefix, router, device,
        ekvW1W3W2)
      if config.layerKinds[i] == alkMla:
        let attn = MLAttention[void, NoPe].init(
          i, attnPrefix,
          q_proj = Linear.load(
            view, cfgJson, attnPrefix & ".q_proj", device),
          o_proj = Linear.load(
            view, cfgJson, attnPrefix & ".o_proj", device),
          kv_a_proj_with_mqa = Linear.load(
            view, cfgJson, attnPrefix & ".kv_a_proj_with_mqa", device),
          kv_a_layernorm = RmsNorm.init(
            view.getTensorOwned(
              attnPrefix & ".kv_a_layernorm.weight", device),
            qBF16,
            eps = BottleneckNormEps),
          kv_b_proj = Linear.load(
            view, cfgJson, attnPrefix & ".kv_b_proj", device),
          numHeads = config.numAttentionHeads,
          qkNopeHeadDim = config.qkNopeHeadDim,
          kvLoraRank = config.kvLoraRank,
          vHeadDim = config.vHeadDim,
          softmaxScale = mlaSoftmaxScale(config.qkNopeHeadDim,
            config.qkRopeHeadDim),
          cache = cache)
        layers[i] = KimiMlaLayer.init(
          input_layernorm = inputLN, sequence_mixer = attn,
          post_attention_layernorm = postLN,
          hidden_mixer = ffn).to(AnyDecoderLayer)
      else:
        let attn = GatedDeltaNet[perChannel, LowRankGateIn, GateForm.softplus].load(
          view, cfgJson, attnPrefix, i,
          config.kdaNumHeads, config.kdaNumHeads,
          config.kdaHeadDim, config.kdaHeadDim,
          config.shortConvKernelSize, device)
        layers[i] = KimiKdaLayer.init(
          input_layernorm = inputLN, sequence_mixer = attn,
          post_attention_layernorm = postLN,
          hidden_mixer = ffn).to(AnyDecoderLayer)

  let norm = RmsNorm.load(view, cfgJson, "model.norm", device)

  let lmHead = LMHead.load(view, cfgJson, embedTokens, device)

  let tokenizer = loadKimiTokenizer(modelPath)

  result = KimiModel(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    config: config,
    tokenizer: tokenizer,
    device: device,
  )

proc loadKimiLinearModel*(modelPath: string, device: DeviceKind): AnyModel =
  ## Returns the loaded Kimi-Linear model wrapped as an AnyModel.
  let kimiModel = loadKimiLinearModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  kimiModel.to(AnyModel)

static:
  # Register the Kimi-Linear (kimi_linear) model in the registry
  ModelRegistry["KimiLinearForCausalLM"] = loadKimiLinearModel
