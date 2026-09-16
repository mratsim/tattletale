# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
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
  ../layers/attn_ssm/gated_delta_net,
  workspace/transformers/src/layers/rope,
  ../layers/attn_ssm/multi_head_latent_attention,
  ../deserialization,
  ../models/loading/config_json,
  ../models/loading/layer_kinds,
  ../quantizations/datatypes,
  ../stateful/inference_context,
  workspace/safetensors/src/collections,
  workspace/safetensors/src/safetensors_libtorch,
  ./all_interfaces

## Ling-3.0-tiny (bailing_hybrid) inference model.
##
## Parses the bailing_hybrid config.json and derives the hybrid MLA and KDA
## schedule to per-layer kinds, then loads the decoder stack and tokenizer
## into a Ling3Model.

{.experimental: "callOperator".}

type
  ## Config parsed from the bailing_hybrid config.json.
  LingConfig* = ref object
    architecture*: string
    modelType*: string
    transformersVersion*: string

    vocabSize*: int
    hiddenSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int

    # MLA shape:
    #   compressed-Q with a head-wise gated output
    qLoraRank*: int
    kvLoraRank*: int
    qkNopeHeadDim*: int
    qkRopeHeadDim*: int
    qkHeadDim*: int
    vHeadDim*: int

    # Hybrid schedule:
    #   MLA iff (layer_idx + 1) mod layerGroupSize == 0,
    # materialized into one typed per-layer-kind seq at parse
    layerGroupSize*: int
    layerKinds*: seq[AttentionLayerKind]

    # Feed-forward blocks and the routed MoE
    firstKDenseReplace*: int
    intermediateSize*: int
    moeIntermediateSize*: int
    moeSharedExpertIntermediateSize*: int
    nSharedExperts*: int
    nRoutedExperts*: int
    numExpertsPerTok*: int
    routedScalingFactor*: float64
    nGroup*: int
    topkGroup*: int
    normTopkProb*: bool

    # KDA mixer shape (lower-bound-sigmoid full-rank gate)
    kdaHeadDim*: int
    shortConvKernelSize*: int
    kdaLowerBound*: float64

    # Numerics and position handling
    rmsNormEps*: float64
    ropeTheta*: float64
    maxPositionEmbeddings*: int
    dtype*: string

    # Token ids come from config.json alone.
    eosTokenIds*: seq[int]

type
  LingMlaLayer* = DecoderLayer[HeadwiseGatedMLAttention[RmsNorm, FullRoPe],
      BlockSparseFFN, RmsNorm]
    ## MLA variant with head-wise sigmoid-gated latent attention and routed MoE FFN.
    ## - Layer assignment runs through the config-driven hybrid schedule (layer_group_size).
    ## - Head-wise sigmoid gate before the output projection, noaux_tc sigmoid
    ##   top-8 router with the grouped selection.
    ## - Whole-plane rotation applies, the width from the config qk_rope_head_dim.

  LingKdaLayer* = DecoderLayer[GatedDeltaNet[perChannel, FullRankGateIn, lowerBoundSigmoid],
      BlockSparseFFN, RmsNorm]
    ## Routed KDA layers assigned by the same config-driven schedule.
    ## - The lower-bound-sigmoid full-rank gate derivation and the fused
    ##   single-rounding output norm.

  LingDenseLayer* = DecoderLayer[GatedDeltaNet[perChannel, FullRankGateIn, lowerBoundSigmoid],
      GatedDenseFFN, RmsNorm]
    ## Layer 0, first_k_dense_replace 1 keeps every later block routed.

# Ling-3.0-tiny config parsing

proc parseHybridScheduleGroups(numLayers, layerGroupSize: int): seq[AttentionLayerKind] =
  ## Ling's hybrid schedule from the layer_group_size rule.
  ##
  ## Returns the 0-based per-layer kind sequence.
  ## - MLA iff (layer_idx + 1) mod layer_group_size == 0.
  ## - The LAST layer must be an MLA layer.
  ## Raises ValueError naming the violated invariant.
  checkValue(numLayers > 0,
    "[ttt] LingSchedule: the layer count must be positive, found " & $numLayers)
  checkValue(layerGroupSize > 0,
    "[ttt] LingSchedule: layer_group_size must be positive, found " &
    $layerGroupSize)
  result = newSeq[AttentionLayerKind](numLayers)
  for i in 0 ..< numLayers:
    result[i] = if (i + 1) mod layerGroupSize == 0: alkMla else: alkGatedDeltaNet
  checkValue(result[^1] == alkMla,
    "[ttt] LingSchedule: layer_group_size " & $layerGroupSize &
    " leaves the LAST layer KDA, the template requires it to be an MLA layer")

proc parseLing3Config(json: JsonNode): LingConfig =
  ## Flat bailing_hybrid config layout. torch_dtype falls back to the bf16
  ## deploy default. eos_token_id accepts a bare int or a list of ints
  ## through the shared list reader, any other kind raises naming the key.
  let archs = json{"architectures"}
  checkValue(archs.kind == JArray and archs.len != 0,
    "[ttt] No architectures found in config.json")

  result = new LingConfig
  result.architecture = archs[0].getStr()
  result.modelType = json{"model_type"}.getStr()
  result.transformersVersion = json{"transformers_version"}.getStr("")

  result.vocabSize = json{"vocab_size"}.reqPosInt("vocab_size")
  result.hiddenSize = json{"hidden_size"}.reqPosInt("hidden_size")
  result.numHiddenLayers = json{"num_hidden_layers"}.reqPosInt("num_hidden_layers")
  result.numAttentionHeads = json{"num_attention_heads"}.reqPosInt(
    "num_attention_heads")

  # Compressed-Q shape:
  #   q_lora_rank is required on this template,
  # the direct-Q form of the mixer stays out of its build.
  result.qLoraRank = json{"q_lora_rank"}.reqPosInt("q_lora_rank")
  result.kvLoraRank = json{"kv_lora_rank"}.reqPosInt("kv_lora_rank")
  result.qkNopeHeadDim = json{"qk_nope_head_dim"}.reqPosInt("qk_nope_head_dim")
  result.qkRopeHeadDim = json{"qk_rope_head_dim"}.reqPosInt("qk_rope_head_dim")
  result.qkHeadDim = json{"qk_head_dim"}.reqPosInt("qk_head_dim")
  result.vHeadDim = json{"v_head_dim"}.reqPosInt("v_head_dim")

  # Hybrid schedule:
  #   the MLA groups sit at every layerGroupSize-th
  # layer via (layer_idx + 1) mod layer_group_size, the config carries
  # no layer_types array to consult. The shared group-rule parser
  # materializes the typed per-layer-kind seq, the construction loop
  # dispatches on it directly.
  result.layerGroupSize = json{"layer_group_size"}.reqPosInt("layer_group_size")
  result.layerKinds = parseHybridScheduleGroups(
    result.numHiddenLayers, result.layerGroupSize)

  result.firstKDenseReplace = json{"first_k_dense_replace"}.reqInt(
    "first_k_dense_replace")
  result.intermediateSize = json{"intermediate_size"}.reqPosInt(
    "intermediate_size")
  result.moeIntermediateSize = json{"moe_intermediate_size"}.reqPosInt(
    "moe_intermediate_size")
  result.moeSharedExpertIntermediateSize = json{
    "moe_shared_expert_intermediate_size"}.reqPosInt(
    "moe_shared_expert_intermediate_size")
  result.nSharedExperts = json{"num_shared_experts"}.reqInt("num_shared_experts")
  result.nRoutedExperts = json{"num_experts"}.reqPosInt("num_experts")
  result.numExpertsPerTok = json{"num_experts_per_tok"}.reqPosInt(
    "num_experts_per_tok")
  result.routedScalingFactor = json{"routed_scaling_factor"}.reqPosFloat(
    "routed_scaling_factor")
  result.nGroup = json{"n_group"}.reqInt("n_group")
  result.topkGroup = json{"topk_group"}.reqInt("topk_group")
  result.normTopkProb = json{"norm_topk_prob"}.getBool()

  checkValue(json{"kda_safe_gate"}.getBool(false),
    "[ttt] LingConfig: kda_safe_gate is false, only the" &
    " lower-bound-sigmoid full-rank gate is built")
  checkValue(json{"no_kda_lora"}.getBool(false),
    "[ttt] LingConfig: no_kda_lora is false, only the full-rank" &
    " gate projections are built")
  result.kdaHeadDim = json{"head_dim"}.reqPosInt("head_dim")
  result.shortConvKernelSize = json{"short_conv_kernel_size"}.reqPosInt(
    "short_conv_kernel_size")
  # The reader default. Every bailing_hybrid checkpoint on this form carries
  # kda_lower_bound -5, the value lives here, the kernel side has no bound constant.
  result.kdaLowerBound = json{"kda_lower_bound"}.getFloat(-5.0)

  checkValue(json{"gated_attention_proj_granularity_type"}.getStr("") == "head_wise",
    "[ttt] LingConfig: gated_attention_proj_granularity_type is \"" &
    json{"gated_attention_proj_granularity_type"}.getStr("") &
    "\", only the head_wise granularity is built")

  result.rmsNormEps = json{"rms_norm_eps"}.reqPosFloat("rms_norm_eps")
  result.ropeTheta = json{"rope_theta"}.reqPosFloat("rope_theta")
  result.maxPositionEmbeddings = json{"max_position_embeddings"}.reqPosInt(
    "max_position_embeddings")
  result.dtype = json{"torch_dtype"}.getStr("bfloat16")

  # The stop set comes from config.json eos_token_id alone.
  result.eosTokenIds = json{"eos_token_id"}.parseIntList("eos_token_id")
  checkValue(result.eosTokenIds.len != 0,
    "[ttt] LingConfig: config.json eos_token_id carries no stop id")

  checkValue(result.numExpertsPerTok <= result.nRoutedExperts,
    "[ttt] LingConfig: num_experts_per_tok " & $result.numExpertsPerTok &
    " exceeds the routed expert count " & $result.nRoutedExperts)
  checkValue(0 <= result.firstKDenseReplace and
    result.firstKDenseReplace <= result.numHiddenLayers,
    "[ttt] LingConfig: first_k_dense_replace " &
    $result.firstKDenseReplace & " outside 0.." & $result.numHiddenLayers)
  checkValue(result.qkHeadDim == result.qkNopeHeadDim + result.qkRopeHeadDim,
    "[ttt] LingConfig: qk_head_dim " & $result.qkHeadDim &
    " disagrees with qk_nope_head_dim " & $result.qkNopeHeadDim &
    " + qk_rope_head_dim " & $result.qkRopeHeadDim)
  checkValue(result.nRoutedExperts mod result.nGroup == 0,
    "[ttt] LingConfig: " & $result.nRoutedExperts &
    " experts do not split into n_group " & $result.nGroup & " groups")

proc loadLing3Config*(path: string): LingConfig =
  ## Config reader for the fixture suites and the schedule cross-check callers.
  parseFile(path).parseLing3Config()

# Ling-3.0-tiny model assembly

type
  ## Assembled Ling-3.0-tiny model with the decoder stack, norm, head, rotary, config, tokenizer, and device.
  Ling3Model* = ref object
    embedTokens: Embedding
    layers: seq[AnyDecoderLayer]
    norm: RmsNorm
    lmHead: LMHead
    rotary: MlaRotary
    config*: LingConfig
    tokenizer*: BPETokenizer
    device*: DeviceKind

proc forward*(self: Ling3Model, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Returns the logits after a forward pass through the decoder stack.
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

proc getConfig(self: Ling3Model): ModelConfigBase =
  ## Minimal config behind the `generate()` entry point.
  ##
  ## The MLA fields size the per-buffer pool.
  ## - K holds the compressed latent.
  ## - V holds the kpe plane, both single-head.
  ## - The KDA layers allocate their conv and SSM state slots on demand
  ##   inside the same context.
  ## - The stop set comes from the list field, the single-id field keeps
  ##   the conversation-end id.
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

proc getTokenizer(self: Ling3Model): BPETokenizer =
  self.tokenizer

proc getDeviceKind(self: Ling3Model): DeviceKind =
  self.device

proc loadLing3ModelRaw(modelPath: string, device = kCPU): Ling3Model =
  ## Loads the Ling-3.0-tiny model weights from the checkpoint.
  ##
  ## Weight scope covers `model.*` over the main stack 0..num_hidden_layers-1
  ## plus the untied `lm_head.weight`.
  ## - The embedding key is model.word_embeddings on this template.
  ## - expert_bias is stored f32 in the checkpoint but loads through the bf16 grid, matching the reference deployment dtype.
  ## - Routing weights gather from the unbiased f32 scores.
  ## - Explicit-dtype tensors A_log and dt_bias keep checkpoint dtype.
  let config = loadLing3Config(modelPath / "config.json")
  checkValue(config.modelType == "bailing_hybrid",
    "[ttt] loadLing3ModelRaw: model_type \"" & config.modelType &
    "\" is not the bailing_hybrid template")

  let view = SafetensorsCollection.open(modelPath)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  let embedTokens = Embedding.load(view, cfgJson, "model.word_embeddings", device)

  let rotary = MlaRotary.new(
    config.qkRopeHeadDim,
    config.maxPositionEmbeddings,
    config.ropeTheta,
    device,
    # Plain-rope checkpoint:
    #   yarnFactor 0.0 stays spelled explicitly,
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

    let attnPrefix = lp & ".attention"
    if config.layerKinds[i] == alkMla:
      let attn = HeadwiseGatedMLAttention[RmsNorm, FullRoPe].init(
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
        o_proj = Linear.load(view, cfgJson, attnPrefix & ".dense", device),
        g_proj = Linear.load(view, cfgJson, attnPrefix & ".g_proj", device),
        numHeads = config.numAttentionHeads,
        qkNopeHeadDim = config.qkNopeHeadDim,
        kvLoraRank = config.kvLoraRank,
        vHeadDim = config.vHeadDim,
        softmaxScale = mlaSoftmaxScale(config.qkNopeHeadDim,
          config.qkRopeHeadDim),
        cache = cache)

      let routerWeight = view.getTensorOwned(mlpPrefix & ".gate.weight", device)
      let expertBias = view.getTensorOwned(mlpPrefix & ".gate.expert_bias", device)
        .to(kBFloat16).to(kFloat32)
      routers[i] = NoauxTcRouter.init(
        routerWeight, expertBias,
        config.numExpertsPerTok, config.nGroup, config.topkGroup,
        config.routedScalingFactor, config.normTopkProb)
      let ffn = BlockSparseFFN.load(view, cfgJson, mlpPrefix, routers[i], device)
      layers[i] = LingMlaLayer.init(
        input_layernorm = inputLN, sequence_mixer = attn,
        post_attention_layernorm = postLN,
        hidden_mixer = ffn).to(AnyDecoderLayer)
    else:
      let attn = GatedDeltaNet[perChannel, FullRankGateIn, lowerBoundSigmoid].load(
        view, cfgJson, attnPrefix, i,
        config.numAttentionHeads, config.numAttentionHeads,
        config.kdaHeadDim, config.kdaHeadDim,
        config.shortConvKernelSize, device,
        kdaLowerBound = config.kdaLowerBound)

      if i < config.firstKDenseReplace:
        let dense = GatedDenseFFN.load(view, cfgJson, mlpPrefix, device)
        layers[i] = LingDenseLayer.init(
          input_layernorm = inputLN, sequence_mixer = attn,
          post_attention_layernorm = postLN,
          hidden_mixer = dense).to(AnyDecoderLayer)
        routers[i] = nil
      else:
        let routerWeight = view.getTensorOwned(mlpPrefix & ".gate.weight", device)
        let expertBias = view.getTensorOwned(
          mlpPrefix & ".gate.expert_bias", device).to(kBFloat16).to(kFloat32)
        routers[i] = NoauxTcRouter.init(
          routerWeight, expertBias,
          config.numExpertsPerTok, config.nGroup, config.topkGroup,
          config.routedScalingFactor, config.normTopkProb)
        let ffn = BlockSparseFFN.load(view, cfgJson, mlpPrefix, routers[i], device)
        layers[i] = LingKdaLayer.init(
          input_layernorm = inputLN, sequence_mixer = attn,
          post_attention_layernorm = postLN,
          hidden_mixer = ffn).to(AnyDecoderLayer)

  let norm = RmsNorm.load(view, cfgJson, "model.norm", device)

  let lmHead = LMHead.load(view, cfgJson, embedTokens, device)

  let tokenizer = loadHFTokenizer(modelPath / "tokenizer.json")

  result = Ling3Model(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    rotary: rotary,
    config: config,
    tokenizer: tokenizer,
    device: device,
  )

proc loadLing3Model*(modelPath: string, device = kCPU): AnyModel =
  ## Returns the loaded Ling-3.0-tiny model wrapped as an AnyModel.
  let ling3Model = loadLing3ModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  ling3Model.to(AnyModel)

proc censusDerivedSchedule*(modelPath: string, cfg: LingConfig):
    tuple[onDisk, mlaLayers, kdaLayers, loadScope: int] =
  ## Weight-map cross-check helper for the derived schedule.
  ##
  ## Returns the on-disk, MLA, KDA, and load-scope counts.
  ## - Every layer index lands in 0..num_hidden_layers-1.
  ## - MLA vocabulary (attention.dense, q_a_proj, q_b_proj, g_proj) sits
  ##   exactly on the (idx + 1) mod layer_group_size == 0 set.
  ## - KDA vocabulary (attention.A_log, attention.dt_bias, o_norm) sits on the other layers.
  ## - On-disk router bias buffer, A_log and dt_bias stay f32, the loader
  ##   mirrors the reference dtype plan from these raw bytes.
  let indexJson = parseFile(modelPath / "model.safetensors.index.json")
  let weightMap = indexJson{"weight_map"}
  checkValue(weightMap.kind == JObject and weightMap.len != 0,
    "[ttt] censusDerivedSchedule: model.safetensors.index.json carries no weight_map")
  let layerPrefix = "model.layers."
  var mlaSeen, kdaSeen: seq[int]
  var biasKeys, aLogKeys, dtBiasKeys: seq[string]
  for name, shard in weightMap:
    checkValue(shard.kind == JString,
      "[ttt] censusDerivedSchedule: weight_map entry " & $name &
      " is not a filename")
    inc result.onDisk
    if name.startsWith(layerPrefix):
      let rest = name[layerPrefix.len .. ^1]
      let dot = rest.find('.')
      checkValue(dot > 0,
        "[ttt] censusDerivedSchedule: key " & $name & " carries no layer index")
      let layerIdx = parseInt(rest[0 ..< dot])
      checkValue(layerIdx >= 0 and layerIdx < cfg.numHiddenLayers,
        "[ttt] censusDerivedSchedule: layer index " & $layerIdx &
        " outside 0.." & $(cfg.numHiddenLayers - 1) & " in key " & $name)
      if ".attention.dense." in name:
        mlaSeen.add layerIdx
      if ".attention.A_log" in name:
        kdaSeen.add layerIdx
        aLogKeys.add name
      if ".attention.dt_bias" in name:
        dtBiasKeys.add name
      if ".mlp.gate.expert_bias" in name:
        biasKeys.add name
    inc result.loadScope

  result.mlaLayers = mlaSeen.len
  result.kdaLayers = kdaSeen.len
  checkValue(mlaSeen.len + kdaSeen.len == cfg.numHiddenLayers,
    "[ttt] censusDerivedSchedule: " & $mlaSeen.len & " MLA and " &
    $kdaSeen.len & " KDA layers do not cover the " & $cfg.numHiddenLayers &
    " layer stack")
  var mlaDerived: seq[int]
  for i in 0 ..< cfg.numHiddenLayers:
    if cfg.layerKinds[i] == alkMla:
      mlaDerived.add i
  checkValue(mlaSeen.sorted == mlaDerived,
    "[ttt] censusDerivedSchedule: MLA layer set " & $mlaSeen.sorted &
    " disagrees with the derived schedule " & $mlaDerived)
  checkValue(biasKeys.len == cfg.numHiddenLayers - cfg.firstKDenseReplace,
    "[ttt] censusDerivedSchedule: bias buffer count " & $biasKeys.len &
    " disagrees with the routed layer count " &
    $(cfg.numHiddenLayers - cfg.firstKDenseReplace))

  let view = SafetensorsCollection.open(modelPath)
  for name in biasKeys:
    let bias = view.getTensorOwned(name, kCPU)
    checkValue(bias.scalarType() == kFloat32,
      "[ttt] censusDerivedSchedule: bias buffer " & $name & " is " &
      $bias.scalarType() & ", expected f32")
  for name in aLogKeys:
    let aLog = view.getTensorOwned(name, kCPU)
    checkValue(aLog.scalarType() == kFloat32,
      "[ttt] censusDerivedSchedule: A_log " & $name & " is " &
      $aLog.scalarType() & ", expected f32")
  for name in dtBiasKeys:
    let dtBias = view.getTensorOwned(name, kCPU)
    checkValue(dtBias.scalarType() == kFloat32,
      "[ttt] censusDerivedSchedule: dt_bias " & $name & " is " &
      $dtBias.scalarType() & ", expected f32")

static:
  # Register the Ling-3.0-tiny (bailing_hybrid) model in the registry
  ModelRegistry["BailingMoeV3ForCausalLM"] = loadLing3Model
