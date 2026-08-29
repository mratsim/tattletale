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
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/toktoktok,
  ../layers,
  ../layers/mixtures_of_experts,
  ../deserialization,
  ../model/loading/config_json,
  ../model/loading/generation_config,
  ../model/loading/layer_kinds,
  ../stateful/inference_context,
  ../safetensors/collection

{.experimental: "callOperator".}

################################################################################
#                     Qwen3.5 MoE Configuration                                #
################################################################################
#
# Config and generation-config readers for the `qwen3_5_moe` text stack:
# hybrid of Gated DeltaNet and gated full attention, with a routed
# Mixture-of-Experts block in every layer.
#
# `text_config` carries no `intermediate_size` key (no dense MLP). Routed width
# is `moe_intermediate_size`, shared width is `shared_expert_intermediate_size`.
# The file's only `intermediate_size` belongs to `vision_config`, which nothing consumes.

const
  # Last-resort rope values, reached only when no rope level spells the key:
  # `0.25` (back-compat default, `configuration_qwen3_5_moe.py:124`) and `1e7`,
  # the `rope_theta` of every checkpoint in scope, read with no fallback
  # at `modeling_qwen3_5_moe.py:118`.
  DefaultRopeTheta = 1.0e7
  DefaultPartialRotaryFactor = 0.25

type
  Qwen35MoeConfig* = ref object
    ## Text-stack config of a `qwen3_5_moe` checkpoint, parsed from the nested
    ## `text_config` object of the `Qwen3_5MoeForConditionalGeneration` wrapper
    ## config.json.
    ##
    ## Field sources:
    ## - wrapper level: `architecture`, `wrapperModelType`, `imageTokenId`,
    ##   `videoTokenId`, `transformersVersion`
    ## - `text_config.rope_parameters`: `ropeType`, `ropeTheta`,
    ##   `mropeInterleaved`, `mropeSection`
    ## - `text_config` first, `text_config.rope_parameters` as the fallback:
    ##   `partialRotaryFactor`
    ## - `text_config`: every remaining field
    ##
    ## Weight-prefix scope of a text-only load, read from the checkpoint index:
    ## - in scope: `model.language_model.*`, `lm_head.weight`
    ## - out of scope: `model.visual.*` (the vision tower)
    ## - out of scope: `mtp.*` (the multi-token-prediction draft block)
    architecture*: string     ## Wrapper `architectures[0]`, the registry key
    wrapperModelType*: string ## Wrapper `model_type`, the family name
    modelType*: string        ## `text_config.model_type`
    transformersVersion*: string

    vocabSize*: int
    hiddenSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int
    numKeyValueHeads*: int
    headDim*: int             ## Per-head width of a full-attention layer
    intermediateSize*: Option[int]
      ## Dense-MLP width, `none` when absent or null (no dense MLP in this stack).
    mlpOnlyLayers*: seq[int]
      ## Layer indices that would keep a dense MLP. Inert: the vendored tree
      ## deletes the attribute (`modular_qwen3_5_moe.py:118` and `:122`), so no
      ## layer escapes the routed block.

    # Routed block
    numExperts*: int          ## Router output width, one logit per expert
    numExpertsPerTok*: int    ## Experts selected per token (top-k)
    moeIntermediateSize*: int ## Routed-expert MLP width
    sharedExpertIntermediateSize*: int ## Shared-expert MLP width
    outputRouterLogits*: bool ## Emit router logits, the entry point of the aux-loss path
    routerAuxLossCoef*: float64 ## Load-balancing aux-loss weight, consumed
                                ## only when `outputRouterLogits` is set

    # Gated DeltaNet block
    linearNumKeyHeads*: int
    linearKeyHeadDim*: int
    linearNumValueHeads*: int
    linearValueHeadDim*: int
    linearConvKernelDim*: int ## Depthwise conv kernel width, the state keeps this many columns
    mambaSsmDtype*: string    ## Inert: the DeltaNet rule casts to fp32 and reads no such key

    # Full-attention layers
    fullAttentionInterval*: int
      ## Inert: layer kinds come from `layer_types`, which every checkpoint ships.
    attnOutputGate*: bool
      ## Inert: the attention block doubles `q_proj` width unconditionally,
      ## reading no such key.
    layerTypes*: seq[AttentionLayerKind]
      ## Per-layer kind, parsed from the `layer_types` HF key:
      ## alkGatedDeltaNet (Gated DeltaNet) or alkAttention

    # Numerics and position handling
    rmsNormEps*: float64      ## RMSNorm epsilon, applied with the weight as `1 + weight`
    hiddenAct*: string        ## Expert MLP activation
    maxPositionEmbeddings*: int
    dtype*: string            ## Activation dtype name, read from the `dtype` key
    tieWordEmbeddings*: bool  ## Advisory: the safetensors index decides at load time
                              ## (an `lm_head.weight` entry means untied).
    ropeType*: string
    ropeTheta*: float64
    partialRotaryFactor*: float64 ## Fraction of the head that rotates
    mropeInterleaved*: bool   ## Inert for text: equal position axes make the interleave an identity
    mropeSection*: seq[int]

    # Draft-block footprint, consumed by nothing here.
    mtpNumHiddenLayers*: int
    mtpUseDedicatedEmbeddings*: bool

    # Token ids
    bosTokenId*: Option[int]
    textEosTokenId*: int      ## `text_config.eos_token_id` scalar. The stop set
                              ## a generator applies lives in `generation_config.json`
    padTokenId*: Option[int]  ## `none` when absent or null
    imageTokenId*: int        ## Multimodal placeholder, out of text-only scope
    videoTokenId*: int        ## Multimodal placeholder, out of text-only scope

################################################################################
#                          Qwen3.5 MoE Parsing                                  #
################################################################################


proc parseQwen35MoeConfig(json: JsonNode): Qwen35MoeConfig =
  ## Parse a `qwen3_5_moe` wrapper config.json into the text-stack config.
  ##
  ## Four read styles cover every key:
  ## - `reqPosInt`/`reqPosFloat` keys must be present and positive. Absent,
  ##   null, wrong-typed and non-positive values all raise `ValueError`
  ##   naming the key
  ## - `text_config.eos_token_id` goes through `reqInt`: a present `JInt`,
  ##   no positive check
  ## - `optInt` keys accept a present `JInt` or a null, and raise
  ##   `ValueError` for any other kind
  ## - every remaining key goes to a packedjson getter, which answers a default
  ##   for a missing, null or wrong-typed key. The default is the one written
  ##   at the read, as `getInt(4)`, or the getter's own 0, `""`, false, 0.0
  ##
  ## Raises `ValueError` when the wrapper has no `architectures[]` entry or no
  ## `text_config` object.
  let textCfg = json{"text_config"}
  if textCfg.kind != JObject:
    raise newException(ValueError,
      "[ttt] Qwen35MoeConfig.parse: text_config is missing or not an object, found " &
      $textCfg.kind)
  let ropeParams = textCfg{"rope_parameters"}

  let archs = json{"architectures"}
  if archs.kind != JArray or archs.len == 0:
    raise newException(ValueError, "[ttt] No architectures found in config.json")

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
  result.sharedExpertIntermediateSize =
    textCfg{"shared_expert_intermediate_size"}.reqPosInt("shared_expert_intermediate_size")
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
  # Parsed at the caller: every `layer_types` array entry must be a string,
  # and the error quotes the array path with the entry index.
  result.layerTypes = newSeq[AttentionLayerKind]()
  let rawKinds = textCfg{"layer_types"}
  if rawKinds.kind == JArray:
    for i in 0 ..< rawKinds.len:
      let elem = rawKinds[i]
      if elem.kind != JString:
        raise newException(ValueError,
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
  # The two rope keys read in opposite orders. Each order matches one line
  # of the vendored `modeling_rope_utils.py`: `rope_theta` fills a gap only
  # (`:786` setdefault, a `rope_parameters` entry survives),
  # `partial_rotary_factor` overwrites under an `is not None` guard
  # (`:788`, a set `text_config` attribute wins over a `rope_parameters` entry).
  result.ropeTheta = ropeParams{"rope_theta"}.getFloat(
    textCfg{"rope_theta"}.getFloat(DefaultRopeTheta))
  result.partialRotaryFactor = textCfg{"partial_rotary_factor"}.getFloat(
    ropeParams{"partial_rotary_factor"}.getFloat(DefaultPartialRotaryFactor))
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

  # One attention kind per layer of the stack. A positive `num_hidden_layers`,
  # enforced by the `reqPosInt` reader earlier in this parse.
  if result.layerTypes.len != result.numHiddenLayers:
    raise newException(ValueError,
      "[ttt] Qwen35MoeConfig: layer_types has " & $result.layerTypes.len &
      " entries, expected one attention kind per layer of the " &
      $result.numHiddenLayers & "-layer stack")


proc loadQwen35MoeConfig*(path: string): Qwen35MoeConfig =
  ## Load a `qwen3_5_moe` config.json from disk and parse it. Raises
  ## `ValueError` for a config with no `architectures[]` entry, no
  ## `text_config` object, an invalid `layer_types` list, or an essential key
  ## that is absent, wrong-typed or not positive.
  parseFile(path).parseQwen35MoeConfig()


const LmHeadKey* = "lm_head.weight"


################################################################################
#                     Qwen3.5 MoE Model Assembly                               #
################################################################################

type
  Qwen35MoeDecoderLayer* = ref object
    ## One of the 40 hybrid decoder layers. Exactly one attention variant:
    ## non-nil `gdn` marks the alkGatedDeltaNet layers, non-nil
    ## `attn` marks the attention layers. Each layer carries
    ## a routed block, no dense MLP ships in this checkpoint. Residual
    ## pattern: local residuals, BF16 additions, matching the vendored
    ## Qwen35MoeDecoderLayer.forward.
    layer_type*: AttentionLayerKind
    cfg: Qwen35MoeConfig ## Shared geometry, `numExpertsPerTok` drives the router
    input_layernorm: RmsNorm                     # weight applied as 1 + w
    gdn: GatedDeltaNet                     # nil on attention layers
    attn: RopeGQAttention             # nil on Gated DeltaNet layers
    post_attention_layernorm: RmsNorm            # weight applied as 1 + w
    routerWeight: Tensor                   ## [E, H] the routed-block router
    experts: MixtureOfExperts               ## [E, 2I, H] / [E, H, I] fused expert body
    sharedExpert: GatedMLP                 ## Width `sharedExpertIntermediateSize`
    sharedGateWeight: Tensor               ## [1, H] per-token shared-expert gate
    hiddenSize: int

proc initMoeLayer(
    layer_type: AttentionLayerKind,
    cfg: Qwen35MoeConfig,
    input_layernorm, post_attention_layernorm: RmsNorm,
    gdn: GatedDeltaNet,
    attn: RopeGQAttention,
    routerWeight: Tensor,
    experts: MixtureOfExperts,
    sharedExpert: GatedMLP,
    sharedGateWeight: Tensor,
    hiddenSize: int): Qwen35MoeDecoderLayer =
  ## Assemble one hybrid decoder layer. Exactly one of `gdn` / `attn`
  ## is non-nil, selected by `layer_type`: alkGatedDeltaNet or alkAttention.
  ## Every layer carries the routed block.
  Qwen35MoeDecoderLayer(
    layer_type: layer_type,
    cfg: cfg,
    input_layernorm: input_layernorm,
    gdn: gdn,
    attn: attn,
    post_attention_layernorm: post_attention_layernorm,
    routerWeight: routerWeight,
    experts: experts,
    sharedExpert: sharedExpert,
    sharedGateWeight: sharedGateWeight,
    hiddenSize: hiddenSize
  )

proc forward*(
    self: Qwen35MoeDecoderLayer,
    ctx: var InferenceContext,
    hidden: Tensor,
    routedRecord: ptr SparseMoeResult = nil
  ): Tensor =
  ## Run one hybrid decoder layer with local residuals, the vendored
  ## Qwen35MoeDecoderLayer.forward. Recomputed sequence:
  ##
  ##   h = input_layernorm(hidden)
  ##   h = hidden + gdn_or_attn(ctx, h)
  ##   h = post_attention_layernorm(h)
  ##   h = h + routed_block(h)
  ##
  ## Dispatch on `layer_type`:
  ## - alkGatedDeltaNet: the Gated DeltaNet forward, conv + SSM state in ctx
  ## - alkAttention: the softmax attention, KV pages in ctx
  ##
  ## The routed block is spelled
  ## over [B*T, hiddenSize], the flattened view the reference reaches
  ## through `view(-1, hidden_dim)`.
  ##
  ## When `routedRecord` is non-nil, that record receives the full
  ## routed-block result: router logits, top-k indices and weights
  ## and the shared gate. Otherwise the record is discarded.
  let residual = hidden
  let hNorm = self.input_layernorm.forward(hidden)
  let attnOut =
    if self.layer_type == alkGatedDeltaNet:
      self.gdn(ctx, hNorm)
    else:
      self.attn(ctx, hNorm)
  let h1 = residual + attnOut
  let residual2 = h1
  let hNorm2 = self.post_attention_layernorm.forward(h1)

  let batchTokens = hNorm2.numel() div self.hiddenSize
  let moe = sparseMoeForward(
    self.cfg.numExpertsPerTok,
    hNorm2.reshape(batchTokens, self.hiddenSize),
    self.routerWeight,
    self.experts,
    self.sharedExpert,
    self.sharedGateWeight)
  if routedRecord != nil:
    routedRecord[] = moe

  result = residual2 + moe.output.reshape(
    hNorm2.size(0), hNorm2.size(1), self.hiddenSize)

template `()`*(layer: Qwen35MoeDecoderLayer,
            ctx: var InferenceContext,
            x: Tensor): untyped =
  layer.forward(ctx, x)

type
  Qwen35MoeModel* = ref object
    ## Text stack of Qwen3_5MoeForConditionalGeneration: the embedding,
    ## 40 hybrid decoder layers, the final norm, the lm_head. Weight
    ## prefixes: `model.language_model.*` plus `lm_head.weight`.
    embedTokens: Embedding
    layers: seq[Qwen35MoeDecoderLayer]
    norm: RmsNorm                                # weight applied as 1 + w
    lmHead: LMHead
    rotary: RotaryPositionEmbeddingRef
    config*: Qwen35MoeConfig
    tokenizer*: BPETokenizer
    device*: DeviceKind
    loadedTensorCount*: int  ## Name-based tensor requests made by the loader:
                             ## one per checkpoint key, language_model keys
                             ## plus the head entry when untied, nothing
                             ## outside those prefixes is requested

proc forward*(self: Qwen35MoeModel, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Text forward pass: embed -> 40 hybrid layers -> RMSNorm -> lm_head.
  ##
  ## Each layer runs `Qwen35MoeDecoderLayer.forward`, the layer type
  ## `config.layerTypes[i]` picks the attention variant.
  ## `ctx.setRopeForPositions` runs once per forward. Only full-attn
  ## layers read ctx.cos/sin, the GDN layers carry no rope. GDN
  ## per-sequence state lives in ctx, conv + SSM, full-attn state
  ## in ctx.pages.
  var h = self.embedTokens.forward(input_ids)

  ctx.setRopeForPositions(self.rotary)

  for layer in self.layers:
    h = layer.forward(ctx, h)

  let normed = self.norm.forward(h)
  self.lmHead.forward(normed)

template `()`*(model: Qwen35MoeModel,
            ctx: var InferenceContext,
            x: Tensor): untyped =
  model.forward(ctx, x)

proc loadQwen35MoeModelRaw*(modelPath: string, device = kCPU): Qwen35MoeModel =
  ## Load the Qwen3.6-35B-A3B text stack from the 26-shard checkpoint
  ## directory `modelPath` through the index-backed SafetensorsCollection view.
  ##
  ## Tensor requests are name-based under the ConditionalGeneration prefix
  ## prefix set: 692 `model.language_model.*` keys. An untied
  ## checkpoint adds `lm_head.weight` (shard 26) as one more key.
  ## Foreign prefixes `model.visual.*` and `mtp.*` stay unrequested.
    ##
  ## Head lane decided from the index, never from one shard's key set:
  ## an `lm_head.weight` entry in the checkpoint index proves the untied
  ## head, and the loader fails loudly when `tie_word_embeddings` is false
  ## and the entry is missing. The untied lane requests the head tensor.
  ## The tied lane builds on the embedding reference, no head request.
  ## Both lanes carry the [248320, 2048] shape, so a wrong pick stays
  ## undetectable by shape.
  ##
  ## Every layer loads its two RMSNorms (weights applied as 1 + w), its routed block
  ## with the router, the fused rank-3 expert body and the sigmoid-gated
  ## shared expert, keyed by layer type under the `linear_attn.*` or `self_attn.*` prefixes.
  ## The final norm carries the 1 + w weight spelling too. No dense-MLP tensors ship: the text config lacks
  ## `intermediate_size`, so `GatedMLP` only wraps the shared expert.
  ##
  ## All requested tensors materialize as owned copies before the view
  ## closes. Every checkpoint defect caught along the view, the config
  ## parser or the module inits raises ValueError naming the tensor key.
  let config = loadQwen35MoeConfig(modelPath / "config.json")
  if config.hiddenAct != "silu":
    raise newException(ValueError,
      "[ttt] loadQwen35MoeModelRaw: unsupported hidden_act \"" &
      config.hiddenAct & "\", the routed block implements SiLU")
  let view = openSafetensorsCollection(modelPath)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = activationDtype(cfgJson)

  var tensorRequests = 0

  let embedWeight = view.getTensor("model.language_model.embed_tokens.weight", device)
  inc tensorRequests
  doAssert embedWeight.size(0) == config.vocabSize
  doAssert embedWeight.size(1) == config.hiddenSize
  let embedTokens = Embedding.init(embedWeight)

  let rotary = RotaryPositionEmbeddingRef.new(
    config.headDim,
    config.maxPositionEmbeddings,
    config.ropeTheta,
    actDtype,
    device,
    rotary_dim = int(config.headDim.float64 * config.partialRotaryFactor))

  var layers = newSeq[Qwen35MoeDecoderLayer](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lp = "model.language_model.layers." & $i & "."

    let inputLN = RmsNorm.init(
      view.getTensor(lp & "input_layernorm.weight", device),
      eps = config.rmsNormEps, constant_bias = 1.0)
    inc tensorRequests
    let postLN = RmsNorm.init(
      view.getTensor(lp & "post_attention_layernorm.weight", device),
      eps = config.rmsNormEps, constant_bias = 1.0)
    inc tensorRequests

    # Routed block, carrying the same 7 tensor keys per layer, the keys
    # gen_qwen36_layer_fixtures.py records: router, fused rank-3 expert
    # body, shared-expert gate/up/down and the per-token shared-expert gate.
    let routerWeight = view.getTensor(lp & "mlp.gate.weight", device)
    inc tensorRequests
    let gateUpProj = view.getTensor(lp & "mlp.experts.gate_up_proj", device)
    inc tensorRequests
    let downProj = view.getTensor(lp & "mlp.experts.down_proj", device)
    inc tensorRequests
    let sharedG = view.getTensor(lp & "mlp.shared_expert.gate_proj.weight", device)
    inc tensorRequests
    let sharedU = view.getTensor(lp & "mlp.shared_expert.up_proj.weight", device)
    inc tensorRequests
    let sharedD = view.getTensor(lp & "mlp.shared_expert.down_proj.weight", device)
    inc tensorRequests
    let sharedGateWeight = view.getTensor(lp & "mlp.shared_expert_gate.weight", device)
    inc tensorRequests
    let experts = MixtureOfExperts.init(gateUpProj, downProj)
    let sharedExpert = GatedMLP.init(sharedG, sharedU, sharedD)

    var gdn: GatedDeltaNet = nil
    var attn: RopeGQAttention = nil
    if config.layerTypes[i] == alkGatedDeltaNet:
      let qkvProj = Linear.init(view.getTensor(lp & "linear_attn.in_proj_qkv.weight", device))
      inc tensorRequests
      let zProj = Linear.init(view.getTensor(lp & "linear_attn.in_proj_z.weight", device))
      inc tensorRequests
      let aProj = Linear.init(view.getTensor(lp & "linear_attn.in_proj_a.weight", device))
      inc tensorRequests
      let bProj = Linear.init(view.getTensor(lp & "linear_attn.in_proj_b.weight", device))
      inc tensorRequests
      let convWeight = view.getTensor(lp & "linear_attn.conv1d.weight", device)
      inc tensorRequests
      let aLog = view.getTensor(lp & "linear_attn.A_log", device)
      inc tensorRequests
      let dtBias = view.getTensor(lp & "linear_attn.dt_bias", device)
      inc tensorRequests
      let gdnNorm = RmsNormGated.init(
        view.getTensor(lp & "linear_attn.norm.weight", device),
        eps = config.rmsNormEps)
      inc tensorRequests
      let outProj = Linear.init(view.getTensor(lp & "linear_attn.out_proj.weight", device))
      inc tensorRequests
      gdn = GatedDeltaNet.init(
        i, lp & "linear_attn",
        qkvProj, zProj, aProj, bProj,
        convWeight, aLog, dtBias, gdnNorm, outProj,
        config.linearNumKeyHeads,
        config.linearNumValueHeads,
        config.linearKeyHeadDim,
        config.linearValueHeadDim,
        config.linearConvKernelDim)
    else:
      let qProj = Linear.init(view.getTensor(lp & "self_attn.q_proj.weight", device))
      inc tensorRequests
      let kProj = Linear.init(view.getTensor(lp & "self_attn.k_proj.weight", device))
      inc tensorRequests
      let vProj = Linear.init(view.getTensor(lp & "self_attn.v_proj.weight", device))
      inc tensorRequests
      let oProj = Linear.init(view.getTensor(lp & "self_attn.o_proj.weight", device))
      inc tensorRequests
      let qNorm = RmsNorm.init(
        view.getTensor(lp & "self_attn.q_norm.weight", device),
        eps = config.rmsNormEps, constant_bias = 1.0)
      inc tensorRequests
      let kNorm = RmsNorm.init(
        view.getTensor(lp & "self_attn.k_norm.weight", device),
        eps = config.rmsNormEps, constant_bias = 1.0)
      inc tensorRequests
      attn = RopeGQAttention.init(
        i, lp & "self_attn",
        qProj, kProj, vProj, oProj,
        config.numAttentionHeads,
        config.numKeyValueHeads,
        config.headDim,
        rotary,
        q_norm = some(qNorm), k_norm = some(kNorm), fused_gate = true)

    layers[i] = initMoeLayer(
      config.layerTypes[i], config, inputLN, postLN, gdn, attn,
      routerWeight, experts, sharedExpert, sharedGateWeight, config.hiddenSize)

  let norm = RmsNorm.init(
    view.getTensor("model.language_model.norm.weight", device),
    eps = config.rmsNormEps, constant_bias = 1.0)
  inc tensorRequests

  # Head lane decided from the index, proven by an `lm_head.weight`
  # index entry: untied, requests the head tensor here. Tied: reuses
  # the embedding storage. The load refuses loudly on a false tie
  # flag with no entry: the head would silently become the embedding,
  # and the equal shapes cannot expose that pick.
  var lmHead: LMHead
  if view.hasTensor(LmHeadKey):
    let headWeight = view.getTensor(LmHeadKey, device)
    inc tensorRequests
    doAssert headWeight.size(0) == config.vocabSize
    doAssert headWeight.size(1) == config.hiddenSize
    lmHead = LMHead.init(headWeight)
  elif config.tieWordEmbeddings:
    lmHead = LMHead.initTied(embedTokens)
  else:
    raise newException(ValueError,
      "[ttt] tie_word_embeddings is false and lm_head.weight is absent from " &
      "the checkpoint index " & view.indexFilename &
      ", the head would silently become the embedding")

  close(view)

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
    loadedTensorCount: tensorRequests
  )

