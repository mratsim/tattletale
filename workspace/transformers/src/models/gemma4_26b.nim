# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  std/options,
  std/tables,
  std/os,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/positron,
  workspace/toktoktok,
  ../layers,
  ../layers/moe_router,
  ../quantizations/datatypes,
  ../deserialization,
  ../models/loading/layer_kinds,
  ../stateful/inference_context,
  ./all_interfaces

{.experimental: "callOperator".}

# ─── gemma-4 26B configuration ─────────────────────────────────────────────

type
  Gemma4Text26BConfig* = ref object
    architecture: string
    model_type: string
    num_hidden_layers*: int
    hidden_size: int
    vocab_size: int
    rms_norm_eps: float
    torch_dtype: string
    head_dim: int
      ## Per-head width of the sliding_attention layers.
    global_head_dim*: int
      ## Per-head width of the full_attention layers.
    num_attention_heads: int
    num_key_value_heads*: int
      ## KV-head count of the sliding_attention layers, the widest count
      ## in the stack and the page-pool geometry.
    numGlobalKvHeads: int
      ## KV-head count of the full_attention layers,
      ## the config num_global_key_value_heads row.
    intermediate_size: int
      ## Dense mlp width, present beside the routed block on every layer.
    moeIntermediateSize: int
      ## Routed expert width.
    numExperts: int
    topKExperts: int
    sliding_window: int
    max_position_embeddings: int
    final_logit_softcapping*: float
      ## Logit tanh cap of the model forward, 0 disables the cap.
    ropeFullTheta: float64
    ropeFullRotaryDim: int
      ## Rotating width of the full_attention table, head_dim
      ## at the proportional spelling.
    ropeFullActivePairs: int
      ## Active pair count of the proportional table,
      ## int(partial_rotary_factor * head_dim / 2).
    ropeSlidingTheta: float64
    layerKinds*: seq[AttentionLayerKind]
      ## Per-layer attention kinds, parsed
      ## from the text_config `layer_types` list.
    bos_token_id: int
    eos_token_ids: seq[int]
      ## Stop set of the checkpoint config, an eos id list.

func parseGemma4Text26BConfig(json: JsonNode): Gemma4Text26BConfig =
  result = new Gemma4Text26BConfig

  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = json{"model_type"}.getStr()

  # The language stack lives under text_config, the multimodal wrappers
  # carry their own towers this port does not load.
  let tc = json{"text_config"}
  result.num_hidden_layers = tc{"num_hidden_layers"}.getInt().int
  result.hidden_size = tc{"hidden_size"}.getInt().int
  result.vocab_size = tc{"vocab_size"}.getInt().int
  result.rms_norm_eps = tc{"rms_norm_eps"}.getFloat()
  result.torch_dtype = tc{"dtype"}.getStr(
    tc{"torch_dtype"}.getStr("bfloat16"))
  result.head_dim = tc{"head_dim"}.getInt().int
  result.global_head_dim = tc{"global_head_dim"}.getInt(
    tc{"head_dim"}.getInt()).int
  result.num_attention_heads = tc{"num_attention_heads"}.getInt().int
  result.num_key_value_heads = tc{"num_key_value_heads"}.getInt().int
  result.numGlobalKvHeads = tc{"num_global_key_value_heads"}.getInt(
    tc{"num_key_value_heads"}.getInt()).int
  result.intermediate_size = tc{"intermediate_size"}.getInt().int
  result.moeIntermediateSize = tc{"moe_intermediate_size"}.getInt().int
  result.numExperts = tc{"num_experts"}.getInt().int
  result.topKExperts = tc{"top_k_experts"}.getInt().int
  result.sliding_window = tc{"sliding_window"}.getInt().int
  result.max_position_embeddings = tc{"max_position_embeddings"}.getInt().int
  result.final_logit_softcapping = tc{"final_logit_softcapping"}.getFloat(0.0)

  let ropeFull = tc{"rope_parameters"}{"full_attention"}
  checkValue(ropeFull{"rope_type"}.getStr() == "proportional",
    "[ttt] Gemma4Text26BConfig: full_attention rope_type is \"" &
    ropeFull{"rope_type"}.getStr() &
    "\", the port implements the proportional family only")
  result.ropeFullTheta = ropeFull{"rope_theta"}.getFloat()
  result.ropeFullRotaryDim = result.global_head_dim
  result.ropeFullActivePairs = int(
    ropeFull{"partial_rotary_factor"}.getFloat(1.0) *
    result.ropeFullRotaryDim.float64 / 2.0)
  result.ropeSlidingTheta = tc{"rope_parameters"}{"sliding_attention"}{"rope_theta"}.getFloat()

  if tc.hasKey("layer_types"):
    var layerIdx = 0
    for entry in tc{"layer_types"}.items():
      result.layerKinds.add parseAttnFromHfTransformers(entry.getStr(),
        "layer_types[" & $layerIdx & "]")
      inc layerIdx
    checkValue(result.layerKinds.len == result.num_hidden_layers,
      "[ttt] Gemma4Text26BConfig: layer_types carries " &
      $result.layerKinds.len & " entries for " &
      $result.num_hidden_layers & " layers")
  else:
    raise newException(ValueError,
      "[ttt] Gemma4Text26BConfig: the checkpoint reads its layer kinds off " &
      "the layer_types list, the config carries none")

  result.bos_token_id = tc{"bos_token_id"}.getInt().int

  # The checkpoint family states its stop set at the top level or under
  # text_config, an id list or a single id.
  let eosNode = json{"eos_token_id"}
  if eosNode.kind == JArray:
    for e in eosNode.items():
      result.eos_token_ids.add e.getInt().int
  elif eosNode.kind == JInt:
    result.eos_token_ids.add eosNode.getInt().int
  else:
    result.eos_token_ids.add tc{"eos_token_id"}.getInt().int

proc loadGemma4Text26BConfig(path: string): Gemma4Text26BConfig =
  let json = path.parseFile()
  result = parseGemma4Text26BConfig(json)

# ─── gemma-4 26B decoder layer ─────────────────────────────────────────────

type
  Gemma4Text26BDecoderLayer* = ref object
    ## Sandwich decoder block with the routed tail (enable_moe_block).
    ## The dense mlp and the routed block run in parallel off the same
    ## pre-feedforward norm output while the routed branch reads the post-attention residual:
    ##
    ##   h1     = post-attention residual
    ##   dense  = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h1)))
    ##
    ##   routed = post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(h1), route(h1)))
    ##   block  = h1 + post_feedforward_layernorm(dense + routed)
    ##
    ## The layer scalar multiply sits in the model loop, after the block.
    inputLayernorm*: FusedRmsNorm
    postAttentionLayernorm*: FusedRmsNorm
    preFfnLayernorm*: FusedRmsNorm
    postFfnLayernorm*: FusedRmsNorm
    postFfnLayernorm1*: FusedRmsNorm
    preFfnLayernorm2*: FusedRmsNorm
    postFfnLayernorm2*: FusedRmsNorm
    attn*: RopeGQAttention[FusedRmsNorm]
    mlp*: GatedDenseFFN
    router*: SoftmaxTopkRouter
    expertsGateUp*: Tensor
      ## Rank-3 fused routed bodies [num_experts, 2 * moe_intermediate, hidden],
      ## gate_proj rows 0:moe_intermediate, up_proj rows moe_intermediate:2*moe_intermediate.
    expertsDown*: Tensor
      ## Rank-3 routed output bodies [num_experts, hidden, moe_intermediate].
    layerScalar*: Tensor
      ## bf16 `[1]` output scale, a real checkpoint buffer.

func forward*(self: Gemma4Text26BDecoderLayer, ctx: var InferenceContext, x: Tensor, residual: Option[Tensor]): (Tensor, Tensor) =
  ## Forward pass for one routed sandwich decoder block.
  ##
  ## Returns:
  ##
  ## - the pair `(block contribution, carried residual)`
  ## - contribution + residual equals the block output, the caller adds
  ##   the pair at the next block boundary or at the model final
  ##
  ## The routed branch reads the post-attention residual twice, the router
  ## over its shaped rows, the experts over the pre_feedforward_layernorm_2 output.
  ##
  ## The expert compute is the transformers grouped_mm spelling, the f32
  ## routing weights multiply the expert outputs in f32 and the per-token
  ## sum rounds to the hidden dtype once.
  let (hNorm, h) =
    if residual.isSome():
      self.inputLayernorm(x, residual.unsafeGet())
    else:
      (self.inputLayernorm.forward(x), x)

  let mixerOut = self.attn(ctx, hNorm)
  let h1 = h + self.postAttentionLayernorm(mixerOut)

  let h2 = self.preFfnLayernorm(h1)
  let mlpOut = self.mlp(h2)
  let dense = self.postFfnLayernorm1(mlpOut)

  let hiddenSize = h1.size(h1.dim() - 1)
  let rows = h1.numel() div hiddenSize
  let flat = h1.reshape([rows, hiddenSize])
  let decision = self.router(flat)
  let expertsIn = self.preFfnLayernorm2(flat)
  let expertsOut = expertForwardPairs(self.expertsGateUp, self.expertsDown,
    expertsIn, decision.indices, decision.weights, kGeluTanh,
    weightsAtHiddenDtype = false)
  let routed = self.postFfnLayernorm2(expertsOut.reshape(
    [h1.size(0), h1.size(1), hiddenSize]))

  let combined = dense + routed
  let postNormed = self.postFfnLayernorm(combined)
  result = (postNormed, h1)

template `()`*(layer: Gemma4Text26BDecoderLayer,
              ctx: var InferenceContext,
              x: Tensor,
              residual: Option[Tensor]): untyped =
  layer.forward(ctx, x, residual)

# ─── gemma-4 26B model ─────────────────────────────────────────────────────

type
  Gemma4Text26BModel* = ref object
    embedTokens: Embedding
    embedScale: Tensor
      ## bf16 scalar `hidden_size^0.5` multiplier of the embedding output
      ## with Gemma4TextScaledWordEmbedding semantics.
    layers: seq[AnyDecoderLayer]
    layerScalars: seq[Tensor]
      ## Per-layer bf16 `[1]` output scale, real checkpoint buffers.
    norm: FusedRmsNorm
    lmHead: LMHead
    config: Gemma4Text26BConfig
    rotaryFull: RotaryPositionEmbedding
      ## Proportional rope table of the full_attention layers, the zero-tail
      ## pair construction at the global head width.
    rotarySliding: RotaryPositionEmbedding
      ## Rope table of the sliding_attention layers, the default theta.
    tokenizerPath: string
      ## Deferred tokenizer binding, the gemma-4 tokenizer loads at the first `getTokenizer` use, never at model load.
    tokenizer: BPETokenizer
      ## Nil until the first successful `getTokenizer` call.
    device: DeviceKind

proc forward*(self: Gemma4Text26BModel, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Full forward to logits over the input token ids.
  ##
  ## Expected input:
  ##
  ## - `input_ids`, token ids of shape (batch, seq_len)
  ##
  ## Output:
  ##
  ## - logits of shape (batch, seq_len, vocab_size)
  ##
  ## Chain:
  ##   x → routed sandwich block → layer scalar → next layer input
  ##
  ## The layer scalar scales the whole layer output, residual stream included,
  ## so each layer consumes the previous layer's full output and no residual
  ## crosses a layer boundary.
  var x = self.embedTokens(input_ids) * self.embedScale

  for i in 0 ..< self.layers.len:
    # Dual-theta rope. Each layer kind ropes with its own table, the rows
    # travel as per-forward request state on the context.
    ctx.setRopeForPositions(
      if self.config.layerKinds[i] == alkSlidingAttention: self.rotarySliding
      else: self.rotaryFull)
    let (post, h1) = self.layers[i].forward(ctx, x, none(Tensor))
    x = (h1 + post) * self.layerScalars[i]

  let normed = self.norm.forward(x)
  result = self.lmHead.forward(normed)
  if self.config.final_logit_softcapping > 0.0:
    # The reference softcap, divide by the cap, tanh, multiply back.
    let cap = Scalar(self.config.final_logit_softcapping)
    result = (result / cap).tanh() * cap

func getConfig(self: Gemma4Text26BModel): ModelConfigBase =
  ModelConfigBase(
    architecture: self.config.architecture,
    model_type: self.config.model_type,
    num_hidden_layers: self.config.num_hidden_layers,
    hidden_size: self.config.hidden_size,
    vocab_size: self.config.vocab_size,
    rms_norm_eps: self.config.rms_norm_eps.float,
    torch_dtype: self.config.torch_dtype,
    num_attention_heads: self.config.num_attention_heads,
    num_key_value_heads: self.config.num_key_value_heads,
    head_dim: self.config.head_dim,
    intermediate_size: self.config.intermediate_size,
    max_position_embeddings: self.config.max_position_embeddings,
    eosTokenIds: self.config.eos_token_ids,
    layerKinds: self.config.layerKinds,
    kvHeadDimMax: self.config.global_head_dim
  )

func getTokenizer(self: Gemma4Text26BModel): BPETokenizer =
  ## Checkpoint tokenizer, loaded on first use.
  ##
  ## Raises:
  ##   - `ValueError` on every call, the gemma-4 tokenizer has no converter
  ##     yet and text tokenization is unavailable
  ##
  ## The suites replay recorded token ids and never encode.
  if self.tokenizer.isNil:
    raise newException(ValueError,
      "[ttt] Gemma4Text26BModel: the gemma-4 tokenizer has no converter " &
      "in toktoktok yet, text tokenization is unavailable for " &
      self.tokenizerPath)
  self.tokenizer

func getDeviceKind(self: Gemma4Text26BModel): DeviceKind =
  self.device

proc loadGemma4Text26BModelRaw(modelPath: string, device: DeviceKind): Gemma4Text26BModel =
  let config = loadGemma4Text26BConfig(modelPath / "config.json")
  let weights = SafetensorsCollection.open(modelPath)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  checkValue(config.num_hidden_layers mod 6 == 0,
    "[ttt] Gemma4Text26BConfig: the 5:1 sliding/full pattern expects a " &
    "layer count divisible by 6, found " & $config.num_hidden_layers)

  # The language stack keys sit under the model.language_model prefix,
  # the multimodal towers of the checkpoint stay unread.
  let lp0 = "model.language_model."

  let embedTokens = Embedding.load(weights, cfgJson, lp0 & "embed_tokens", device)
  let embedScale = F.full(1, sqrt(config.hidden_size.float64),
    F.tensorOptions(F.kBFloat16, device))

  # Dual-theta tables. The full_attention table is the proportional
  # zero-tail construction at the global head width, the first
  # activePairs pairs rotate and the tail pairs keep a zero angle.
  let rotaryFull = RotaryPositionEmbedding.new(
    config.global_head_dim,
    config.max_position_embeddings,
    config.ropeFullTheta,
    actDtype,
    device,
    rotary_dim = config.ropeFullRotaryDim,
    activePairs = config.ropeFullActivePairs,
    f32Cache = true)
  let rotarySliding = RotaryPositionEmbedding.new(
    config.head_dim,
    config.max_position_embeddings,
    config.ropeSlidingTheta,
    actDtype,
    device,
    f32Cache = true)

  var layers = newSeq[AnyDecoderLayer](config.num_hidden_layers)
  var layerScalars = newSeq[Tensor](config.num_hidden_layers)

  for i in 0 ..< config.num_hidden_layers:

    let lp = lp0 & "layers." & $i & "."
    let sliding = config.layerKinds[i] == alkSlidingAttention
    let headDim = if sliding: config.head_dim else: config.global_head_dim
    let kvHeads = if sliding: config.num_key_value_heads
                  else: config.numGlobalKvHeads

    let inputLN = FusedRmsNorm.load(weights, cfgJson, lp & "input_layernorm", device)
    let postAttnLN = FusedRmsNorm.load(weights, cfgJson, lp & "post_attention_layernorm", device)
    let preFfnLN = FusedRmsNorm.load(weights, cfgJson, lp & "pre_feedforward_layernorm", device)
    let postFfnLN = FusedRmsNorm.load(weights, cfgJson, lp & "post_feedforward_layernorm", device)
    let postFfnLN1 = FusedRmsNorm.load(weights, cfgJson, lp & "post_feedforward_layernorm_1", device)
    let preFfnLN2 = FusedRmsNorm.load(weights, cfgJson, lp & "pre_feedforward_layernorm_2", device)
    let postFfnLN2 = FusedRmsNorm.load(weights, cfgJson, lp & "post_feedforward_layernorm_2", device)

    # The value-path norm carries no checkpoint weight, a ones weight leaves
    # the norm unscaled (the with_scale=False spelling).
    # KV-tied full_attention layers (attention_k_eq_v):
    # no v_proj weight exists.
    #
    # The value rows derive from the shared k projection at forward while
    # the cache keeps its separate K and V entries.
    let vNorm = FusedRmsNorm.init(
      F.ones(headDim, F.tensorOptions(F.kBFloat16, device)),
      qBF16, config.rms_norm_eps)

    let attn = RopeGQAttention[FusedRmsNorm].load(
      weights, cfgJson, lp & "self_attn", i,
      config.num_attention_heads, kvHeads, headDim,
      if sliding: rotarySliding else: rotaryFull, device,
      window = if sliding: config.sliding_window else: FullVisibilityWindow,
      softmaxScale = 1.0,
      vNorm = vNorm,
      kEqV = not sliding)

    let mlp = GatedDenseFFN.load(weights, cfgJson, lp & "mlp", device,
      activation = kGeluTanh)

    # Routed block:
    # softmax top-k router over 128 experts with top 8.
    #
    # The decision weights stay f32 from the softmax through the expert
    # multiply and the per-expert scale sits in the router decision.
    let router = SoftmaxTopkRouter.init(
      weights.getTensorOwned(lp & "router.proj.weight", device),
      weights.getTensorOwned(lp & "router.scale", device),
      weights.getTensorOwned(lp & "router.per_expert_scale", device),
      config.topKExperts, config.rms_norm_eps, device)

    let layerScalarsI = weights.getTensorOwned(lp & "layer_scalar", device)

    let layer = Gemma4Text26BDecoderLayer(
      inputLayernorm: inputLN,
      postAttentionLayernorm: postAttnLN,
      preFfnLayernorm: preFfnLN,
      postFfnLayernorm: postFfnLN,
      postFfnLayernorm1: postFfnLN1,
      preFfnLayernorm2: preFfnLN2,
      postFfnLayernorm2: postFfnLN2,
      attn: attn,
      mlp: mlp,
      router: router,
      expertsGateUp: weights.getTensorOwned(lp & "experts.gate_up_proj", device),
      expertsDown: weights.getTensorOwned(lp & "experts.down_proj", device),
      layerScalar: layerScalarsI
    )
    layers[i] = layer.to(AnyDecoderLayer)
    layerScalars[i] = layerScalarsI

  let norm = FusedRmsNorm.load(weights, cfgJson, lp0 & "norm", device)
  let lmHead = LMHead.load(weights, cfgJson, embedTokens, device)
  result = Gemma4Text26BModel(
    embedTokens: embedTokens,
    embedScale: embedScale,
    layers: layers,
    layerScalars: layerScalars,
    norm: norm,
    lmHead: lmHead,
    config: config,
    rotaryFull: rotaryFull,
    rotarySliding: rotarySliding,
    tokenizerPath: modelPath / "tokenizer.json",
    device: device
  )

proc loadGemma4Text26BModel*(modelPath: string, device: DeviceKind): AnyModel =
  ## Loads a gemma-4-26B-A4B checkpoint from `modelPath` onto `device`,
  ## wrapped in the AnyModel model interface, the language stack only.
  let model = loadGemma4Text26BModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  model.to(AnyModel)
