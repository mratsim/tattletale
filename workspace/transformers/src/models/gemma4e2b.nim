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
  std/os,
  std/tables,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/positron,
  workspace/toktoktok,
  ../layers,
  ../quantizations/datatypes,
  ../deserialization,
  ../models/loading/layer_kinds,
  workspace/safetensors/src/collections,
  ../stateful/inference_context,
  ./all_interfaces

{.experimental: "callOperator".}

# ─── gemma-4 E2B configuration ─────────────────────────────────────────────

type
  Gemma4E2BConfig* = ref object
    architecture: string
    model_type: string
    num_hidden_layers*: int
    hidden_size*: int
    vocab_size: int
    rms_norm_eps: float
    torch_dtype: string
    head_dim: int
      ## Per-head width of the sliding_attention layers.
    global_head_dim*: int
      ## Per-head width of the full_attention layers.
    num_attention_heads: int
    num_key_value_heads*: int
    num_kv_shared_layers: int
      ## Tail layer count that projects no k/v and gathers from the last
      ## same-kind layer before the sharing point.
    intermediate_size: int
      ## FFN width of the self-cached layers.
    use_double_wide_mlp: bool
      ## Shared-kv layers widen the FFN to twice `intermediate_size`.
    hidden_size_per_layer_input*: int
      ## Per-layer embedding width (PLE), 0 disables the PLE tail.
    sliding_window: int
      ## Visibility band of the sliding_attention layers.
    max_position_embeddings: int
    final_logit_softcapping*: float
      ## Logit tanh cap of the model forward, 0 disables the cap.
    ropeFullTheta: float64
    ropeFullRotaryDim: int
      ## Rotating width of the full_attention table, head_dim at the proportional spelling.
    ropeFullActivePairs: int
      ## Active pair count of the proportional table,
      ## int(partial_rotary_factor * head_dim / 2).
    ropeSlidingTheta: float64
    layerKinds*: seq[AttentionLayerKind]
      ## Per-layer attention kinds, parsed from the text_config `layer_types` list.
    bos_token_id: int
    eos_token_ids: seq[int]
      ## Stop set of the checkpoint config, an eos id list.

func parseGemma4E2BConfig(json: JsonNode): Gemma4E2BConfig =
  result = new Gemma4E2BConfig

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
  result.num_kv_shared_layers = tc{"num_kv_shared_layers"}.getInt(0)
  result.intermediate_size = tc{"intermediate_size"}.getInt().int
  result.use_double_wide_mlp = tc{"use_double_wide_mlp"}.getBool(false)
  result.hidden_size_per_layer_input = tc{"hidden_size_per_layer_input"}.getInt(0)
  result.sliding_window = tc{"sliding_window"}.getInt().int
  result.max_position_embeddings = tc{"max_position_embeddings"}.getInt().int
  result.final_logit_softcapping = tc{"final_logit_softcapping"}.getFloat(0.0)

  let ropeFull = tc{"rope_parameters"}{"full_attention"}
  checkValue(ropeFull{"rope_type"}.getStr() == "proportional",
    "[ttt] Gemma4E2BConfig: full_attention rope_type is \"" &
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
      "[ttt] Gemma4E2BConfig: layer_types carries " & $result.layerKinds.len &
      " entries for " & $result.num_hidden_layers & " layers")
  else:
    raise newException(ValueError,
      "[ttt] Gemma4E2BConfig: the checkpoint reads its layer kinds off the " &
      "layer_types list, the config carries none")

  result.bos_token_id = tc{"bos_token_id"}.getInt().int
  for e in json{"eos_token_id"}.items():
    result.eos_token_ids.add e.getInt().int

proc loadGemma4E2BConfig(path: string): Gemma4E2BConfig =
  let json = path.parseFile()
  result = parseGemma4E2BConfig(json)

# ─── gemma-4 E2B model ─────────────────────────────────────────────────────

type
  Gemma4E2BDecoderLayer* = SandwichDecoderLayer[
    RopeGQAttention[FusedRmsNorm], GatedDenseFFN, FusedRmsNorm]
    ## Sandwich decoder block. The per-layer-embedding tail and the layer
    ## scalar sit in the model loop, after the sandwich block.

  Gemma4E2BModel* = ref object
    embedTokens: Embedding
    embedScale: Tensor
      ## bf16 scalar `hidden_size^0.5` multiplier of the embedding output
      ## with Gemma4TextScaledWordEmbedding semantics.
    pleEmbedTokens: Embedding
      ## PLE token-identity embedding, constructed directly off the packed
      ## `[vocab, num_layers * ple_dim]` weight of embed_tokens_per_layer.
      ##
      ## Built by hand, the vocab/hidden shape check of Embedding.load
      ## does not fit the packed layout.
    pleTokScale: Tensor
      ## bf16 scalar `ple_dim^0.5` multiplier of the PLE embedding output.
    perLayerModelProjection: Linear
      ## Context-aware PLE projection `[num_layers * ple_dim, hidden]`.
    perLayerProjectionNorm: FusedRmsNorm
      ## Norm over the ple_dim slice of the projected PLE rows.
    layers: seq[Gemma4E2BDecoderLayer]
    perLayerInputGate: seq[Linear]
      ## Per-layer PLE gate weight `[ple_dim, hidden]`.
    perLayerProjection: seq[Linear]
      ## Per-layer PLE projection `[hidden, ple_dim]`.
    postPleNorm: seq[FusedRmsNorm]
      ## Per-layer norm after the PLE projection.
    layerScalars: seq[Tensor]
      ## Per-layer bf16 `[1]` output scale, real checkpoint buffers.
    norm: FusedRmsNorm
    lmHead: LMHead
    config: Gemma4E2BConfig
    rotaryFull: RotaryPositionEmbedding
      ## Proportional rope table of the full_attention layers, the zero-tail pair construction at the global head width.
    rotarySliding: RotaryPositionEmbedding
      ## Rope table of the sliding_attention layers, the default theta.
    tokenizerPath: string
      ## Deferred tokenizer binding, the gemma-4 tokenizer is loaded on first `getTokenizer` use, never at model load.
    tokenizer: BPETokenizer
      ## Nil until the first successful `getTokenizer` call.
    device: DeviceKind

proc forward*(self: Gemma4E2BModel, ctx: var InferenceContext, input_ids: Tensor): Tensor =
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
  ##   x → sandwich block → PLE tail → layer scalar → next layer input
  ##   - the sandwich block
  ##   - the per-layer-embedding tail and the layer scalar multiply, one
  ##     rounding sequence per layer exactly as the reference layer spells it
  ##
  ## The layer scalar scales the whole layer output, residual stream included,
  ## so each layer consumes the previous layer's full output and no residual
  ## crosses a layer boundary.
  let batch = input_ids.size(0)
  let seqLen = input_ids.size(1)
  let numLayers = self.config.num_hidden_layers
  let pleDim = self.config.hidden_size_per_layer_input

  var x = self.embedTokens(input_ids) * self.embedScale

  # Per-Layer Embeddings (token-identity and context-aware halves),
  # combined once and sliced per layer in the loop.
  let pleTok = self.pleEmbedTokens(input_ids).reshape(
    [batch, seqLen, numLayers, pleDim]) * self.pleTokScale
  let pleProjRaw = self.perLayerModelProjection.forward(x) *
    Scalar(pow(self.config.hidden_size.float64, -0.5))
  let pleProj = self.perLayerProjectionNorm.forward(
    pleProjRaw.reshape([batch, seqLen, numLayers, pleDim]))
  let combined = (pleProj + pleTok) * Scalar(pow(2.0, -0.5))

  for i in 0 ..< self.layers.len:
    # Dual-theta rope. Each layer kind ropes with its own table, the rows
    # travel as per-forward request state on the context.
    ctx.setRopeForPositions(
      if self.config.layerKinds[i] == alkSlidingAttention: self.rotarySliding
      else: self.rotaryFull)
    let (post, h1) = self.layers[i].forward(ctx, x, none(Tensor))
    var h = h1 + post

    # Per-layer-embedding tail, in order:
    # the perLayerInputGate multiply, the tanh-gelu, the PLE product,
    # the projection, the norm, then the residual add and the layer scalar.
    let pleInput = combined[_, _, i, _]
    let gated = gelu_tanh(self.perLayerInputGate[i].forward(h)) * pleInput
    h = h + self.postPleNorm[i].forward(self.perLayerProjection[i].forward(gated))
    x = h * self.layerScalars[i]

  let normed = self.norm.forward(x)
  result = self.lmHead.forward(normed)
  if self.config.final_logit_softcapping > 0.0:
    # The reference softcap, divide by the cap, tanh, multiply back.
    let cap = Scalar(self.config.final_logit_softcapping)
    result = (result / cap).tanh() * cap

func getConfig(self: Gemma4E2BModel): ModelConfigBase =
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

func getTokenizer(self: Gemma4E2BModel): BPETokenizer =
  ## Checkpoint tokenizer, loaded on first use.
  ##
  ## Raises:
  ##   - `ValueError` on every call, the gemma-4 tokenizer has no converter
  ##     yet and text tokenization is unavailable
  ##
  ## The suites replay recorded token ids and never encode.
  if self.tokenizer.isNil:
    raise newException(ValueError,
      "[ttt] Gemma4E2BModel: the gemma-4 tokenizer has no converter " &
      "in toktoktok yet, text tokenization is unavailable for " &
      self.tokenizerPath)
  self.tokenizer

func getDeviceKind(self: Gemma4E2BModel): DeviceKind =
  self.device

proc loadGemma4E2BModelRaw(modelPath: string, device: DeviceKind): Gemma4E2BModel =
  let config = loadGemma4E2BConfig(modelPath / "config.json")
  let weights = SafetensorsCollection.open(modelPath)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  checkValue(config.hidden_size_per_layer_input > 0,
    "[ttt] Gemma4E2BConfig: hidden_size_per_layer_input is 0, the port " &
    "implements the per-layer-embedding checkpoint family only")
  checkValue(config.use_double_wide_mlp,
    "[ttt] Gemma4E2BConfig: use_double_wide_mlp is false, the port implements " &
    "the double-wide shared-layer checkpoint family only")

  # The language stack keys sit under the model.language_model prefix,
  # the multimodal towers of the checkpoint stay unread.
  let lp0 = "model.language_model."

  let embedTokens = Embedding.load(weights, cfgJson, lp0 & "embed_tokens", device)
  let embedScale = F.full(1, sqrt(config.hidden_size.float64),
    F.tensorOptions(F.kBFloat16, device))

  let pleEmbedTokens = Embedding.init(
    weights.getTensorOwned(lp0 & "embed_tokens_per_layer.weight", device))
  let pleTokScale = F.full(1, sqrt(config.hidden_size_per_layer_input.float64),
    F.tensorOptions(F.kBFloat16, device))
  let perLayerModelProjection = Linear.init(
    weights.getTensorOwned(lp0 & "per_layer_model_projection.weight", device))
  let perLayerProjectionNorm = FusedRmsNorm.init(
    weights.getTensorOwned(lp0 & "per_layer_projection_norm.weight", device),
    qBF16, config.rms_norm_eps)

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

  let firstShared = config.num_hidden_layers - config.num_kv_shared_layers

  var layers = newSeq[Gemma4E2BDecoderLayer](config.num_hidden_layers)
  var perLayerInputGate = newSeq[Linear](config.num_hidden_layers)
  var perLayerProjection = newSeq[Linear](config.num_hidden_layers)
  var postPleNorm = newSeq[FusedRmsNorm](config.num_hidden_layers)
  var layerScalars = newSeq[Tensor](config.num_hidden_layers)

  for i in 0 ..< config.num_hidden_layers:

    let lp = lp0 & "layers." & $i & "."
    let sliding = config.layerKinds[i] == alkSlidingAttention
    let headDim = if sliding: config.head_dim else: config.global_head_dim

    # Shared-kv layers gather from the last same-kind layer sitting
    # before the sharing point, whose pages carry the full-length kv.
    var kvSource = -1
    if i >= firstShared:
      for j in countdown(i - 1, 0):
        if j < firstShared and config.layerKinds[j] == config.layerKinds[i]:
          kvSource = j
          break

    let inputLN = FusedRmsNorm.load(weights, cfgJson, lp & "input_layernorm", device)
    let postAttnLN = FusedRmsNorm.load(weights, cfgJson, lp & "post_attention_layernorm", device)
    let preFfnLN = FusedRmsNorm.load(weights, cfgJson, lp & "pre_feedforward_layernorm", device)
    let postFfnLN = FusedRmsNorm.load(weights, cfgJson, lp & "post_feedforward_layernorm", device)

    # The value-path norm carries no checkpoint weight, a ones weight
    # leaves the norm unscaled (the with_scale=False spelling).
    let vNorm =
      if kvSource < 0:
        FusedRmsNorm.init(
          F.ones(headDim, F.tensorOptions(F.kBFloat16, device)),
          qBF16, config.rms_norm_eps)
      else:
        nil

    let attn = RopeGQAttention[FusedRmsNorm].load(
      weights, cfgJson, lp & "self_attn", i,
      config.num_attention_heads, config.num_key_value_heads, headDim,
      if sliding: rotarySliding else: rotaryFull, device,
      window = if sliding: config.sliding_window else: FullVisibilityWindow,
      softmaxScale = 1.0,
      kvSourceLayer = kvSource,
      vNorm = vNorm)

    # Shared-kv layers double the FFN width against the config width,
    # the checkpoint weight shapes carry the width.
    let mlp = GatedDenseFFN.load(weights, cfgJson, lp & "mlp", device,
      activation = kGeluTanh)

    perLayerInputGate[i] = Linear.init(
      weights.getTensorOwned(lp & "per_layer_input_gate.weight", device))
    perLayerProjection[i] = Linear.init(
      weights.getTensorOwned(lp & "per_layer_projection.weight", device))
    postPleNorm[i] = FusedRmsNorm.init(
      weights.getTensorOwned(lp & "post_per_layer_input_norm.weight", device),
      qBF16, config.rms_norm_eps)
    layerScalars[i] = weights.getTensorOwned(lp & "layer_scalar", device)

    layers[i] = Gemma4E2BDecoderLayer.init(inputLN, attn, postAttnLN,
      preFfnLN, mlp, postFfnLN)

  let norm = FusedRmsNorm.load(weights, cfgJson, lp0 & "norm", device)
  let lmHead = LMHead.load(weights, cfgJson, embedTokens, device)
  result = Gemma4E2BModel(
    embedTokens: embedTokens,
    embedScale: embedScale,
    pleEmbedTokens: pleEmbedTokens,
    pleTokScale: pleTokScale,
    perLayerModelProjection: perLayerModelProjection,
    perLayerProjectionNorm: perLayerProjectionNorm,
    layers: layers,
    perLayerInputGate: perLayerInputGate,
    perLayerProjection: perLayerProjection,
    postPleNorm: postPleNorm,
    layerScalars: layerScalars,
    norm: norm,
    lmHead: lmHead,
    config: config,
    rotaryFull: rotaryFull,
    rotarySliding: rotarySliding,
    tokenizerPath: modelPath / "tokenizer.json",
    device: device
  )

proc loadGemma4E2BModel*(modelPath: string, device: DeviceKind): AnyModel =
  ## Loads a gemma-4-E2B checkpoint from `modelPath` onto `device`,
  ## wrapped in the AnyModel model interface, the language stack only.
  let gemma4E2BModel = loadGemma4E2BModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  gemma4E2BModel.to(AnyModel)

