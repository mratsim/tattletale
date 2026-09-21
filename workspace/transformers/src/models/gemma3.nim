# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
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
  ../deserialization,
  ../models/loading/layer_kinds,
  workspace/safetensors/src/collections,
  ../stateful/inference_context,
  ./all_interfaces

{.experimental: "callOperator".}

# ─── Gemma3 configuration ──────────────────────────────────────────────────

func deriveGemma3LayerKinds(numLayers, pattern: int): seq[AttentionLayerKind] =
  ## Per-layer attention kinds of a checkpoint that ships no explicit
  ## `layer_types` list. The HF `sliding_window_pattern` period then
  ## derives the schedule, sliding layers by default.
  ##
  ##   - Full attention sits at 0-based indices congruent to pattern - 1 (mod pattern)
  ##
  ##   deriveGemma3LayerKinds(18, 6) → full attention at 5, 11, 17
  for i in 0 ..< numLayers:
    result.add(if (i + 1) mod pattern == 0: alkAttention else: alkSlidingAttention)

type
  Gemma3Config* = ref object
    architecture: string
    model_type: string
    num_hidden_layers*: int
    hidden_size: int
    vocab_size: int
    rms_norm_eps: float
    torch_dtype: string
    num_attention_heads: int
    num_key_value_heads*: int
    head_dim*: int
    intermediate_size: int
    hidden_activation: string
    max_position_embeddings: int
    rope_theta: float64
      ## Global RoPE theta of the full_attention layers.
    rope_local_base_freq: float64
      ## Local RoPE base of the sliding_attention layers.
    query_pre_attn_scalar: int
      ## Denominator of the attention softmax scale.
    sliding_window: int
      ## Visibility band of the sliding_attention layers.
    sliding_window_pattern: int
      ## Period of the sliding/full schedule for configs without a `layer_types` list.
    layerKinds*: seq[AttentionLayerKind]
      ## Per-layer attention kinds, parsed or derived at load.
    bos_token_id: int
    eos_token_id: int
      ## Stop token when the config ships a single eos id.
    eos_token_ids: seq[int]
      ## Stop set when the config ships an eos id list.

func parseGemma3Config(json: JsonNode): Gemma3Config =
  result = new Gemma3Config

  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = json{"model_type"}.getStr()
  result.vocab_size = json{"vocab_size"}.getInt().int
  result.hidden_size = json{"hidden_size"}.getInt().int
  result.num_hidden_layers = json{"num_hidden_layers"}.getInt().int
  result.rms_norm_eps = json{"rms_norm_eps"}.getFloat()
  result.torch_dtype = json{"torch_dtype"}.getStr("bfloat16")
  result.num_attention_heads = json{"num_attention_heads"}.getInt().int
  result.num_key_value_heads = json{"num_key_value_heads"}.getInt().int
  result.head_dim = json{"head_dim"}.getInt().int
  result.intermediate_size = json{"intermediate_size"}.getInt().int
  result.hidden_activation = json{"hidden_activation"}.getStr()
  result.max_position_embeddings = json{"max_position_embeddings"}.getInt().int
  result.rope_theta = json{"rope_theta"}.getFloat()
  result.rope_local_base_freq = json{"rope_local_base_freq"}.getFloat()
  result.query_pre_attn_scalar = json{"query_pre_attn_scalar"}.getInt().int
  result.sliding_window = json{"sliding_window"}.getInt().int
  result.sliding_window_pattern = json{"sliding_window_pattern"}.getInt(
    json{"_sliding_window_pattern"}.getInt(6))
  result.bos_token_id = json{"bos_token_id"}.getInt().int

  # The attention schedule comes from an explicit `layer_types` list when
  # the checkpoint ships one (gemma-3-270m-it), otherwise the window
  # pattern derives it (gemma-3-1b-it).
  if json.hasKey("layer_types"):
    var layerIdx = 0
    for entry in json{"layer_types"}.items():
      result.layerKinds.add parseAttnFromHfTransformers(entry.getStr(),
        "layer_types[" & $layerIdx & "]")
      inc layerIdx
    checkValue(result.layerKinds.len == result.num_hidden_layers,
      "[ttt] Gemma3Config: layer_types carries " & $result.layerKinds.len &
      " entries for " & $result.num_hidden_layers & " layers")
  else:
    result.layerKinds = deriveGemma3LayerKinds(
      result.num_hidden_layers, result.sliding_window_pattern)

  # eos_token_id ships as an int or as a stop-set list, gemma-3-1b-it
  # lists [1, 106].
  let eos = json{"eos_token_id"}
  if eos.kind == JArray:
    for e in eos.items():
      result.eos_token_ids.add e.getInt().int
  else:
    result.eos_token_id = eos.getInt().int

proc loadGemma3Config(path: string): Gemma3Config =
  let json = path.parseFile()
  result = parseGemma3Config(json)

# ─── Gemma3 model ──────────────────────────────────────────────────────────

type
  Gemma3DecoderLayer* = SandwichDecoderLayer[
    RopeGQAttention[RmsNormOne], GatedDenseFFN, RmsNormOne]

  Gemma3Model* = ref object
    embedTokens: Embedding
    embedScale: Tensor
      ## bf16 scalar `hidden_size^0.5` multiplier of the embedding output, Gemma3TextScaledWordEmbedding semantics
    layers: seq[Gemma3DecoderLayer]
    norm: RmsNormOne
    lmHead: LMHead
    config: Gemma3Config
    rotaryFull: RotaryPositionEmbedding
      ## RoPE table of the full_attention layers, the global rope_theta.
    rotarySliding: RotaryPositionEmbedding
      ## RoPE table of the sliding_attention layers, the local base freq.
    tokenizerPath: string
      ## Deferred tokenizer binding, the gemma-3 metaspace tokenizer is
      ## loaded on first `getTokenizer` use, never at model load.
    tokenizer: BPETokenizer
      ## Nil until the first successful `getTokenizer` call.
    device: DeviceKind

proc forward*(self: Gemma3Model, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Full forward to logits over the input token ids.
  ##
  ## Expected input:
  ##
  ## - `input_ids`, token ids of shape (batch, seq_len)
  ##
  ## Output:
  ##
  ## - logits of shape (batch, seq_len, vocab_size)

  var x = self.embedTokens(input_ids) * self.embedScale

  var residual: Option[Tensor]
  for i in 0 ..< self.layers.len:
    # Dual-theta rope. Each layer kind ropes with its own table, the rows
    # travel as per-forward request state on the context.
    ctx.setRopeForPositions(
      if self.config.layerKinds[i] == alkSlidingAttention: self.rotarySliding
      else: self.rotaryFull)
    let layerOut = self.layers[i].forward(ctx, x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])

  let finalResidual = residual.get(x)
  let normed = self.norm(x + finalResidual)
  result = self.lmHead(normed)

func getConfig(self: Gemma3Model): ModelConfigBase =
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
    eosTokenId: self.config.eos_token_id,
    eosTokenIds: self.config.eos_token_ids,
    layerKinds: self.config.layerKinds
  )

func getTokenizer(self: Gemma3Model): BPETokenizer =
  ## Checkpoint tokenizer, loaded on first use.
  ##
  ## Raises:
  ##   - `ValueError` on every call, the gemma-3 metaspace tokenizer has
  ##     no converter yet and text tokenization is unavailable
  ##
  ## The gemma-3 tokenizer is a SentencePiece metaspace family
  ## no HF-to-tiktoken converter represents, hence the deferred binding:
  ##
  ##   - the normalizer maps spaces to `U+2581` and the pre_tokenizer
  ##     splits on bare spaces, so BPE runs over the whole U+2581-joined text
  ##   - the conversion path byte-decodes vocab keys with the GPT-2 alphabet,
  ##     dropping every U+2581-prefixed entry
  ##   - the converter rejects a Split pre_tokenizer without a regexp
  ##
  ## The suites replay recorded token ids and never encode.
  if self.tokenizer.isNil:
    raise newException(ValueError,
      "[ttt] Gemma3Model: the gemma-3 metaspace tokenizer has no converter " &
      "in toktoktok yet, text tokenization is unavailable for " &
      self.tokenizerPath)
  self.tokenizer

func getDeviceKind(self: Gemma3Model): DeviceKind =
  self.device

proc loadGemma3ModelRaw(modelPath: string, device: DeviceKind): Gemma3Model =
  let config = loadGemma3Config(modelPath / "config.json")
  let weightsPath = modelPath / "model.safetensors"
  let weights = SafetensorsCollection.open(weightsPath)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  checkValue(config.hidden_activation == "gelu_pytorch_tanh",
    "[ttt] Gemma3Config: hidden_activation is \"" & config.hidden_activation &
    "\", the port implements the gelu_pytorch_tanh checkpoint family only")

  let embedTokens = Embedding.load(weights, cfgJson, "model.embed_tokens", device)
  let embedScale = F.full(1, sqrt(config.hidden_size.float64),
    F.tensorOptions(F.kBFloat16, device))

  let softmaxScale = 1.0 / sqrt(config.query_pre_attn_scalar.float64)

  # Dual-theta tables. Sliding_attention layers rope with the local base
  # frequency and full_attention layers with the global rope_theta.
  let rotaryFull = RotaryPositionEmbedding.new(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta,
    actDtype,
    device
  )
  let rotarySliding = RotaryPositionEmbedding.new(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_local_base_freq,
    actDtype,
    device
  )

  var layers = newSeq[Gemma3DecoderLayer](config.num_hidden_layers)

  for i in 0 ..< config.num_hidden_layers:

    let lp = "model.layers." & $i & "."
    let sliding = config.layerKinds[i] == alkSlidingAttention

    let attn_norm = RmsNormOne.load(weights, cfgJson, lp & "input_layernorm", device)
    let mlp_norm = RmsNormOne.load(weights, cfgJson, lp & "post_attention_layernorm", device)
    let attn = RopeGQAttention[RmsNormOne].load(
      weights, cfgJson, lp & "self_attn", i,
      config.num_attention_heads, config.num_key_value_heads, config.head_dim,
      if sliding: rotarySliding else: rotaryFull, device,
      window = if sliding: config.sliding_window else: FullVisibilityWindow,
      softmaxScale = softmaxScale)
    let pre_ffn_norm = RmsNormOne.load(weights, cfgJson, lp & "pre_feedforward_layernorm", device)
    let mlp = GatedDenseFFN.load(weights, cfgJson, lp & "mlp", device,
      activation = kGeluTanh)
    let post_ffn_norm = RmsNormOne.load(weights, cfgJson, lp & "post_feedforward_layernorm", device)
    layers[i] = Gemma3DecoderLayer.init(attn_norm, attn, mlp_norm,
      pre_ffn_norm, mlp, post_ffn_norm)

  let norm = RmsNormOne.load(weights, cfgJson, "model.norm", device)
  let lmHead = LMHead.load(weights, cfgJson, embedTokens, device)
  result = Gemma3Model(
    embedTokens: embedTokens,
    embedScale: embedScale,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    config: config,
    rotaryFull: rotaryFull,
    rotarySliding: rotarySliding,
    tokenizerPath: modelPath / "tokenizer.json",
    device: device
  )

proc loadGemma3Model*(modelPath: string, device: DeviceKind): AnyModel =
  ## Loads a gemma-3 checkpoint from `modelPath` onto `device`, wrapped
  ## in the AnyModel model interface.
  let gemma3Model = loadGemma3ModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  gemma3Model.to(AnyModel)

static:
  # Register Gemma3 model in the registry
  ModelRegistry["Gemma3ForCausalLM"] = loadGemma3Model
