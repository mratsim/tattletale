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
  workspace/safetensors,
  workspace/toktoktok,
  ../layers,
  ../deserialization,
  workspace/safetensors/src/collections,
  ../stateful/inference_context,
  ../models/loading/layer_kinds,
  ./all_interfaces

{.experimental: "callOperator".}

################################################################################
#                          Qwen3.5 Configuration                               #
################################################################################

type
  Qwen35Config* = ref object
    architecture*: string
    model_type*: string
    vocab_size*: int
    hidden_size*: int
    num_hidden_layers*: int
    num_attention_heads*: int
    num_key_value_heads*: int
    head_dim*: int
    intermediate_size*: int
    rms_norm_eps*: float
    hidden_act*: string
    max_position_embeddings*: int
    dtype*: string          ## Activation dtype (config uses `dtype`, there is no `torch_dtype` key)
    rope_theta*: float64
    partial_rotary_factor*: float64
    mrope_interleaved*: bool
    mrope_section*: seq[int]
    rope_type*: string
    attn_output_gate*: bool  ## Full-attention q_proj emits q and a raw per-head scaling vector
    full_attention_interval*: int
    layer_types*: seq[AttentionLayerKind]
      ## Per-layer kind, parsed from the `layer_types` HF key:
      ## alkGatedDeltaNet (Gated DeltaNet) or alkAttention
    linear_conv_kernel_dim*: int
    linear_key_head_dim*: int
    linear_num_key_heads*: int
    linear_num_value_heads*: int
    linear_value_head_dim*: int
    mamba_ssm_dtype*: string
    attention_bias*: bool
    attention_dropout*: float64
    use_cache*: bool
    tie_word_embeddings*: bool
    mlp_only_layers*: seq[string]
    mtp_num_hidden_layers*: int
    mtp_use_dedicated_embeddings*: bool
    bos_token_id*: Option[int]
    eos_token_id*: int
    image_token_id*: int
    video_token_id*: int
    transformers_version*: string

proc parseQwen35Config(json: JsonNode): Qwen35Config =
  let textCfg = json{"text_config"}
  let ropeParams = textCfg{"rope_parameters"}

  result = new Qwen35Config
  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = textCfg{"model_type"}.getStr()
  result.vocab_size = textCfg{"vocab_size"}.getInt().int
  result.hidden_size = textCfg{"hidden_size"}.getInt().int
  result.num_hidden_layers = textCfg{"num_hidden_layers"}.getInt().int
  result.num_attention_heads = textCfg{"num_attention_heads"}.getInt().int
  result.num_key_value_heads = textCfg{"num_key_value_heads"}.getInt().int
  result.head_dim = textCfg{"head_dim"}.getInt().int
  result.intermediate_size = textCfg{"intermediate_size"}.getInt().int
  result.rms_norm_eps = textCfg{"rms_norm_eps"}.getFloat()
  result.hidden_act = textCfg{"hidden_act"}.getStr()
  result.max_position_embeddings = textCfg{"max_position_embeddings"}.getInt().int
  result.dtype = textCfg{"dtype"}.getStr("bfloat16")
  result.rope_theta = ropeParams{"rope_theta"}.getFloat(1e6)
  result.partial_rotary_factor = ropeParams{"partial_rotary_factor"}.getFloat(1.0)
  result.mrope_interleaved = ropeParams{"mrope_interleaved"}.getBool(false)
  result.rope_type = ropeParams{"rope_type"}.getStr("default")
  result.mrope_section = newSeq[int]()
  if ropeParams{"mrope_section"}.kind != JNull:
    for i in 0 ..< ropeParams{"mrope_section"}.len:
      result.mrope_section.add(ropeParams{"mrope_section"}[i].getInt().int)
  result.attn_output_gate = textCfg{"attn_output_gate"}.getBool(false)
  result.full_attention_interval = textCfg{"full_attention_interval"}.getInt(4)
  result.layer_types = newSeq[AttentionLayerKind]()
  let rawKinds = textCfg{"layer_types"}
  if rawKinds.kind == JArray:
    for i in 0 ..< rawKinds.len:
      let elem = rawKinds[i]
      if elem.kind != JString:
        raise newException(ValueError,
          "[ttt] text_config.layer_types[" & $i & "]: expected a string, found " & $elem.kind)
      result.layer_types.add parseAttnFromHfTransformers(
        elem.getStr(), "text_config.layer_types[" & $i & "]")
  result.linear_conv_kernel_dim = textCfg{"linear_conv_kernel_dim"}.getInt().int
  result.linear_key_head_dim = textCfg{"linear_key_head_dim"}.getInt().int
  result.linear_num_key_heads = textCfg{"linear_num_key_heads"}.getInt().int
  result.linear_num_value_heads = textCfg{"linear_num_value_heads"}.getInt().int
  result.linear_value_head_dim = textCfg{"linear_value_head_dim"}.getInt().int
  result.mamba_ssm_dtype = textCfg{"mamba_ssm_dtype"}.getStr("float32")
  result.attention_bias = textCfg{"attention_bias"}.getBool(false)
  result.attention_dropout = textCfg{"attention_dropout"}.getFloat(0.0)
  result.use_cache = textCfg{"use_cache"}.getBool(true)
  result.tie_word_embeddings = textCfg{"tie_word_embeddings"}.getBool(true)
  result.mlp_only_layers = newSeq[string]()
  if textCfg{"mlp_only_layers"}.kind != JNull:
    for i in 0 ..< textCfg{"mlp_only_layers"}.len:
      result.mlp_only_layers.add(textCfg{"mlp_only_layers"}[i].getStr())
  result.mtp_num_hidden_layers = textCfg{"mtp_num_hidden_layers"}.getInt().int
  result.mtp_use_dedicated_embeddings = textCfg{"mtp_use_dedicated_embeddings"}.getBool(false)
  result.bos_token_id = if textCfg{"bos_token_id"}.kind == JNull:
    none(int)
  else:
    some(textCfg{"bos_token_id"}.getInt().int)
  result.eos_token_id = textCfg{"eos_token_id"}.getInt().int
  result.image_token_id = json{"image_token_id"}.getInt().int
  result.video_token_id = json{"video_token_id"}.getInt().int
  result.transformers_version = json{"transformers_version"}.getStr("")

proc loadQwen35Config(path: string): Qwen35Config =
  let json = path.parseFile()
  result = parseQwen35Config(json)

################################################################################
#                          Qwen3.5 Model                                      #
################################################################################

type
  Qwen35GdnDecoderLayer = DecoderLayer[GatedDeltaNet[perHead, FullRankGateIn, GateForm.softplus], GatedDenseFFN, RmsNormOne]

  Qwen35AttnDecoderLayer = DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedDenseFFN, RmsNormOne]

  Qwen35Model* = ref object
    embedTokens: Embedding
    layers: seq[AnyDecoderLayer]
    norm: RmsNormOne
    lmHead: LMHead
    rotary: RotaryPositionEmbedding
    config*: Qwen35Config
    tokenizer*: BPETokenizer
    device*: DeviceKind

proc forward*(self: Qwen35Model, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  var x = self.embedTokens(input_ids)

  ctx.setRopeForPositions(self.rotary)

  var residual: Option[Tensor]
  for layer in self.layers:
    let layerOut = layer(ctx, x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])

  let finalResidual = residual.get(x)
  let normed = self.norm(x + finalResidual)
  result = self.lmHead(normed)

proc getConfig(self: Qwen35Model): ModelConfigBase =
  ## Minimal config for InferenceContext creation in `generate()`.
  ModelConfigBase(
    architecture: self.config.architecture,
    model_type: self.config.model_type,
    num_hidden_layers: self.config.num_hidden_layers,
    hidden_size: self.config.hidden_size,
    vocab_size: self.config.vocab_size,
    rms_norm_eps: self.config.rms_norm_eps,
    torch_dtype: self.config.dtype,
    num_attention_heads: self.config.num_attention_heads,
    num_key_value_heads: self.config.num_key_value_heads,
    head_dim: self.config.head_dim,
    intermediate_size: self.config.intermediate_size,
    max_position_embeddings: self.config.max_position_embeddings,
    eosTokenId: self.config.eos_token_id
  )

proc getTokenizer(self: Qwen35Model): BPETokenizer =
  self.tokenizer

proc getDeviceKind(self: Qwen35Model): DeviceKind =
  self.device

proc loadQwen35ModelRaw(modelPath: string, device = kCPU): Qwen35Model =
  ## The checkpoint file also carries foreign tensors: `model.visual.*`
  ## (vision tower) and `mtp.*` (draft head). This model never requests
  ## them, so the load skips them without error.
  let config = loadQwen35Config(modelPath / "config.json")
  let weightsPath = modelPath / "model.safetensors-00001-of-00001.safetensors"
  let weights = SafetensorsCollection.open(weightsPath)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  let embedTokens = Embedding.load(weights, cfgJson, "model.language_model.embed_tokens", device)

  let rotary = RotaryPositionEmbedding.new(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta,
    actDtype,
    device,
    rotary_dim = int(config.head_dim.float64 * config.partial_rotary_factor))

  var layers = newSeq[AnyDecoderLayer](config.num_hidden_layers)
  if config.layer_types.len != config.num_hidden_layers:
    raise newException(ValueError,
      "layer_types has " & $config.layer_types.len &
      " entries, expected " & $config.num_hidden_layers)
  for i in 0 ..< config.num_hidden_layers:
    let lp = "model.language_model.layers." & $i & "."

    let inputLN = RmsNormOne.load(weights, cfgJson, lp & "input_layernorm", device)
    let postLN = RmsNormOne.load(weights, cfgJson, lp & "post_attention_layernorm", device)

    let mlp = GatedDenseFFN.load(weights, cfgJson, lp & "mlp", device)

    var gdn: GatedDeltaNet[perHead, FullRankGateIn, GateForm.softplus] = nil
    var attn: RopeElementWiseGatedAttention[RmsNormOne] = nil
    if config.layer_types[i] == alkGatedDeltaNet:
      let gdnPrefix = lp & "linear_attn"
      gdn = GatedDeltaNet[perHead, FullRankGateIn, GateForm.softplus].load(
        weights, cfgJson, gdnPrefix, i,
        config.linear_num_key_heads, config.linear_num_value_heads,
        config.linear_key_head_dim, config.linear_value_head_dim,
        config.linear_conv_kernel_dim, device)
    else:
      attn = RopeElementWiseGatedAttention[RmsNormOne].load(
        weights, cfgJson, lp & "self_attn", i,
        config.num_attention_heads, config.num_key_value_heads, config.head_dim,
        rotary, device)

    layers[i] =
      if config.layer_types[i] == alkGatedDeltaNet:
        Qwen35GdnDecoderLayer.init(
          input_layernorm = inputLN, sequence_mixer = gdn,
          post_attention_layernorm = postLN,
          hidden_mixer = mlp).to(AnyDecoderLayer)
      else:
        Qwen35AttnDecoderLayer.init(
          input_layernorm = inputLN, sequence_mixer = attn,
          post_attention_layernorm = postLN,
          hidden_mixer = mlp).to(AnyDecoderLayer)

  let norm = RmsNormOne.load(weights, cfgJson, "model.language_model.norm", device)

  let lmHead = LMHead.load(weights, cfgJson, embedTokens, device)

  let tokenizerPath = modelPath / "tokenizer.json"
  let tokenizer = loadHFTokenizer(tokenizerPath)
  result = Qwen35Model(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    rotary: rotary,
    config: config,
    tokenizer: tokenizer,
    device: device,
  )

proc loadQwen35Model*(modelPath: string, device = kCPU): AnyModel =
  let qwen35Model = loadQwen35ModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  qwen35Model.to(AnyModel)

static:
  # Register Qwen3.5 model in the registry
  ModelRegistry["Qwen3_5ForConditionalGeneration"] = loadQwen35Model
