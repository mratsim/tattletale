# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  std/os,
  std/tables,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch,
  workspace/safetensors,
  workspace/positron,
  workspace/toktoktok,
  ../layers,
  ../deserialization,
  ../quantizations/datatypes,
  workspace/safetensors/src/collections,
  ../stateful/inference_context,
  ./all_interfaces

{.experimental: "callOperator".}

################################################################################
#                          Qwen3 Configuration                                 #
################################################################################

type
  Qwen3Config* = ref object
    architecture*: string
    model_type*: string
    num_hidden_layers*: int
    hidden_size*: int
    vocab_size*: int
    rms_norm_eps*: float
    torch_dtype*: string
    num_attention_heads*: int
    num_key_value_heads*: int
    head_dim*: int
    intermediate_size*: int
    hidden_act*: string
    max_position_embeddings*: int
    rope_theta*: float64
    rope_scaling*: JsonNode
    partial_rotary_factor*: float64
    use_qk_norm*: bool
    attention_bias*: bool
    attention_dropout*: float64
    use_cache*: bool
    tie_word_embeddings*: bool
    bos_token_id*: int
    eos_token_id*: int
    sliding_window*: Option[int]
    use_sliding_window*: bool
    max_window_layers*: int

proc parseQwen3Config(json: JsonNode): Qwen3Config =
  result = new Qwen3Config

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
  result.hidden_act = json{"hidden_act"}.getStr()
  result.max_position_embeddings = json{"max_position_embeddings"}.getInt().int
  result.rope_theta = json{"rope_theta"}.getFloat()
  result.rope_scaling = json{"rope_scaling"}
  result.partial_rotary_factor = json{"partial_rotary_factor"}.getFloat(1.0)
  result.use_qk_norm = json{"use_qk_norm"}.getBool(false)
  result.attention_bias = json{"attention_bias"}.getBool(false)
  result.attention_dropout = json{"attention_dropout"}.getFloat(0.0)
  result.use_cache = json{"use_cache"}.getBool(true)
  result.tie_word_embeddings = json{"tie_word_embeddings"}.getBool(true)
  result.bos_token_id = json{"bos_token_id"}.getInt().int
  result.eos_token_id = json{"eos_token_id"}.getInt().int
  result.sliding_window = if json{"sliding_window"}.kind == JNull:
    none(int)
  else:
    some(json{"sliding_window"}.getInt().int)
  result.use_sliding_window = json{"use_sliding_window"}.getBool(false)
  result.max_window_layers = json{"max_window_layers"}.getInt().int

proc loadQwen3Config(path: string): Qwen3Config =
  let json = path.parseFile()
  result = parseQwen3Config(json)

################################################################################
#                          Qwen3 Model                                         #
################################################################################

type
  Qwen3DecoderLayer = DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm]

  Qwen3Model* = ref object
    embedTokens: Embedding
    layers: seq[Qwen3DecoderLayer]
    norm: RmsNorm
    lmHead: LMHead
    config*: Qwen3Config
    rotary*: RotaryPositionEmbedding
    tokenizer*: BPETokenizer
    device*: DeviceKind

proc forward*(self: Qwen3Model, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Input: (batch, seq_len) token ids.
  ## Output: logits of (batch, seq_len, vocab_size).

  var x = self.embedTokens(input_ids)

  # Populate ctx.cos/sin from model's RoPE cache
  ctx.setRopeForPositions(self.rotary)

  var residual: Option[Tensor]
  for layer in mitems(self.layers):
    let layerOut = layer(ctx, x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])

  let finalResidual = residual.get(x)
  let normed = self.norm(x + finalResidual)
  result = self.lmHead(normed)

proc getConfig(self: Qwen3Model): ModelConfigBase =
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
    eosTokenId: self.config.eos_token_id
  )

proc getTokenizer(self: Qwen3Model): BPETokenizer =
  self.tokenizer

proc getDeviceKind(self: Qwen3Model): DeviceKind =
  self.device

proc loadQwen3ModelRaw(modelPath: string, device: DeviceKind): Qwen3Model =
  ## Load Qwen3 model — no quantization knowledge, all dispatched via
  ## deserialization.nim and QuantLoaderRegistry.
  let config = loadQwen3Config(modelPath / "config.json")
  let weightsPath = modelPath / "model.safetensors"
  let weights = SafetensorsCollection.open(weightsPath)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  let embedTokens = Embedding.load(weights, cfgJson, "model.embed_tokens", device)

  var layers = newSeq[Qwen3DecoderLayer](config.num_hidden_layers)

  let rotary = RotaryPositionEmbedding.new(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta,
    actDtype,
    device
  )

  for i in 0..<config.num_hidden_layers:

    let lp = "model.layers." & $i & "."

    let attn_norm = RmsNorm.load(weights, cfgJson, lp & "input_layernorm", device)
    let mlp_norm = RmsNorm.load(weights, cfgJson, lp & "post_attention_layernorm", device)
    let attn = RopeGQAttention[RmsNorm].load(
      weights, cfgJson, lp & "self_attn", i,
      config.num_attention_heads, config.num_key_value_heads, config.head_dim,
      rotary, device)
    let mlp = GatedDenseFFN.load(weights, cfgJson, lp & "mlp", device)
    layers[i] = Qwen3DecoderLayer.init(attn_norm, attn, mlp_norm, mlp)

  let norm = RmsNorm.load(weights, cfgJson, "model.norm", device)
  let lmHead = LMHead.load(weights, cfgJson, embedTokens, device)
  let tokenizerPath = modelPath / "tokenizer.json"
  let tokenizer = loadHFTokenizer(tokenizerPath)
  result = Qwen3Model(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    config: config,
    rotary: rotary,
    tokenizer: tokenizer,
    device: device
  )

proc loadQwen3Model*(modelPath: string, device: DeviceKind): AnyModel =
  let qwen3Model = loadQwen3ModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  qwen3Model.to(AnyModel)

static:
  # Register Qwen3 model in the registry
  ModelRegistry["Qwen3ForCausalLM"] = loadQwen3Model
