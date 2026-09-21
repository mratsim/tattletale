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

# ─── Mistral configuration ─────────────────────────────────────────────────

type
  MistralConfig* = ref object
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
    hidden_act: string
    max_position_embeddings: int
    rope_theta: float64
      ## Single rope base, the all-sliding shape ships one theta.
    sliding_window: int
      ## Visibility band of every layer, the all-sliding shape.
    bos_token_id: int
    eos_token_id: int
      ## Stop token of the checkpoint config.

func parseMistralConfig(json: JsonNode): MistralConfig =
  result = new MistralConfig

  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = json{"model_type"}.getStr()
  result.vocab_size = json{"vocab_size"}.getInt().int
  result.hidden_size = json{"hidden_size"}.getInt().int
  result.num_hidden_layers = json{"num_hidden_layers"}.getInt().int
  result.rms_norm_eps = json{"rms_norm_eps"}.getFloat()
  result.torch_dtype = json{"torch_dtype"}.getStr("bfloat16")
  result.num_attention_heads = json{"num_attention_heads"}.getInt().int
  result.num_key_value_heads = json{"num_key_value_heads"}.getInt().int
  result.head_dim = json{"head_dim"}.getInt(json{"hidden_size"}.getInt().int div
    json{"num_attention_heads"}.getInt().int)
  result.intermediate_size = json{"intermediate_size"}.getInt().int
  result.hidden_act = json{"hidden_act"}.getStr()
  result.max_position_embeddings = json{"max_position_embeddings"}.getInt().int
  result.rope_theta = json{"rope_theta"}.getFloat()
  result.sliding_window = json{"sliding_window"}.getInt().int
  result.bos_token_id = json{"bos_token_id"}.getInt().int
  result.eos_token_id = json{"eos_token_id"}.getInt().int

proc loadMistralConfig(path: string): MistralConfig =
  let json = path.parseFile()
  result = parseMistralConfig(json)

# ─── Mistral model ─────────────────────────────────────────────────────────

type
  MistralDecoderLayer* = DecoderLayer[RopeGQAttention[void], GatedDenseFFN, RmsNorm]

  MistralModel* = ref object
    embedTokens: Embedding
    layers: seq[MistralDecoderLayer]
    norm: RmsNorm
    lmHead: LMHead
    config: MistralConfig
    layerKinds: seq[AttentionLayerKind]
      ## Per-layer attention kinds, all sliding_attention on this shape.
    rotary: RotaryPositionEmbedding
      ## Single rope table, one theta over every layer.
    tokenizerPath: string
      ## Deferred tokenizer binding, the Mistral metaspace tokenizer is
      ## loaded on first `getTokenizer` use, never at model load.
    tokenizer: BPETokenizer
      ## Nil until the first successful `getTokenizer` call.
    device: DeviceKind

proc forward*(self: MistralModel, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Full forward to logits over the input token ids.
  ##
  ## Expected input:
  ##
  ## - `input_ids`, token ids of shape (batch, seq_len)
  ##
  ## Output:
  ##
  ## - logits of shape (batch, seq_len, vocab_size)

  var x = self.embedTokens(input_ids)

  # Single-theta rope over every layer of the all-sliding shape, the rows
  # travel as per-forward request state on the context.
  ctx.setRopeForPositions(self.rotary)

  var residual: Option[Tensor]
  for layer in self.layers:
    let layerOut = layer.forward(ctx, x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])

  let finalResidual = residual.get(x)
  let normed = self.norm(x + finalResidual)
  result = self.lmHead(normed)

func getConfig(self: MistralModel): ModelConfigBase =
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
    layerKinds: self.layerKinds
  )

func getTokenizer(self: MistralModel): BPETokenizer =
  ## Checkpoint tokenizer, loaded on first use.
  ##
  ## Raises:
  ##   - `ValueError` on every call, the Mistral metaspace tokenizer has
  ##     no converter yet and text tokenization is unavailable
  ##
  ## SentencePiece metaspace family, the Mistral-7B pre_tokenizer
  ## maps spaces to `U+2581` with `prepend_scheme: first`, a shape
  ## the HF-to-tiktoken conversion path drops and this port defers.
  ##
  ## Deferred binding, the same pattern as the gemma-3 lineage.
  ##
  ## The suites replay recorded token ids and never encode.
  if self.tokenizer.isNil:
    raise newException(ValueError,
      "[ttt] MistralModel: the Mistral metaspace tokenizer has no converter " &
      "in toktoktok yet, text tokenization is unavailable for " &
      self.tokenizerPath)
  self.tokenizer

func getDeviceKind(self: MistralModel): DeviceKind =
  self.device

proc loadMistralModelRaw(modelPath: string, device: DeviceKind): MistralModel =
  let config = loadMistralConfig(modelPath / "config.json")
  let weights = SafetensorsCollection.open(modelPath)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  checkValue(config.hidden_act == "silu",
    "[ttt] MistralConfig: hidden_act is \"" & config.hidden_act &
    "\", the port implements the silu checkpoint family only")

  # All-sliding shape, one global sliding_window and no layer_types row.
  # Every decoder layer windows, no full layer exists.
  var layerKinds = newSeq[AttentionLayerKind](config.num_hidden_layers)
  for i in 0 ..< config.num_hidden_layers:
    layerKinds[i] = alkSlidingAttention

  let embedTokens = Embedding.load(weights, cfgJson, "model.embed_tokens", device)

  # Single-theta table, the one rope base of the all-sliding checkpoint.
  let rotary = RotaryPositionEmbedding.new(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta,
    actDtype,
    device
  )

  var layers = newSeq[MistralDecoderLayer](config.num_hidden_layers)

  for i in 0 ..< config.num_hidden_layers:

    let lp = "model.layers." & $i & "."

    let attn_norm = RmsNorm.load(weights, cfgJson, lp & "input_layernorm", device)
    let mlp_norm = RmsNorm.load(weights, cfgJson, lp & "post_attention_layernorm", device)
    let attn = RopeGQAttention[void].load(
      weights, cfgJson, lp & "self_attn", i,
      config.num_attention_heads, config.num_key_value_heads, config.head_dim,
      rotary, device,
      window = config.sliding_window)
    let mlp = GatedDenseFFN.load(weights, cfgJson, lp & "mlp", device)
    layers[i] = MistralDecoderLayer.init(attn_norm, attn, mlp_norm, mlp)

  let norm = RmsNorm.load(weights, cfgJson, "model.norm", device)
  let lmHead = LMHead.load(weights, cfgJson, embedTokens, device)
  result = MistralModel(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    config: config,
    layerKinds: layerKinds,
    rotary: rotary,
    tokenizerPath: modelPath / "tokenizer.json",
    device: device
  )

proc loadMistralModel*(modelPath: string, device: DeviceKind): AnyModel =
  ## Loads a Mistral-7B checkpoint from `modelPath` onto `device`, wrapped
  ## in the AnyModel model interface.
  let mistralModel = loadMistralModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  mistralModel.to(AnyModel)

static:
  # Register Mistral model in the registry
  ModelRegistry["MistralForCausalLM"] = loadMistralModel
