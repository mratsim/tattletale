# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  std/os,
  std/memfiles,
  std/tables,

  pkg/iface,
  pkg/packedjson,

  workspace/libtorch,
  workspace/safetensors,
  workspace/positron,

  # Transformers local imports
  ../layers,
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

proc numKvGroups(cfg: Qwen3Config): int =
  cfg.num_attention_heads div cfg.num_key_value_heads

################################################################################
#                          Qwen3 Model                                         #
################################################################################

type
  Qwen3Model* = ref object
    embedTokens: Embedding
    layers: seq[TransformerBlock]
    norm: RmsNorm
    lmHead: LMHead
    config*: Qwen3Config
    rotary*: RotaryPositionEmbeddingRef

proc forward*(self: Qwen3Model, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Forward pass for Qwen3 model.
  ##
  ## Args:
  ## Forward pass for Qwen3 model.
  ##
  ## Args:
  ##   ctx: InferenceContext with KV caches and position_ids
  ##   input_ids: Input token IDs of shape (batch, seq_len)
  ##
  ## Returns:
  ##   Logits of shape (batch, seq_len, vocab_size)
  ##
  ## Computes:
  ##   x = self.embedTokens(input_ids)
  ##   ctx.setRopeForPositions(self.rotary)
  ##   for layer in self.layers:
  ##     (x, residual) = layer(ctx, x, residual)
  ##   x = self.norm(x + residual)
  ##   return self.lmHead(x)

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

proc loadQwen3ModelRaw(modelPath: string, device = kCPU): Qwen3Model =
  ## Load Qwen3 model and return as concrete Qwen3Model type (not interface).
  ## Use this when you need to access internal fields for instrumentation/testing.
  let config = loadQwen3Config(modelPath / "config.json")

  let weightsPath = modelPath / "model.safetensors"
  var weightsMemFile = memFiles.open(weightsPath, mode = fmRead)
  defer: close(weightsMemFile)

  var weightsSt = safetensors.load(weightsMemFile)

  let embedWeight = weightsSt.getTensorOwned("model.embed_tokens.weight")
  let embedTokens = Embedding.init(embedWeight)

  var layers = newSeq[TransformerBlock](config.num_hidden_layers)

  # Create ONE RoPE instance for entire model (shared across all layers)
  let rotary = RotaryPositionEmbeddingRef.new(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta,
    kBFloat16,
    device
  )

  for i in 0..<config.num_hidden_layers:
    let layerPrefix = "model.layers." & $i & "."
    let inputLnWeight = weightsSt.getTensorOwned(layerPrefix & "input_layernorm.weight")
    let postAttnWeight = weightsSt.getTensorOwned(layerPrefix & "post_attention_layernorm.weight")
    let qWeight = weightsSt.getTensorOwned(layerPrefix & "self_attn.q_proj.weight")
    let kWeight = weightsSt.getTensorOwned(layerPrefix & "self_attn.k_proj.weight")
    let vWeight = weightsSt.getTensorOwned(layerPrefix & "self_attn.v_proj.weight")
    let oWeight = weightsSt.getTensorOwned(layerPrefix & "self_attn.o_proj.weight")
    let qNormWeight = weightsSt.getTensorOwned(layerPrefix & "self_attn.q_norm.weight")
    let kNormWeight = weightsSt.getTensorOwned(layerPrefix & "self_attn.k_norm.weight")
    let gateWeight = weightsSt.getTensorOwned(layerPrefix & "mlp.gate_proj.weight")
    let upWeight = weightsSt.getTensorOwned(layerPrefix & "mlp.up_proj.weight")
    let downWeight = weightsSt.getTensorOwned(layerPrefix & "mlp.down_proj.weight")

    let attn_norm = RmsNorm.init(inputLnWeight)
    let attn = RopeGQAttention.init(
      i,  # layer_idx
      layerPrefix & "self_attn",  # name
      qWeight, kWeight, vWeight, oWeight,
      qNormWeight, kNormWeight,
      config.num_attention_heads, config.num_key_value_heads, config.head_dim,
      rotary,  # ← SHARED across all layers
      rms_norm_eps = config.rms_norm_eps
    )
    let mlp_norm = RmsNorm.init(postAttnWeight)
    let mlp = GatedMLP.init(gateWeight, upWeight, downWeight, kSilu)

    layers[i] = TransformerBlock.init(i, attn_norm, attn, mlp_norm, mlp)

  let finalNormWeight = weightsSt.getTensorOwned("model.norm.weight")
  let norm = RmsNorm.init(finalNormWeight)
  let lmHead = LMHead.initTied(embedTokens)

  result = Qwen3Model(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    config: config,
    rotary: rotary
  )

proc loadQwen3Model*(modelPath: string, device = kCPU): Model =
  let qwen3Model = loadQwen3ModelRaw(modelPath, device)
  # iface generates to[Model] converter automatically
  qwen3Model.to(Model)

static:
  # Register Qwen3 model in the registry
  ModelRegistry["Qwen3ForCausalLM"] = loadQwen3Model
