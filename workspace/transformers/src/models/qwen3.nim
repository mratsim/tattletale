# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/json,
  std/options,
  std/os,
  std/memfiles,
  std/tables,

  pkg/iface,
  workspace/libtorch,
  workspace/safetensors,
  workspace/containers,
  workspace/positron,

  # Transformers local imports
  ../layers/all,
  ./all_interfaces

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
    layers: Vec[TransformerBlock]
    norm: RmsNorm
    lmHead: LMHead
    config*: Qwen3Config

proc forward*(self: Qwen3Model, input: TorchTensor, positions: TorchTensor, cache: var KVCache): TorchTensor =
  var x = self.embedTokens.forward(input)
  var residual: Option[TorchTensor]
  for layer in mitems(self.layers):
    layer.attn.resetCache()
    let layerOut = layer.forward(x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])
  let finalResidual = residual.get(x)
  let normed = self.norm.forward(x + finalResidual)
  self.lmHead.forward(normed)

# iface generates to[Model] converter automatically

proc loadQwen3Model(modelPath: string, device = kCPU): Model =
  let config = loadQwen3Config(modelPath)

  let weightsPath = modelPath / "model.safetensors"
  var weightsMemFile = memFiles.open(weightsPath, mode = fmRead)
  defer: close(weightsMemFile)

  var weightsSt = safetensors.load(weightsMemFile)

  let embedWeight = weightsSt.getTensorOwned("model.embed_tokens.weight")
  let embedTokens = Embedding.init(embedWeight)

  var layers = Vec[TransformerBlock].new(config.num_hidden_layers)

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

    var rotary = RotaryPositionEmbedding.init(
      config.head_dim,
      config.max_position_embeddings,
      config.rope_theta,
      kBFloat16,
      device
    )

    let attn_norm = RmsNorm.init(inputLnWeight)
    var attn = RopeGQAttention.init(
      qWeight, kWeight, vWeight, oWeight,
      qNormWeight, kNormWeight,
      config.num_attention_heads, config.num_key_value_heads, config.head_dim,
      rotary,
      rms_norm_eps = config.rms_norm_eps
    )
    let mlp_norm = RmsNorm.init(postAttnWeight)
    let mlp = GatedMLP.init(gateWeight, upWeight, downWeight, kSilu)

    layers[i] = TransformerBlock.init(attn_norm, attn, mlp_norm, mlp)

  let finalNormWeight = weightsSt.getTensorOwned("model.norm.weight")
  let norm = RmsNorm.init(finalNormWeight)
  let lmHead = LMHead.initTied(embedTokens)

  let qwen3Model = Qwen3Model(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    config: config
  )

  qwen3Model.to(Model)

static:
  # Register Qwen3 model in the registry
  ModelRegistry["Qwen3ForCausalLM"] = loadQwen3Model