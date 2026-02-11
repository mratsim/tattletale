# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  std/math,
  std/json,

  workspace/libtorch,

  # Transformers local imports
  ./all,
  ../layers/rotary_emb,
  ../layers/rms_norm

################################################################################
#                                                                              #
#                          Qwen3 Configuration                                 #
#                                                                              #
################################################################################

type Qwen3Config* = ref object of ModelConfig
  vocab_size*: int
  hidden_size*: int
  num_hidden_layers*: int
  num_attention_heads*: int
  num_key_value_heads*: int
  head_dim*: int
  intermediate_size*: int
  hidden_act*: string
  max_position_embeddings*: int
  rope_theta*: float64
  rope_scaling*: JsonNode
  partial_rotary_factor*: float64
  rms_norm_eps*: float64
  use_qk_norm*: bool
  attention_bias*: bool
  attention_dropout*: float64
  use_cache*: bool
  tie_word_embeddings*: bool
  bos_token_id*: int
  eos_token_id*: int
  torch_dtype*: string
  sliding_window*: Option[int]
  use_sliding_window*: bool
  max_window_layers*: int

proc parseQwen3Config*(json: JsonNode): Qwen3Config =
  result = new Qwen3Config

  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = json{"model_type"}.getStr()

  result.vocab_size = json{"vocab_size"}.getInt().int
  result.hidden_size = json{"hidden_size"}.getInt().int
  result.num_hidden_layers = json{"num_hidden_layers"}.getInt().int
  result.num_attention_heads = json{"num_attention_heads"}.getInt().int
  result.num_key_value_heads = json{"num_key_value_heads"}.getInt().int
  result.head_dim = json{"head_dim"}.getInt().int

  result.intermediate_size = json{"intermediate_size"}.getInt().int
  result.hidden_act = json{"hidden_act"}.getStr()

  result.max_position_embeddings = json{"max_position_embeddings"}.getInt().int
  result.rope_theta = json{"rope_theta"}.getFloat()
  result.rope_scaling = json{"rope_scaling"}
  result.partial_rotary_factor = json{"partial_rotary_factor"}.getFloat(1.0)

  result.rms_norm_eps = json{"rms_norm_eps"}.getFloat()
  result.use_qk_norm = json{"use_qk_norm"}.getBool(false)
  result.attention_bias = json{"attention_bias"}.getBool(false)
  result.attention_dropout = json{"attention_dropout"}.getFloat(0.0)

  result.use_cache = json{"use_cache"}.getBool(true)
  result.tie_word_embeddings = json{"tie_word_embeddings"}.getBool(true)

  result.bos_token_id = json{"bos_token_id"}.getInt().int
  result.eos_token_id = json{"eos_token_id"}.getInt().int
  result.torch_dtype = json{"torch_dtype"}.getStr("bfloat16")

  result.sliding_window = if json{"sliding_window"}.kind == JNull:
    none(int)
  else:
    some(json{"sliding_window"}.getInt().int)
  result.use_sliding_window = json{"use_sliding_window"}.getBool(false)
  result.max_window_layers = json{"max_window_layers"}.getInt().int

proc loadQwen3Config*(path: string): Qwen3Config =
  let json = path.parseFile()
  result = parseQwen3Config(json)

proc numKvGroups*(cfg: Qwen3Config): int =
  cfg.num_attention_heads div cfg.num_key_value_heads

################################################################################
#                                                                              #
#                          Qwen3 Implementation                                #
#                                                                              #
################################################################################

type Qwen3MLP* = ref object
  gate_proj*: TorchTensor
  up_proj*: TorchTensor
  down_proj*: TorchTensor

proc newQwen3MLP*(cfg: Qwen3Config, vb: VarBuilder): Qwen3MLP =
  new result

  result.gate_proj = vb.pp("gate_proj").get_tensor("weight")
  result.up_proj = vb.pp("up_proj").get_tensor("weight")
  result.down_proj = vb.pp("down_proj").get_tensor("weight")

proc forward*(self: Qwen3MLP, x: TorchTensor): TorchTensor =
  let gate = linear(x, self.gate_proj)
  let up = linear(x, self.up_proj)
  let act_gate = silu(gate)
  let down = linear(act_gate * up, self.down_proj)
  down

type ConcatKvCache* = ref object
  k_cache*: seq[TorchTensor]
  v_cache*: seq[TorchTensor]
  concat_dim*: int

proc newConcatKvCache*(concat_dim: int = 2): ConcatKvCache =
  new result
  result.concat_dim = concat_dim
  result.k_cache = @[]
  result.v_cache = @[]

proc append*(self: ConcatKvCache, k, v: TorchTensor): (TorchTensor, TorchTensor) =
  self.k_cache.add(k)
  self.v_cache.add(v)

  let k_cat = cat(self.k_cache, self.concat_dim)
  let v_cat = cat(self.v_cache, self.concat_dim)
  (k_cat, v_cat)

proc reset*(self: ConcatKvCache) =
  self.k_cache = @[]
  self.v_cache = @[]

type Qwen3Attention* = ref object
  q_proj*: TorchTensor
  k_proj*: TorchTensor
  v_proj*: TorchTensor
  o_proj*: TorchTensor
  q_norm*: RmsNorm
  k_norm*: RmsNorm
  num_head*: int
  num_kv_head*: int
  num_kv_group*: int
  head_dim*: int
  hidden_size*: int
  rotary_emb*: RotaryEmb
  kv_cache*: ConcatKvCache

proc newQwen3Attention*(
  cfg: Qwen3Config,
  rotary_emb: RotaryEmb,
  vb: VarBuilder
): Qwen3Attention =
  new result

  let head_dim = cfg.head_dim
  let num_heads = cfg.num_attention_heads
  let num_kv_heads = cfg.num_key_value_heads
  let num_kv_groups = num_heads div num_kv_heads
  let hidden_size = cfg.hidden_size

  result.num_head = num_heads
  result.num_kv_head = num_kv_heads
  result.num_kv_group = num_kv_groups
  result.head_dim = head_dim
  result.hidden_size = hidden_size
  result.rotary_emb = rotary_emb
  result.kv_cache = newConcatKvCache(2)

  result.q_proj = vb.pp("q_proj").get_tensor("weight")
  result.k_proj = vb.pp("k_proj").get_tensor("weight")
  result.v_proj = vb.pp("v_proj").get_tensor("weight")
  result.o_proj = vb.pp("o_proj").get_tensor("weight")

  result.q_norm = newRmsNorm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))
  result.k_norm = newRmsNorm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))

proc repeat_kv*(x: TorchTensor, num_groups: int): TorchTensor =
  let shape = x.sizes()
  if num_groups == 1:
    return x
  let new_shape = @[shape[0], shape[1], num_groups.int64, shape[2] div num_groups.int64, shape[3]]
  x.reshape(new_shape)
    .reshape(@[shape[0], shape[1] * num_groups.int64, shape[2] div num_groups.int64, shape[3]])
    .contiguous()

proc forward*(
  self: Qwen3Attention,
  x: TorchTensor,
  attn_mask: Option[TorchTensor],
  offset: int
): TorchTensor =
  let batch_size = x.size(0)
  let seq_len = x.size(1)

  let q = linear(x, self.q_proj)
  let k = linear(x, self.k_proj)
  let v = linear(x, self.v_proj)

  let q_reshaped = q.reshape(@[batch_size.int64, seq_len.int64, self.num_head.int64, self.head_dim.int64])
    .transpose(1, 2)
  let k_reshaped = k.reshape(@[batch_size.int64, seq_len.int64, self.num_kv_head.int64, self.head_dim.int64])
    .transpose(1, 2)
  let v_reshaped = v.reshape(@[batch_size.int64, seq_len.int64, self.num_kv_head.int64, self.head_dim.int64])
    .transpose(1, 2)

  let q_flat = q_reshaped.reshape(@[(batch_size * self.num_head).int64, seq_len.int64, self.head_dim.int64])
  let k_flat = k_reshaped.reshape(@[(batch_size * self.num_kv_head).int64, seq_len.int64, self.head_dim.int64])

  let q_normed = self.q_norm.forward(q_flat)
  let k_normed = self.k_norm.forward(k_flat)

  let q_normed_reshaped = q_normed.reshape(@[batch_size.int64, self.num_head.int64, seq_len.int64, self.head_dim.int64])
  let k_normed_reshaped = k_normed.reshape(@[batch_size.int64, self.num_kv_head.int64, seq_len.int64, self.head_dim.int64])

  let (q_rot, k_rot) = self.rotary_emb.apply_rope(q_normed_reshaped, k_normed_reshaped, offset)

  let (k_cat, v_cat) = self.kv_cache.append(k_rot, v_reshaped)

  let k_expanded = repeat_kv(k_cat, self.num_kv_group).contiguous()
  let v_expanded = repeat_kv(v_cat, self.num_kv_group).contiguous()

  let scale = 1.0 / sqrt(self.head_dim.float64)
  var scores = (q_rot.matmul(k_expanded.transpose(2, 3)) * scale)

  if attn_mask.isSome:
    scores = scores + attn_mask.get()

  let probs = softmax(scores, -1)
  let context = probs.matmul(v_expanded)

  let context_transposed = context.transpose(1, 2)
    .reshape(@[batch_size.int64, seq_len.int64, self.hidden_size.int64])

  linear(context_transposed, self.o_proj)

proc clear_kv_cache*(self: Qwen3Attention) =
  self.kv_cache.reset()

type DecoderLayer* = ref object
  self_attn*: Qwen3Attention
  mlp*: Qwen3MLP
  ln1*: RmsNorm
  ln2*: RmsNorm

proc newDecoderLayer*(
  cfg: Qwen3Config,
  rotary_emb: RotaryEmb,
  layer_idx: int,
  vb: VarBuilder
): DecoderLayer =
  new result

  result.self_attn = newQwen3Attention(cfg, rotary_emb, vb.pp("self_attn"))
  result.mlp = newQwen3MLP(cfg, vb.pp("mlp"))
  result.ln1 = newRmsNorm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
  result.ln2 = newRmsNorm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))

proc forward*(
  self: DecoderLayer,
  x: TorchTensor,
  mask: Option[TorchTensor],
  offset: int
): TorchTensor =
  let h = self.ln1.forward(x)
  let attn_out = self.self_attn.forward(h, mask, offset)
  let x_residual = x + attn_out

  let h2 = self.ln2.forward(x_residual)
  let mlp_out = self.mlp.forward(h2)
  x_residual + mlp_out

proc clear_kv_cache*(self: DecoderLayer) =
  self.self_attn.clear_kv_cache()

type Qwen3Model* = ref object
  embed_tokens*: TorchTensor
  layers*: seq[DecoderLayer]
  norm*: RmsNorm
  device*: DeviceKind
  dtype*: ScalarKind

const NEG_INFINITY = -1e10

proc newQwen3Model*(cfg: Qwen3Config, vb: VarBuilder): Qwen3Model =
  new result

  result.embed_tokens = vb.pp("model.embed_tokens").get_tensor("weight")
  result.device = kCPU
  result.dtype = kBfloat16

  let rotary_emb = newRotaryEmb(
    cfg.head_dim,
    cfg.max_position_embeddings,
    cfg.rope_theta,
    result.dtype,
    result.device
  )

  result.layers = @[]
  for i in 0..<cfg.num_hidden_layers:
    let layer_vb = vb.pp("model.layers").pp($i)
    let layer = newDecoderLayer(cfg, rotary_emb, i, layer_vb)
    result.layers.add(layer)

  result.norm = newRmsNorm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))

proc causal_mask*(
  self: Qwen3Model,
  batch_size: int,
  tgt_len: int,
  offset: int,
  sliding_window: Option[int]
): TorchTensor =
  let minf = NEG_INFINITY.float32
  var mask_data = @[0.0'f32]
  mask_data = @[]

  for i in 0..<tgt_len:
    for j in 0..<tgt_len + offset:
      let past_ok = j <= i + offset
      let sw_ok = if sliding_window.isSome:
        (i + offset - j) <= sliding_window.get()
      else:
        true
      if past_ok and sw_ok:
        mask_data.add(0.0)
      else:
        mask_data.add(minf)

  let mask_shape = @[batch_size.int64, 1.int64, tgt_len.int64, (tgt_len + offset).int64]
  mask_data.toTorchTensor().reshape(mask_shape)

proc forward*(
  self: Qwen3Model,
  input_ids: TorchTensor,
  offset: int
): TorchTensor =
  let batch_size = input_ids.size(0)
  let seq_len = input_ids.size(1)

  var h = linear(input_ids, self.embed_tokens)

  let causal_mask = if seq_len == 1:
    none(TorchTensor)
  else:
    some(self.causal_mask(batch_size, seq_len, offset, none(int)))

  for layer in self.layers.mitems:
    h = layer.forward(h, causal_mask, offset)

  self.norm.forward(h)

proc clear_kv_cache*(self: Qwen3Model) =
  for layer in self.layers.mitems:
    layer.clear_kv_cache()

type ModelForCausalLM* = ref object
  base*: Qwen3Model
  lm_head*: TorchTensor
  tie_word_embedding*: bool

proc newModelForCausalLM*(cfg: Qwen3Config, vb: VarBuilder): ModelForCausalLM =
  new result

  result.base = newQwen3Model(cfg, vb.clone())
  result.tie_word_embedding = cfg.tie_word_embeddings

  if cfg.tie_word_embeddings:
    result.lm_head = result.base.embed_tokens.clone()
  else:
    result.lm_head = vb.pp("lm_head").get_tensor("weight")

proc forward*(
  self: ModelForCausalLM,
  input_ids: TorchTensor,
  offset: int
): TorchTensor =
  let sizes = input_ids.sizes()
  let l = sizes[1]
  let base_out = self.base.forward(input_ids, offset)
  let last_token = base_out.narrow(1, l - 1, 1)
  linear(last_token, self.lm_head)

proc clear_kv_cache*(self: ModelForCausalLM) =
  self.base.clear_kv_cache()
