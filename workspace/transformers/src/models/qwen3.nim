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
  ../layers/all,
  ./all
################################################################################
#                                                                              #
#                          Qwen3 Configuration                                 #
#                                                                              #
################################################################################

type
  Qwen3Config* = ref object of ModelConfig
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
