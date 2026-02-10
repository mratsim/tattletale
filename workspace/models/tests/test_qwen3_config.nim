# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/unittest
import std/options
import std/json

import workspace/models/src/config

suite "Qwen3 Config":
  test "Parse Qwen3-0.6B config":
    let cfg = loadQwen3Config("tests/fixtures/configs/config-Qwen3-0.6B.json")

    check cfg.architecture == "Qwen3ForCausalLM"
    check cfg.model_type == "qwen3"
    check cfg.vocab_size == 151936
    check cfg.hidden_size == 1024
    check cfg.num_hidden_layers == 28
    check cfg.num_attention_heads == 16
    check cfg.num_key_value_heads == 8
    check cfg.head_dim == 128
    check cfg.intermediate_size == 3072
    check cfg.hidden_act == "silu"
    check cfg.max_position_embeddings == 40960
    check cfg.rope_theta == 1000000.0
    check cfg.rms_norm_eps == 1e-06
    check cfg.tie_word_embeddings == true
    check cfg.bos_token_id == 151643
    check cfg.eos_token_id == 151645
    check cfg.sliding_window.isNone
    check cfg.rope_scaling.kind == JNull

    check cfg.numKvGroups == 2  # 16 / 8 = 2

  test "Parse Qwen3-4B config":
    let cfg = loadQwen3Config("tests/fixtures/configs/config-Qwen3-4B.json")

    check cfg.hidden_size == 2560
    check cfg.num_hidden_layers == 36
    check cfg.num_attention_heads == 32
    check cfg.intermediate_size == 9728

    check cfg.numKvGroups == 4  # 32 / 8 = 4

  test "Parse Qwen3-4B-AWQ config has quantization":
    let cfg = loadQwen3Config("tests/fixtures/configs/config-Qwen3-4B-AWQ.json")

    check cfg.rope_scaling.isNil == false
