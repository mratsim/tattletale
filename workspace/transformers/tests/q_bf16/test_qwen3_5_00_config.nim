# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/unittest
import std/tables
import std/options
import std/os
import std/sequtils

import pkg/packedjson

const FIXTURES_DIR = currentSourcePath().parentDir() / ".." / "fixtures"
const CONFIGS_DIR = FIXTURES_DIR / "configs"

import workspace/transformers/src/models/qwen3_5 {.all.}
import workspace/transformers/src/models/all_interfaces

suite "Qwen3.5 Config":
  test "Parse Qwen3.5-0.8B config":
    let cfg = loadQwen3_5Config(CONFIGS_DIR / "config-Qwen3.5-0.8B.json")

    # Wrapper-level fields
    check cfg.architecture == "Qwen3_5ForConditionalGeneration"
    check cfg.model_type == "qwen3_5_text"
    check cfg.transformers_version == "4.57.0.dev0"
    check cfg.image_token_id == 248056
    check cfg.video_token_id == 248057

    # Shared text-stack fields
    check cfg.vocab_size == 248320
    check cfg.hidden_size == 1024
    check cfg.num_hidden_layers == 24
    check cfg.num_attention_heads == 8
    check cfg.num_key_value_heads == 2
    check cfg.head_dim == 256
    check cfg.intermediate_size == 3584
    check cfg.rms_norm_eps == 1e-06
    check cfg.hidden_act == "silu"
    check cfg.max_position_embeddings == 262144
    check cfg.tie_word_embeddings == true
    check cfg.attention_bias == false
    check cfg.attention_dropout == 0.0
    check cfg.use_cache == true

    # dtype key replaces torch_dtype (no torch_dtype key exists)
    check cfg.dtype == "bfloat16"

    # rope_parameters dict
    check cfg.rope_theta == 10000000.0
    check cfg.partial_rotary_factor == 0.25
    check cfg.mrope_interleaved == true
    check cfg.mrope_section == @[11, 11, 10]
    check cfg.rope_type == "default"

    # Gated full attention
    check cfg.attn_output_gate == true
    check cfg.full_attention_interval == 4

    # Hybrid layer types: [linear_attention x3, full_attention] x6
    check cfg.layer_types.len == 24
    check cfg.layer_types.countIt(it == "linear_attention") == 18
    check cfg.layer_types.countIt(it == "full_attention") == 6
    for i in 0 ..< 24:
      let expectFull = (i mod 4) == 3
      check (cfg.layer_types[i] == "full_attention") == expectFull

    # Gated DeltaNet dims
    check cfg.linear_conv_kernel_dim == 4
    check cfg.linear_key_head_dim == 128
    check cfg.linear_num_key_heads == 16
    check cfg.linear_num_value_heads == 16
    check cfg.linear_value_head_dim == 128
    check cfg.mamba_ssm_dtype == "float32"

    # Out-of-scope towers (present in config, not built)
    check cfg.mlp_only_layers.len == 0
    check cfg.mtp_num_hidden_layers == 1
    check cfg.mtp_use_dedicated_embeddings == false

    # Token ids: no bos, eos present
    check cfg.bos_token_id.isNone
    check cfg.eos_token_id == 248044

    check cfg.numKvGroups == 4  # 8 / 2 = 4

  test "Wrapper JSON: top-level model_type and vision_config presence":
    let json = (CONFIGS_DIR / "config-Qwen3.5-0.8B.json").parseFile()
    check json["model_type"].getStr() == "qwen3_5"
    check json.hasKey("vision_config")
    check json["text_config"]["model_type"].getStr() == "qwen3_5_text"

  test "Registry resolves Qwen3_5ForConditionalGeneration":
    const registry = static(ModelRegistry)
    check registry.hasKey("Qwen3_5ForConditionalGeneration")
