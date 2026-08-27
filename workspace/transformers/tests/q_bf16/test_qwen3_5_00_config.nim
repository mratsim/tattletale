# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-config \
##   --nimcache:nimcache/tests/qwen35-config \
##   workspace/transformers/tests/q_bf16/test_qwen3_5_00_config.nim

import
  std/tables,
  std/options,
  std/os,
  std/sequtils,
  pkg/packedjson,
  workspace/libtorch_testutils,
  workspace/transformers/src/models/qwen3_5 {.all.},
  workspace/transformers/src/models/all_interfaces

const FIXTURES_DIR = currentSourcePath().parentDir() / ".." / "fixtures"
const CONFIGS_DIR = FIXTURES_DIR / "configs"

proc main() =
  runCppTest "Parse Qwen3.5-0.8B config":
    proc(): bool =
      let cfg = loadQwen3_5Config(CONFIGS_DIR / "config-Qwen3.5-0.8B.json")

      # Wrapper-level fields
      doAssert cfg.architecture == "Qwen3_5ForConditionalGeneration"
      doAssert cfg.model_type == "qwen3_5_text"
      doAssert cfg.transformers_version == "4.57.0.dev0"
      doAssert cfg.image_token_id == 248056
      doAssert cfg.video_token_id == 248057

      # Shared text-stack fields
      doAssert cfg.vocab_size == 248320
      doAssert cfg.hidden_size == 1024
      doAssert cfg.num_hidden_layers == 24
      doAssert cfg.num_attention_heads == 8
      doAssert cfg.num_key_value_heads == 2
      doAssert cfg.head_dim == 256
      doAssert cfg.intermediate_size == 3584
      doAssert cfg.rms_norm_eps == 1e-06
      doAssert cfg.hidden_act == "silu"
      doAssert cfg.max_position_embeddings == 262144
      doAssert cfg.tie_word_embeddings == true
      doAssert cfg.attention_bias == false
      doAssert cfg.attention_dropout == 0.0
      doAssert cfg.use_cache == true

      # dtype key replaces torch_dtype (no torch_dtype key exists)
      doAssert cfg.dtype == "bfloat16"

      # rope_parameters dict
      doAssert cfg.rope_theta == 10000000.0
      doAssert cfg.partial_rotary_factor == 0.25
      doAssert cfg.mrope_interleaved == true
      doAssert cfg.mrope_section == @[11, 11, 10]
      doAssert cfg.rope_type == "default"

      # Gated full attention
      doAssert cfg.attn_output_gate == true
      doAssert cfg.full_attention_interval == 4

      # Hybrid layer types: [linear_attention x3, full_attention] x6
      doAssert cfg.layer_types.len == 24
      doAssert cfg.layer_types.countIt(it == "linear_attention") == 18
      doAssert cfg.layer_types.countIt(it == "full_attention") == 6
      for i in 0 ..< 24:
        let expectFull = (i mod 4) == 3
        doAssert (cfg.layer_types[i] == "full_attention") == expectFull

      # Gated DeltaNet dims
      doAssert cfg.linear_conv_kernel_dim == 4
      doAssert cfg.linear_key_head_dim == 128
      doAssert cfg.linear_num_key_heads == 16
      doAssert cfg.linear_num_value_heads == 16
      doAssert cfg.linear_value_head_dim == 128
      doAssert cfg.mamba_ssm_dtype == "float32"

      # Out-of-scope towers (present in config, not built)
      doAssert cfg.mlp_only_layers.len == 0
      doAssert cfg.mtp_num_hidden_layers == 1
      doAssert cfg.mtp_use_dedicated_embeddings == false

      # Token ids: no bos, eos present
      doAssert cfg.bos_token_id.isNone
      doAssert cfg.eos_token_id == 248044

      doAssert cfg.numKvGroups == 4  # 8 / 2 = 4
      true

  runCppTest "Wrapper JSON: top-level model_type and vision_config presence":
    proc(): bool =
      let json = (CONFIGS_DIR / "config-Qwen3.5-0.8B.json").parseFile()
      doAssert json["model_type"].getStr() == "qwen3_5"
      doAssert json.hasKey("vision_config")
      doAssert json["text_config"]["model_type"].getStr() == "qwen3_5_text"
      true

  runCppTest "Registry resolves Qwen3_5ForConditionalGeneration":
    proc(): bool =
      const registry = static(ModelRegistry)
      doAssert registry.hasKey("Qwen3_5ForConditionalGeneration")
      true

when isMainModule:
  main()
