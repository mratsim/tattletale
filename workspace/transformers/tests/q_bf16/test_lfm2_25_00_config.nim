# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/lfm2-config \
##   --nimcache:nimcache/tests/lfm2-config \
##   workspace/transformers/tests/q_bf16/test_lfm2_25_00_config.nim

import
  std/options,
  std/os,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/transformers/src/models/lfm2 {.all.}

# Real checkpoint config: hf_models/LFM2.5-230M is a git-ignored symlink
# to the HF-layout checkpoint directory, the convention the Qwen3.5 suites use.
const ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "LFM2.5-230M"

proc checkConfig(): bool =
  let cfg = loadLfm2Config(ModelDir / "config.json")
  doAssert cfg.architecture == "Lfm2ForCausalLM"
  doAssert cfg.model_type == "lfm2"
  doAssert cfg.vocab_size == 65536
  doAssert cfg.hidden_size == 1024
  doAssert cfg.intermediate_size == 2560
  doAssert cfg.num_hidden_layers == 14
  doAssert cfg.num_attention_heads == 16
  doAssert cfg.num_key_value_heads == 8
  doAssert cfg.head_dim == 64
  doAssert cfg.max_position_embeddings == 128000
  doAssert cfg.rms_norm_eps == 1e-5
  doAssert cfg.rope_theta == 1e6
  doAssert cfg.conv_dim == 1024
  doAssert cfg.conv_L_cache == 3
  doAssert cfg.conv_bias == false
  doAssert cfg.tie_word_embeddings == true
  doAssert cfg.layer_types.len == 14
  doAssert cfg.layer_types == @[
    "conv", "conv", "full_attention", "conv", "full_attention", "conv",
    "full_attention", "conv", "full_attention", "conv", "full_attention",
    "conv", "full_attention", "conv"]
  # This checkpoint spells `layer_types` only, so parsing leaves `full_attn_idxs`
  # empty. The derivation path runs only when `layer_types` is absent.
  doAssert cfg.full_attn_idxs.len == 0
  doAssert cfg.dtype == "bfloat16"
  doAssert cfg.bos_token_id == some(1)
  doAssert cfg.eos_token_id == some(7)
  doAssert cfg.pad_token_id == some(0)
  result = true

when isMainModule:
  runCppTest("LFM2.5-230M config parse", checkConfig)
