# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  std/tables,
  pkg/iface,
  workspace/libtorch,
  workspace/safetensors,
  workspace/toktoktok/src/bpe_codec,

  ../stateful/inference_context

type ModelConfigBase* = ref object
  ## Minimal config shared by all model types for InferenceContext creation.
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
  max_position_embeddings*: int
  eosTokenId*: int  # EOS token ID for generation stop condition

iface *Model:
  proc forward(ctx: var InferenceContext, input_ids: Tensor): Tensor
  proc getConfig(): ModelConfigBase
  proc getTokenizer(): BPETokenizer

var ModelRegistry* {.compileTime.}: Table[string, proc(modelPath: string, device: DeviceKind): Model {.nimcall.}]
  ## Model registry - populated by each model module at initialization
