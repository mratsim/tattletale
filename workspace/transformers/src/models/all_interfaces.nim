# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  std/tables,
  pkg/iface,
  workspace/libtorch,
  workspace/safetensors,
  workspace/toktoktok/src/bpe_codec,

  ../instrumentation,
  ../stateful/inference_context,
  ./loading/layer_kinds

export instrumentation

## Shared model interfaces for the transformers package.
##
## Defines ModelConfigBase and the AnyModel interface that every model
## module implements for the generate() entry point.

## Minimal config shared by all model types for InferenceContext creation.
type ModelConfigBase* = ref object
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
  eosTokenIds*: seq[int] = @[]
  # Stop set for a checkpoint whose eos_token_id ships as a list.
  # Empty keeps the single-eos field governing, zero behavior change across families.
  # MLA latent-cache pool shape, zero on non-MLA checkpoints.
  # The K buffer carries the compressed latent, the V buffer holds the kpe
  # plane. Zero keeps the pool shape keyed on num_key_value_heads and head_dim.
  mlaKvLoraRank*: int
  mlaKpeWidth*: int
  # Widest per-head KV width across the checkpoint's layer kinds, zero when
  # every layer stores head_dim channels. A dual-width checkpoint, one
  # whose sliding window sits narrower than the full-attention head dim,
  # writes full-width rows in its full layers, the pool slots must carry
  # the widest width.
  kvHeadDimMax*: int
  layerKinds*: seq[AttentionLayerKind] = @[]

iface *AnyModel:
  proc forward(ctx: var InferenceContext, input_ids: Tensor): Tensor
  proc getConfig(): ModelConfigBase
  proc getTokenizer(): BPETokenizer
  proc getDeviceKind(): DeviceKind
var ModelRegistry* {.compileTime.}: Table[string, proc(modelPath: string, device: DeviceKind): AnyModel {.nimcall.}]
  ## Model registry - populated by each model module at initialization
