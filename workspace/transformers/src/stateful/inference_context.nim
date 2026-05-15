# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  ./kvcache,
  ../layers/rope

type InferenceContext* = object
  ## State container for a SINGLE forward pass.
  ## Created by orchestrator per request, passed through layers.
  ##
  ## LIFECYCLE:
  ##   - Created at start of sequence (prefill)
  ##   - Reused across decode steps (kv_caches accumulate)
  ##   - position_ids updated each forward pass
  ##   - Discarded when sequence completes

  kv_caches*: seq[KVCache]    # One per layer (preallocated)
  position_ids*: Tensor       # [0,1,2] for prefill, [3] for decode, [6,3,11] for ragged
  ## Debug metadata — describes the context configuration
  num_layers*: int
  batch_size*: int
  kv_heads*: int
  max_seq*: int
  head_dim*: int

  ## RoPE cos/sin for current forward pass.
  ## Sliced from the model's precomputed cache using position_ids.
  cos*: Tensor   ## (seq_len, head_dim) — valid after setRopeForPositions()
  sin*: Tensor   ## (seq_len, head_dim) — valid after setRopeForPositions()

proc init*(
    _: type InferenceContext,
    num_layers: int,
    batch_size, kv_heads, max_seq, head_dim: int,
    dtype: ScalarKind,
    device: DeviceKind): InferenceContext =
  ## Initialize InferenceContext with preallocated KV caches for all layers.
  ##
  ## Args:
  ##   num_layers: Number of transformer layers (one KV cache per layer)
  ##   batch_size: Number of sequences in batch
  ##   kv_heads: Number of KV heads (GQA)
  ##   max_seq: Maximum sequence length
  ##   head_dim: Dimension per head
  ##   dtype: Data type (e.g., kBFloat16)
  ##   device: Device (e.g., kCUDA)

  var kv_caches = newSeq[KVCache](num_layers)
  for i in 0..<num_layers:
    kv_caches[i] = KVCache.init(batch_size, kv_heads, max_seq, head_dim, dtype, device)

  InferenceContext(
    kv_caches: kv_caches,
    position_ids: F.empty(0),
    cos: F.empty(0),
    sin: F.empty(0),
    num_layers: num_layers,
    batch_size: batch_size,
    kv_heads: kv_heads,
    max_seq: max_seq,
  head_dim: head_dim
  )

proc setRopeForPositions*(ctx: var InferenceContext, rotary: RotaryPositionEmbeddingRef) =
  ## Populate ctx.cos and ctx.sin from the model's RoPE cache.
  ##
  ## Internally calls `rotary.ropeByPositions(ctx.position_ids)` and stores
  ## the result in `ctx.cos` / `ctx.sin` for downstream attention layers.
  ##
  ## Called once per forward pass. The model is responsible for calling this
  ## so the orchestrator stays ignorant of rope variants.
  (ctx.cos, ctx.sin) = rotary.ropeByPositions(ctx.position_ids)

proc reset*(ctx: var InferenceContext) =
  ## Reset for NEW sequence (not new token!).
  ## Keeps allocated buffers for reuse.
  ctx.position_ids = F.empty(0)

proc setPositionIds*(ctx: var InferenceContext, position_ids: Tensor) =
  ## Set position_ids for current forward pass.
  ##
  ## Args:
  ##   position_ids: Tensor of shape (batch, seq_len) or (seq_len,)
  ##
  ## Note: Called every forward pass (prefill + each decode step)
  ctx.position_ids = position_ids

proc setPositionIdsArange*(ctx: var InferenceContext, seq_len: int, offset: int = 0, device: DeviceKind = kCPU) =
  ## Set position_ids to arange(offset, offset+seq_len).
  ##
  ## Convenience proc for common case.
  ##
  ## Args:
  ##   seq_len: Sequence length
  ##   offset: Starting offset (default 0 for prefill)
  ##   device: Device for tensor
  ctx.position_ids = F.arange(offset, offset + seq_len, device=device)
