# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  ./kvcache,
  ./page_pool,
  ../layers/rope

type InferenceContext* = ref object
  ## State container for a SINGLE forward pass.
  ## Created by orchestrator per request, passed through layers.
  ##
  ## LIFECYCLE:
  ##   - Created at start of sequence (prefill)
  ##   - Reused across decode steps (pages accumulate)
  ##   - position_ids updated each forward pass
  ##   - Discarded when sequence completes

  pages*: seq[Page]           # KV pages: matched from trie + newly borrowed
  kv_position*: int            # Write cursor (next token position to write into)
  input_tokens*: seq[uint32]   # Token sequence tracking (for graftPages at sequence end)

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
    batch_size: int,
    kv_heads: int,
    max_seq: int,
    head_dim: int): InferenceContext =
  ## Initialize InferenceContext with empty KV page tracking.
  ## KV buffer dimensions are on PagePool (owned by orchestrator).
  ##
  ## Args:
  ##   num_layers: Number of transformer layers
  ##   batch_size: Number of sequences in batch (metadata)
  ##   kv_heads: Number of KV heads (GQA) (metadata)
  ##   max_seq: Maximum sequence length (metadata)
  ##   head_dim: Dimension per head (metadata)

  InferenceContext(
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

proc clearState*(ctx: var InferenceContext) =
  ## Clear KV state for reuse in a new sequence.
  ## Drops page references (GC may recycle to pool).
  ## Keeps metadata fields (num_layers, head_dim, etc.) for reuse.
  ##
  ## NOTE: cos/sin (RoPE) are NOT cleared — they are stable per model
  ## and are overwritten by the next `setRopeForPositions` call.
  ## Stale cos/sin cannot leak between sequences because
  ## `setRopeForPositions` is always called before any forward pass.
  ctx.pages = default(seq[Page])
  ctx.kv_position = 0
  ctx.input_tokens = default(seq[uint32])
  ctx.position_ids = nil

proc setPositionIds*(ctx: var InferenceContext, position_ids: Tensor) =
  ## Set position_ids for current forward pass.
  ##
  ## Args:
  ##   position_ids: Tensor of shape (batch, seq_len) or (seq_len,)
  ##
  ## Note: Called every forward pass (prefill + each decode step)
  ctx.position_ids = position_ids

proc setPositionIdsArange*(ctx: var InferenceContext, seq_len: int, offset: int = 0, device: DeviceKind = kCPU) =
  ## Set position_ids to arange(offset, offset+seq_len) as int64.
  ##
  ## Convenience proc for common case.
  ##
  ## Note: dtype is now kInt64 (changed from previous default).
  ## The old `device=device` keyword argument was replaced with
  ## explicit `tensorOptions(kInt64, device)`.
  ##
  ## Args:
  ##   seq_len: Sequence length
  ##   offset: Starting offset (default 0 for prefill)
  ##   device: Device for tensor
  let opts = F.tensorOptions(F.kInt64, device)
  ctx.position_ids = F.arange(offset, offset + seq_len, opts)
