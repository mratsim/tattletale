# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  ./kvcache

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

proc init*(_: type InferenceContext, num_layers: int): InferenceContext =
  ## Initialize InferenceContext with KV caches for all layers.
  ##
  ## Args:
  ##   num_layers: Number of transformer layers (one KV cache per layer)
  ##
  ## Note: KV caches are NOT preallocated here.
  ## Call allocateCaches() after to allocate with proper shapes.
  var kv_caches = newSeq[KVCache](num_layers)
  for i in 0..<num_layers:
    kv_caches[i] = KVCache.init()
  
  InferenceContext(
    kv_caches: kv_caches,
    position_ids: F.empty(0)
  )

proc allocateCaches*(ctx: var InferenceContext, batch_size, kv_heads, max_seq, head_dim: int, dtype: ScalarKind, device: DeviceKind) =
  ## Preallocate all KV caches.
  ##
  ## Args:
  ##   batch_size: Number of sequences in batch
  ##   kv_heads: Number of KV heads (GQA)
  ##   max_seq: Maximum sequence length
  ##   head_dim: Dimension per head
  ##   dtype: Data type (e.g., kBFloat16)
  ##   device: Device (e.g., kCUDA)
  for i in 0..<ctx.kv_caches.len:
    ctx.kv_caches[i].allocate(batch_size, kv_heads, max_seq, head_dim, dtype, device)

proc reset*(ctx: var InferenceContext) =
  ## Reset for NEW sequence (not new token!).
  ## Keeps allocated buffers for reuse.
  for i in 0..<ctx.kv_caches.len:
    ctx.kv_caches[i].reset()
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
