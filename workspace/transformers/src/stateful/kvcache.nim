# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F

type KVCache* = object
  ## KV Cache for attention layers.
  ## Managed by Orchestrator, passed to layers via InferenceContext.
  ##
  ## INVARIANT:
  ##   - keys/values shape: (batch, kv_heads, max_seq, head_dim)
  ##   - Preallocated (no cat in hot path)
  ##   - offset <= max_seq

  keys*: Tensor
  values*: Tensor
  offset*: int              # Current write position

proc init*(_: type KVCache): KVCache =
  ## Initialize empty KVCache.
  ## Actual allocation happens when first write occurs.
  KVCache(
    keys: F.empty(0),
    values: F.empty(0),
    offset: 0
  )

proc reset*(cache: var KVCache) =
  ## Reset cache for new sequence.
  ## Does NOT deallocate — keeps buffer for reuse.
  cache.offset = 0

proc allocate*(cache: var KVCache, batch_size, kv_heads, max_seq, head_dim: int, dtype: ScalarKind, device: DeviceKind) =
  ## Preallocate KV cache buffer.
  ##
  ## Args:
  ##   batch_size: Number of sequences in batch
  ##   kv_heads: Number of KV heads (GQA)
  ##   max_seq: Maximum sequence length
  ##   head_dim: Dimension per head
  ##   dtype: Data type (e.g., kBFloat16)
  ##   device: Device (e.g., kCUDA)
  ##
  ## Shape: (batch, kv_heads, max_seq, head_dim)
  ## Note: Permuted for efficient attention backend access
  cache.keys = F.empty([batch_size, kv_heads, max_seq, head_dim]).to(dtype).to(device)
  cache.values = F.empty([batch_size, kv_heads, max_seq, head_dim]).to(dtype).to(device)
  cache.offset = 0

proc store*(cache: var KVCache, k, v: Tensor, offset: int) =
  ## Store KV to cache at given offset.
  ## Existing values are overwritten
  ##
  ## Args:
  ##   k, v: New key/value tensors of shape (batch, seq_len, kv_heads, head_dim)
  ##   offset: Write position (usually cache.offset)
  ##
  ## Computes:
  ##   cache.keys[:, :, offset:offset+seq_len, :] = k.permute(0, 2, 1, 3)
  ##   cache.values[:, :, offset:offset+seq_len, :] = v.permute(0, 2, 1, 3)
  ##   cache.offset = offset + seq_len

  let batch = k.size(0)
  let seq_len = k.size(1)
  let kv_heads = k.size(2)
  let head_dim = k.size(3)

  # Permute to (batch, kv_heads, seq_len, head_dim) for storage
  let k_perm = k.permute([0, 2, 1, 3])
  let v_perm = v.permute([0, 2, 1, 3])

  # Slice assignment -- TODO: the []= macro should be updated to support this
  # cache.keys[:, :, offset:offset+seq_len, :] = k_perm
  cache.keys.narrow(2, offset, seq_len).copyFrom(k_perm)
  cache.values.narrow(2, offset, seq_len).copyFrom(v_perm)

  # Update offset
  cache.offset = offset + seq_len

proc getKV*(cache: KVCache, seq_len: int): (Tensor, Tensor) =
  ## Get KV cache up to current length.
  ##
  ## Returns:
  ##   (keys, values) of shape (batch, kv_heads, seq_len, head_dim)
  ##   where seq_len = cache.offset
  ##
  ## Note: Returns view, not copy
  (
    cache.keys.narrow(2, 0, cache.offset),
    cache.values.narrow(2, 0, cache.offset)
  )
