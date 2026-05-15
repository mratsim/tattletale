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
  ## DESIGN: Dumb buffer only. No offset tracking.
  ## Offset is owned by the attention layer (derived from ctx.position_ids).
  ## This enforces single-responsibility: cache stores, layer decides where.
  ##
  ## INVARIANT:
  ##   - keys/values shape: (batch_size, kv_heads, max_seq, head_dim)
  ##   - Preallocated (no cat in hot path)

  keys*: Tensor
  values*: Tensor
  ## Debug metadata — derived from buffer shape, useful for diagnostics
  batch_size*: int
  kv_heads*: int
  max_seq*: int
  head_dim*: int

proc init*(
    _: type KVCache,
    batch_size, kv_heads, max_seq, head_dim: int,
    dtype: ScalarKind,
    device: DeviceKind): KVCache =
  ## Initialize KVCache with preallocated buffers.
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

  let shape = [batch_size, kv_heads, max_seq, head_dim]
  KVCache(
    keys: F.empty(shape).to(dtype).to(device),
    values: F.empty(shape).to(dtype).to(device),
    batch_size: batch_size,
    kv_heads: kv_heads,
    max_seq: max_seq,
    head_dim: head_dim
  )

proc write*(cache: var KVCache, k, v: Tensor, offset: int) =
  ## Write KV to cache at given offset.
  ##
  ## Args:
  ##   k, v: Key/value tensors of shape (batch, seq_len, kv_heads, head_dim)
  ##   offset: Write position (owned by attention layer)
  ##
  ## Computes:
  ##   cache.keys[:, :, offset:offset+seq_len, :] = k.permute(0, 2, 1, 3)
  ##   cache.values[:, :, offset:offset+seq_len, :] = v.permute(0, 2, 1, 3)

  let seq_len = k.size(1)

  # Permute to (batch, kv_heads, seq_len, head_dim) for storage
  let k_perm = k.permute([0, 2, 1, 3])
  let v_perm = v.permute([0, 2, 1, 3])

  # Slice assignment
  cache.keys.narrow(2, offset, seq_len).copyFrom(k_perm)
  cache.values.narrow(2, offset, seq_len).copyFrom(v_perm)

proc read*(cache: KVCache, seq_len: int): (Tensor, Tensor) =
  ## Read KV cache from position 0 up to seq_len.
  ##
  ## Returns:
  ##   (keys, values) of shape (batch, kv_heads, seq_len, head_dim)
  ##
  ## Note: Returns view, not copy

  (
    cache.keys.narrow(2, 0, seq_len),
    cache.values.narrow(2, 0, seq_len)
  )
