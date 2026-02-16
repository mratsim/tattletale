# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  workspace/libtorch as F

type
  RotaryPositionEmbedding* = object
    head_dim*: int
    max_seq_len*: int
    rope_theta*: float64
    cos_cache*: TorchTensor
    sin_cache*: TorchTensor

# Input/Output: (batch, head, seq, head_dim)
func rotate_half*(x: TorchTensor): TorchTensor =
  let head_dim = x.size(3)
  let half_dim = head_dim div 2
  let x1 = x[_, _, _, 0..<half_dim]
  let x2 = x[_, _, _, half_dim..<head_dim]
  F.cat([x2.neg(), x1], -1)

# Freestanding RoPE implementation with pre-computed cos/sin tensor
# Input q,k: (batch, seq, head, head_dim)
# Input cos, sin: (batch, seq, head_dim) or (seq, head_dim)
# Output: (batch, seq, head, head_dim)
func apply_rope_impl*(
  q: TorchTensor,
  k: TorchTensor,
  cos: TorchTensor,
  sin: TorchTensor
): (TorchTensor, TorchTensor) =
  # Transpose to (batch, head, seq, head_dim) for rotation
  var q_t = q.transpose(1, 2)
  var k_t = k.transpose(1, 2)
  
  # cos/sin can be:
  # - (seq, head_dim): slice from cache -> unsqueeze(0,1) -> (1, 1, seq, head_dim)
  # - (batch, seq, head_dim): from HF fixture -> unsqueeze(1) -> (batch, 1, seq, head_dim)
  let cos = 
    if cos.dim() == 2: cos.unsqueeze(0).unsqueeze(1)
    else: cos.unsqueeze(1)
  let sin = 
    if sin.dim() == 2: sin.unsqueeze(0).unsqueeze(1)
    else: sin.unsqueeze(1)
  
  # Apply rotation for q
  let q_rot_t = q_t * cos + rotate_half(q_t) * sin
  
  # Apply rotation for k
  let k_rot_t = k_t * cos + rotate_half(k_t) * sin
  
  # Transpose back to (batch, seq, head, head_dim)
  result = (q_rot_t.transpose(1, 2), k_rot_t.transpose(1, 2))

# Output: cos_cache (max_seq_len, head_dim), sin_cache (max_seq_len, head_dim)
func init*(_: type RotaryPositionEmbedding, head_dim, max_seq_len: int, rope_theta: float64, dtype: ScalarKind, device: DeviceKind): RotaryPositionEmbedding =
  let head_dim_float = head_dim.float64
  let inv_freq = F.arange(0, head_dim, 2).to(kFloat64) / head_dim_float
  let rope_theta_tensor = F.full([1], rope_theta, kFloat64)
  let inv_freq_final = F.pow(rope_theta_tensor, -inv_freq)
  let positions = F.arange(0, max_seq_len, kFloat64).unsqueeze(1) * inv_freq_final.unsqueeze(0)
  let fused = F.cat(positions, positions, axis = -1)
  let emb = F.cat(fused.cos(), fused.sin(), axis = -1)
  result.head_dim = head_dim
  result.max_seq_len = max_seq_len
  result.rope_theta = rope_theta
  result.cos_cache = emb[0..<max_seq_len, 0..<head_dim].to(dtype).to(device)
  result.sin_cache = emb[0..<max_seq_len, head_dim..<2*head_dim].to(dtype).to(device)

# Method using cache - calls freestanding apply_rope_impl
# Input q.k: (batch, seq, head, head_dim)
# Input offset: scalar offset into the cache (for prefill, typically 0)
# Output: (batch, seq, head, head_dim)
proc apply_rope*(
  self: RotaryPositionEmbedding,
  q: TorchTensor,
  k: TorchTensor,
  offset: int
): (TorchTensor, TorchTensor) =
  let seq_len = q.size(1)
  
  # Slice cache: (seq_len, head_dim) - contiguous due to first-dim slice
  let cos_seq = self.cos_cache[offset..<offset+seq_len, _]
  let sin_seq = self.sin_cache[offset..<offset+seq_len, _]
  
  # Apply rotation using freestanding impl (pass 2D cache, let impl handle broadcasting)
  result = apply_rope_impl(q, k, cos_seq, sin_seq)
