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
    rope_theta: float64
    cachePos: int
    cos_cache: TorchTensor
    sin_cache: TorchTensor

func rotateHalf(x: TorchTensor): TorchTensor =
  # Input/Output: (batch, head, seq, head_dim)
  let head_dim = x.size(3)
  let half_dim = head_dim div 2
  let x1 = x[_, _, _, 0..<half_dim]
  let x2 = x[_, _, _, half_dim..<head_dim]
  F.cat([x2.neg(), x1], -1)

func applyRopeImpl*(
  q: TorchTensor,
  k: TorchTensor,
  cos: TorchTensor,
  sin: TorchTensor
): (TorchTensor, TorchTensor) =
  # Freestanding RoPE implementation with pre-computed cos/sin tensor
  # Input q,k: (batch, seq, head, head_dim)
  # Input cos, sin: (batch, seq, head_dim) or (seq, head_dim)
  # Output: (batch, seq, head, head_dim)

  # Transpose to (batch, head, seq, head_dim) for rotation
  var q_t = q.transpose(1, 2)
  var k_t = k.transpose(1, 2)

  # cos/sin can be:
  # - (seq, head_dim): slice from cache -> unsqueeze(0,1) -> (1, 1, seq, head_dim)
  # - (batch, seq, head_dim): from HF fixture -> unsqueeze(1) -> (batch, 1, seq, head_dim)
  let cos = cos.unsqueeze(1)
  let sin = sin.unsqueeze(1)

  # Apply rotation for q and k
  let q_rot_t = q_t * cos + rotateHalf(q_t) * sin
  let k_rot_t = k_t * cos + rotateHalf(k_t) * sin

  # Transpose back to (batch, seq, head, head_dim)
  result = (q_rot_t.transpose(1, 2), k_rot_t.transpose(1, 2))

func init*(_: type RotaryPositionEmbedding, head_dim, max_seq_len: int, rope_theta: float64, dtype: ScalarKind, device: DeviceKind): RotaryPositionEmbedding =
  # Output: cos_cache (max_seq_len, head_dim), sin_cache (max_seq_len, head_dim)
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
  result.cachePos = 0
  result.cos_cache = emb[0..<max_seq_len, 0..<head_dim].to(dtype).to(device)
  result.sin_cache = emb[0..<max_seq_len, head_dim..<2*head_dim].to(dtype).to(device)

proc applyRope*(
    self: var RotaryPositionEmbedding,
    q: TorchTensor,
    k: TorchTensor,
  ): (TorchTensor, TorchTensor) =
  # Method using cache - calls freestanding apply_rope_impl
  # Input q.k: (batch, seq, head, head_dim)
  # Output: (batch, seq, head, head_dim)

  let seq_len = q.size(1)

  # Slice cache: (seq_len, head_dim) - contiguous due to first-dim slice
  let cos_seq = self.cos_cache[self.cachePos..<self.cachePos+seq_len, _]
  let sin_seq = self.sin_cache[self.cachePos..<self.cachePos+seq_len, _]

  # Advance cache position
  self.cachePos += seq_len

  # Apply rotation using freestanding impl (pass 2D cache, let impl handle broadcasting)
  result = apply_rope_impl(q, k, cos_seq, sin_seq)

func resetCache*(self: var RotaryPositionEmbedding) =
  self.cachePos = 0

func setCache(self: var RotaryPositionEmbedding, cos, sin: TorchTensor) {.used.} =
  # Private for testing only
  self.cos_cache = cos
  self.sin_cache = sin