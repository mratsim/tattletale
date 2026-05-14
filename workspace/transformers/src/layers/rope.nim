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
    cos_cache*: Tensor
    sin_cache*: Tensor

func rotateHalf(x: Tensor): Tensor =
  # Input/Output: (batch, head, seq, head_dim)
  let head_dim = x.size(3)
  let half_dim = head_dim div 2
  let x1 = x[_, _, _, 0..<half_dim]
  let x2 = x[_, _, _, half_dim..<head_dim]
  F.cat([x2.neg(), x1], -1)

func applyRopeImpl(
      q: Tensor,
      k: Tensor,
      cos: Tensor,
      sin: Tensor): (Tensor, Tensor) =
  ## Freestanding RoPE implementation.
  ##
  ## **Contract:** cos and sin MUST be 2D `(seq, head_dim)`.
  ## Shape normalization is the caller's responsibility.
  ##
  ## Input q,k: (batch, seq, head, head_dim)
  ## Input cos, sin: (seq, head_dim)
  ## Output: (batch, seq, head, head_dim)
  doAssert cos.dim == 2, "applyRopeImpl: cos must be 2D (seq, head_dim), got " & $cos.dim & "D"
  doAssert sin.dim == 2, "applyRopeImpl: sin must be 2D (seq, head_dim), got " & $sin.dim & "D"

  # Transpose to (batch, head, seq, head_dim) for rotation
  var q_t = q.transpose(1, 2)
  var k_t = k.transpose(1, 2)

  # Broadcast: (seq, head_dim) -> (1, 1, seq, head_dim) -> matches (batch, head, seq, head_dim)
  let cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
  let sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)

  # Apply rotation for q and k
  let q_rot_t = q_t * cos + rotateHalf(q_t) * sin
  let k_rot_t = k_t * cos + rotateHalf(k_t) * sin

  # Transpose back to (batch, seq, head, head_dim)
  result = (q_rot_t.transpose(1, 2), k_rot_t.transpose(1, 2))

func init*(_: type RotaryPositionEmbedding,
      head_dim, max_seq_len: int,
      rope_theta: float64,
      dtype: ScalarKind,
      device: DeviceKind): RotaryPositionEmbedding =
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
      q: Tensor,
      k: Tensor): (Tensor, Tensor) =
  # Method using cache - calls freestanding applyRopeImpl
  # Input q.k: (batch, seq, head, head_dim)
  # Output: (batch, seq, head, head_dim)

  let seq_len = q.size(1)

  # Slice cache: (seq_len, head_dim) - contiguous due to first-dim slice
  let cos_seq = self.cos_cache[self.cachePos..<self.cachePos+seq_len, _]
  let sin_seq = self.sin_cache[self.cachePos..<self.cachePos+seq_len, _]

  # Advance cache position
  self.cachePos += seq_len

  # Apply rotation using freestanding impl (pass 2D cache, let impl handle broadcasting)
  result = applyRopeImpl(q, k, cos_seq, sin_seq)

func resetCache*(self: var RotaryPositionEmbedding) =
  self.cachePos = 0

func setCache(self: var RotaryPositionEmbedding, cos, sin: Tensor) {.used.} =
  # Private for testing only.
  # Normalizes cos/sin to 2D (seq, head_dim) for storage in cos_cache.
  # RoPE is per-position — identical across all batch items.
  #
  # Handles: (seq, head_dim) [2D] or (batch, seq, head_dim) [3D from HF fixtures]
  var cos_2d = cos
  var sin_2d = sin
  if cos.dim == 3:
    # Take first batch item: (batch, seq, head_dim) -> (seq, head_dim)
    cos_2d = cos[0, _, _]
    sin_2d = sin[0, _, _]
  doAssert cos_2d.dim == 2, "setCache: cos must be 2D or 3D, got " & $cos.dim & "D"
  self.cos_cache = cos_2d
  self.sin_cache = sin_2d
  self.cachePos = 0  # Reset position when loading new cache