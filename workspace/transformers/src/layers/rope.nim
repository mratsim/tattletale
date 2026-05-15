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
  RotaryPositionEmbeddingRef* = ref object
    ## RoPE: precomputed cos/sin cache for all positions (0..max_seq_len-1).
    ##
    ## USAGE:
    ##   - Model calls compute(position_ids) to get (cos, sin) for current forward pass
    ##   - Layers receive precomputed (cos, sin) and call applyRope()
    ##   - NO internal state (cachePos removed)
    head_dim*: int
    max_seq_len*: int
    rope_theta*: float64
    cos_cache*: Tensor    # (max_seq_len, head_dim)
    sin_cache*: Tensor    # (max_seq_len, head_dim)

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

func new*(_: type RotaryPositionEmbeddingRef,
      head_dim, max_seq_len: int,
      rope_theta: float64,
      dtype: ScalarKind,
      device: DeviceKind): RotaryPositionEmbeddingRef =
  ## RoPE: cos(pos * inv_freq[i]), sin(pos * inv_freq[i]) for i in 0..head_dim-1
  ## inv_freq[i] = 1/theta^(i/head_dim), but we compute for i=0,2,4,...,head_dim-2 (64 values for head_dim=128)
  ## Then repeat each value twice to get head_dim cos/sin values.
  let head_dim_float = head_dim.float64
  let inv_freq = F.arange(0, head_dim, 2).to(kFloat64) / head_dim_float  # [0, 2, ..., 126] / 128
  let rope_theta_tensor = F.full([1], rope_theta, kFloat64)
  let inv_freq_final = F.pow(rope_theta_tensor, -inv_freq)  # theta^(-inv_freq), shape (64,)
  let positions = F.arange(0, max_seq_len, kFloat64).unsqueeze(1) * inv_freq_final.unsqueeze(0)  # (max_seq_len, 64)
  let cos_64 = positions.cos()  # (max_seq_len, 64)
  let sin_64 = positions.sin()  # (max_seq_len, 64)
  new(result)
  # Repeat each value twice: [cos0, cos0, cos1, cos1, ...] to get (max_seq_len, head_dim)
  result.head_dim = head_dim
  result.max_seq_len = max_seq_len
  result.rope_theta = rope_theta
  # NEOX style: concatenate cos with itself [c0,c1,...,c63,c0,c1,...,c63]
  # Store in model dtype (BF16 for Qwen3)
  result.cos_cache = F.cat([cos_64, cos_64], -1).to(dtype).to(device)  # (max_seq_len, head_dim)
  result.sin_cache = F.cat([sin_64, sin_64], -1).to(dtype).to(device)  # (max_seq_len, head_dim)
  # NEOX style: concatenate cos with itself [c0,c1,...,c63,c0,c1,...,c63]
  # Store in FP32 for precision (HF pattern), cast during apply
  result.cos_cache = F.cat([cos_64, cos_64], -1).to(dtype).to(device)  # (max_seq_len, 128)
  result.sin_cache = F.cat([sin_64, sin_64], -1).to(dtype).to(device)  # (max_seq_len, 128)

proc compute*(self: RotaryPositionEmbeddingRef, position_ids: Tensor): (Tensor, Tensor) =
  ## Slice cos/sin cache using position_ids.
  ##
  ## Args:
  ##   position_ids: Tensor of shape (seq_len,) or (batch, seq_len)
  ##
  ## Returns:
  ##   (cos, sin) of shape (seq_len, head_dim) — sliced from cache
  ##
  ## Note:
  ##   Called once per forward pass at model level.
  ##   Layers receive precomputed (cos, sin), not position_ids.

  # Handle 1D or 2D position_ids
  var pos_ids = position_ids
  if pos_ids.dim == 2:
    # Take first batch item (positions same for all batch items)
    pos_ids = pos_ids[0, _]

  # Slice cache using position_ids (advanced indexing)
  # cos_cache[position_ids, :] → (seq_len, head_dim)
  result = (self.cos_cache.index_select(0, pos_ids), self.sin_cache.index_select(0, pos_ids))

proc applyRope*(
    self: RotaryPositionEmbeddingRef,
    q: Tensor,
    k: Tensor,
    cos, sin: Tensor): (Tensor, Tensor) =
  ## Apply RoPE using precomputed cos/sin.
  ##
  ## Args:
  ##   q, k: Input tensors of shape (batch, seq, head, head_dim)
  ##   cos, sin: Precomputed RoPE of shape (seq, head_dim)
  ##
  ## Returns:
  ##   (q_rot, k_rot) of shape (batch, seq, head, head_dim)
  ##
  ## Note:
  ##   Pure function — no mutation of self.
  ##   cos/sin must match seq_len of q/k.

  # Just call the freestanding impl
  result = applyRopeImpl(q, k, cos, sin)

func setCache(self: var RotaryPositionEmbeddingRef, cos, sin: Tensor) {.used.} =
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
