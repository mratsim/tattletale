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

func rotate_half*(self: RotaryPositionEmbedding, x: TorchTensor): TorchTensor =
  let dim = x.size(-1)
  let x1 = x[0..<dim div 2]
  let x2 = x[dim div 2..<dim]
  F.cat(@[x2.neg(), x1], axis = -1)

func init*(_: type RotaryPositionEmbedding, head_dim, max_seq_len: int, rope_theta: float64, dtype: ScalarKind, device: DeviceKind): RotaryPositionEmbedding =
  let inv_freq = F.arange(0, head_dim, 2).to(kFloat64) / (head_dim.float)
  inv_freq = inv_freq / pow(rope_theta, inv_freq)
  let positions = F.arange(0, max_seq_len, kFloat64).unsqueeze(1) * inv_freq.unsqueeze(0)
  let fused = F.cat(positions, positions, dim = -1)
  let emb = F.cat(fused.cos(), fused.sin(), dim = -1)
  result.head_dim = head_dim
  result.max_seq_len = max_seq_len
  result.rope_theta = rope_theta
  result.cos_cache = emb[0..<max_seq_len, 0..<head_dim].to(dtype).to(device)
  result.sin_cache = emb[0..<max_seq_len, head_dim..<2*head_dim].to(dtype).to(device)

func apply_rope*(
  self: RotaryPositionEmbedding,
  q: TorchTensor,
  k: TorchTensor,
  offset: int
): (TorchTensor, TorchTensor) =
  let seq_len = q.size(2)
  let cos = self.cos_cache[offset..<offset+seq_len].unsqueeze(1)
  let sin = self.sin_cache[offset..<offset+seq_len].unsqueeze(1)
  let q_rot = q * cos + self.rotate_half(q) * sin
  let k_rot = k * cos + self.rotate_half(k) * sin
  (q_rot, k_rot)
