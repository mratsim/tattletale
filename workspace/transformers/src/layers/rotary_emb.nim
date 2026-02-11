# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/math
import workspace/libtorch

{.experimental: "views".}

const NEG_INFINITY = -1e10

type RotaryEmb* = ref object
  head_dim*: int
  max_seq_len*: int
  rope_theta*: float64
  cos_cache*: TorchTensor
  sin_cache*: TorchTensor

proc newRotaryEmb*(
  head_dim: int,
  max_seq_len: int,
  rope_theta: float64,
  dtype: ScalarKind,
  device: DeviceKind
): RotaryEmb =
  new result
  result.head_dim = head_dim
  result.max_seq_len = max_seq_len
  result.rope_theta = rope_theta

  let inv_freq_len = head_dim div 2

  var inv_freq: seq[float32] = @[]
  for i in 0..<inv_freq_len:
    let exp = -2.0 * i.float64 / head_dim.float64
    let freq = pow(rope_theta, exp).float32
    inv_freq.add(freq)

  let inv_freq_tensor = inv_freq.toTorchTensor()
  let positions = arange(max_seq_len.int64, kFloat32).to(dtype)

  let shape = @[max_seq_len.int64, 1.int64]
  let freqs = positions.reshape(shape) * inv_freq_tensor

  result.cos_cache = freqs.cos().to(dtype)
  result.sin_cache = freqs.sin().to(dtype)

proc rotate_half*(x: TorchTensor): TorchTensor =
  let dim = x.size(-1)
  let half_dim = dim div 2

  let x1 = x.narrow(-1, 0, half_dim)
  let x2 = x.narrow(-1, half_dim, half_dim)

  let neg_x2 = -x2
  result = cat(@[neg_x2, x1], -1)

proc apply_rope*(
  self: RotaryEmb,
  q: TorchTensor,
  k: TorchTensor,
  offset: int
): (TorchTensor, TorchTensor) =
  let seq_len = q.size(2)

  let cos_slice = self.cos_cache.narrow(0, offset, seq_len).unsqueeze(1)
  let sin_slice = self.sin_cache.narrow(0, offset, seq_len).unsqueeze(1)

  let q_rot = q * cos_slice + rotate_half(q) * sin_slice
  let k_rot = k * cos_slice + rotate_half(k) * sin_slice

  (q_rot, k_rot)
