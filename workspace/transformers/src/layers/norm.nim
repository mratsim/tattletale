# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  workspace/libtorch as F,
  workspace/libtorch/src/abi/neural_nets

type
  RmsNorm* = object
    weight*: TorchTensor
    eps*: float
    hidden_size*: int

func init*(_: type RmsNorm, weight: TorchTensor, eps: float = 1e-6): RmsNorm =
  let hidden_size = weight.size(0).int
  RmsNorm(weight: weight, eps: eps, hidden_size: hidden_size)

proc forward*(self: RmsNorm, hidden_state: TorchTensor): TorchTensor =
  let normalized_shape = asTorchView(self.hidden_size)
  rms_norm(hidden_state, normalized_shape, self.weight, self.eps)

proc forward_with_residual*(self: RmsNorm, hidden_state, residual: TorchTensor): (TorchTensor, TorchTensor) =
  # Usually at the end of a transformer block
  # you do
  #   x = self.post_layernorm.forward(hidden_state)
  #   return x + residual
  #
  # Instead you can defer the addition to the beginning of the next layer
  # where it's easier to fuse with rms_norm (once an optimized kernel is there)
  let new_residual = hidden_state + residual # TODO: fused add + RMSNorm
  let normalized_shape = asTorchView(self.hidden_size)
  let normalized = rms_norm(new_residual, normalized_shape, self.weight, self.eps)
  (normalized, new_residual)
