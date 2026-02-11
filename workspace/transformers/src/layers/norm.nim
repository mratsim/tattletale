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

proc forward*(self: RmsNorm, x: TorchTensor): TorchTensor =
  let normalized_shape = asTorchView([self.hidden_size])
  rms_norm(x, normalized_shape, self.weight, self.eps)

proc forward_with_residual*(self: RmsNorm, x, residual: TorchTensor): (TorchTensor, TorchTensor) =
  let orig_dtype = x.scalarType()
  let x_float = x.to(kFloat32).add(residual.to(kFloat32))
  let residual_output = x_float.to(orig_dtype)
  let var_mean = x_float.pow(2).mean(-1, true)
  let inv_std = F.pow(var_mean + self.eps, -0.5)
  let normalized = x_float * inv_std
  result = (normalized.to(orig_dtype) * self.weight, residual_output)
