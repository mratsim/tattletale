# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F

{.experimental: "callOperator".}

type
  RmsNorm* = ref object
    weight: Tensor
    eps*: float
    hidden_size*: int

func init*(_: type RmsNorm, weight: Tensor, eps: float = 1e-6): RmsNorm =
  let hidden_size = weight.size(0)
  RmsNorm(weight: weight, eps: eps, hidden_size: hidden_size)

proc forward*(self: RmsNorm, hidden_state: Tensor): Tensor =
  ## Forward pass with float32 upcasting for normalization:
  ##   1. Converts to FP32 for numerical stability
  ##   2. Squares
  ##   3. `.sqrt().reciprocal`
  ##      The single instruction rsqrt accumulates 0.03 abs error
  ##   4. Converts normalized result back to input dtype
  ##   5. Multiplies by weight in INPUT DTYPE (matches HF's Qwen3RMSNorm)
  ##
  ## HF's Qwen3RMSNorm does: self.weight * hidden_states.to(input_dtype)
  ## The weight multiplication is in the input dtype (bf16), not fp32.
  ##
  ## Without float32 for the normalization step itself, we accumulate errors
  ## of ~0.03 per layer from the rsqrt approximation.
  let input_dtype = hidden_state.scalarType()
  let x = hidden_state.to(kFloat32)
  let variance = x.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(Scalar(self.eps)).rsqrt()
  let normalized = x * rstd
  self.weight * normalized.to(input_dtype)
proc forward_with_residual(self: RmsNorm, hidden_state, residual: Tensor): (Tensor, Tensor) =
  ## Fused residual addition + RMSNorm.
  # The residual addition is done in the input dtype (BF16).
  # The RMSNorm then converts to FP32, normalizes, and multiplies by weight
  # in the input dtype (matches HF's Qwen3RMSNorm behavior exactly).
  let new_residual = hidden_state + residual
  (self.forward(new_residual), new_residual)

## Call operator overloads:
## - norm(x)              → forward(x)           → Tensor
## - norm(x, residual)    → forward_with_residual(x, residual) → (Tensor, Tensor)
template `()`*(layer: RmsNorm, x: Tensor): untyped =
  forward(layer, x)

template `()`*(layer: RmsNorm, x, residual: Tensor): untyped =
  forward_with_residual(layer, x, residual)
