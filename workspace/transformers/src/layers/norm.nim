# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F

type
  RmsNorm* = object
    weight*: Tensor
    eps*: float
    hidden_size*: int

func init*(_: type RmsNorm, weight: Tensor, eps: float = 1e-6): RmsNorm =
  let hidden_size = weight.size(0)
  RmsNorm(weight: weight, eps: eps, hidden_size: hidden_size)

proc forward*(self: RmsNorm, hidden_state: Tensor): Tensor =
  ## Forward pass with float32 upcasting (aphrodite-engine style):
  ##   1. Converts to FP32 for numerical stability
  ##   2. Squaring
  ##   3. `.sqrt().reciprocal`
  ##      The single instruction rsqrt accumulate 0.03 abs error
  ##   4. Converts back to input dtype, then multiplies by BF16 weight
  ##
  ## Without float32 conversion, we accumulate errors of
  ## 1.562500e-02 = 1/64 = 2^-6 every layer
  let input_dtype = hidden_state.scalarType()
  let x = hidden_state.to(kFloat32)
  let variance = x.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(Scalar(self.eps)).rsqrt()
  let normalized = x * rstd
  normalized.to(input_dtype) * self.weight

proc forward_with_residual*(self: RmsNorm, hidden_state, residual: Tensor): (Tensor, Tensor) =
  ## Fused residual addition + RMSNorm.
  # The residual addition is done in the input dtype (BF16).
  # The RMSNorm then converts to FP32, normalizes, and converts back.
  # This matches HF's Qwen3RMSNorm behavior exactly.
  #
  # Without float32 conversion, we accumulate errors of
  # 1.562500e-02 = 1/64 = 2^-6
  # every layer
  let new_residual = hidden_state + residual
  (self.forward(new_residual), new_residual)
