# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/tensorcery/src/tensorcery

{.experimental: "callOperator".}

type
  RmsNorm* = ref object
    ## Root-mean-square layer norm with a learned per-dimension scale.
    ##   `output = (x * rsqrt(mean(x^2) + eps)) * w`
    ## The bias-one variant is RmsNormOne.
    weight: Tensor
    eps: float64
    hidden_size*: int
    # quant_format is qBF16 only: the qExl3 branch is unported.

proc init*(_: type RmsNorm, weight: Tensor, eps: SomeFloat = 1e-6): RmsNorm =
  ## Build one RMSNorm from a `[width]` weight.
  let hidden_size = weight.shape()[0]
  RmsNorm(
    weight: weight, eps: float64(eps),
    hidden_size: hidden_size,
  )

proc forward*(self: RmsNorm, hidden_state: Tensor): Tensor =
  ## RMSNorm with FP32 intermediate.
  ##
  ## Forward pass with float32 upcasting for normalization:
  ##   1. Converts to FP32 for numerical stability
  ##   2. Squares
  ##   3. `.sqrt().reciprocal`
  ##      `.square().mean().add(eps).rsqrt()` is equivalent on CPU
  ##      (no hardware rsqrt). `sqrt().reciprocal()` is faster on CPU.
  ##      On CUDA, a custom kernel using hardware rsqrt matches ext.rms_norm's rounding.
  ##   4. Multiplies by weight
  ##
  ## The multiply order differs by quantization format:
  ##   qExl3: (x*w)*rstd → cast, weight-first, all FP32, matches ext.rms_norm
  ##
  ##   qBF16: (x*rstd).to(dtype)*w, rstd-first, matches HF Qwen3RMSNorm
  ##
  ## The multiplication order (weight-first vs rstd-first) is the dominant
  ## factor in reproducing the ext.rms_norm and HF Qwen3RMSNorm numbers
  ## (0.000244 vs 0.0 CPU diff).
  ## FP16/BF16 intermediates are significantly worse (0.125+ diff).

  # qExl3 branch, out of scope (quant_format is qBF16 only):
  #   EXL3 order: (x*w)*rstd, all FP32. Weight upcast to FP32.
  #
  # HF Qwen3RMSNorm order: (x*rstd).cast * w (w in input dtype)
  # Ported-kernel rounding: one rounding at the store, the rounding boundary
  # note in ops_rmsnorm records the difference against this order.
  #
  # Rmsnorm on this base normalizes a rank-2 (tokens, width) stream.
  # RmsNorm.forward flattens the leading dimensions and restores them
  # over the output after the normalize.
  let s = hidden_state.shape()
  var tokens = 1
  for i in 0 ..< s.len - 1:
    tokens *= s[i]
  let flat = hidden_state.reshape([tokens, self.hidden_size])
  var outShape: seq[int]
  for i in 0 ..< s.len:
    outShape.add(s[i])
  rmsnorm(flat, self.weight, float32(self.eps)).reshape(outShape)

proc forward_with_residual*(self: RmsNorm, hidden_state, residual: Tensor): (Tensor, Tensor) =
  ## Fused residual addition + RMSNorm.
  let new_residual = hidden_state + residual
  (self.forward(new_residual), new_residual)

## Call operator overloads:
## - norm(x)              → forward(x)           → Tensor
## - norm(x, residual)    → forward_with_residual(x, residual) → (Tensor, Tensor)
template `()`*(layer: RmsNorm, x: Tensor): untyped =
  forward(layer, x)

template `()`*(layer: RmsNorm, x, residual: Tensor): untyped =
  forward_with_residual(layer, x, residual)
