# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  workspace/transformers/src/quantizations/datatypes {.all.}

when defined(cuda):
  import workspace/libpositron_cuda

{.experimental: "callOperator".}

type
  RmsNorm* = ref object
    weight*: Tensor
    eps*: float64
    hidden_size*: int
    quant_format*: QuantFormatKind

func init*(_: type RmsNorm, weight: Tensor, quant_format: QuantFormatKind = qBF16,
           eps: SomeFloat = 1e-6): RmsNorm =
  let hidden_size = weight.size(0)
  RmsNorm(
    weight: weight, eps: float64(eps),
    hidden_size: hidden_size,
    quant_format: quant_format,
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
  ##      On CUDA, a custom kernel using hardware rsqrt is used
  ##      to match ext.rms_norm's rounding.
  ##   4. Multiplies by weight
  ##
  ## The multiplication order differs by quantization format:
  ##   qExl3: (x*w)*rstd → cast (weight-first, all FP32, matches ext.rms_norm)
  ##   qBF16: (x*rstd).to(dtype)*w (rstd-first, matches HF Qwen3RMSNorm)
  ##
  ## The multiplication order (weight-first vs rstd-first) is the dominant
  ## factor in matching EXL3 vs HF fixtures (0.000244 vs 0.0 CPU diff).
  ## FP16/BF16 intermediates are significantly worse (0.125+ diff).
  case self.quant_format
  of qExl3:
    # We emulate warp-shuffle reduction
    # TODO: optimized kernel
    # See
    #  - tattletale/workspace/transformers/tests/rounding_rmsnorm/test_exl3_rms_norm.nim
    #  - tattletale/workspace/transformers/tests/rounding_rmsnorm/rmsnorm_common.nim
    when defined(cuda):
      if hidden_state.deviceType() == kCuda:
        return pkl_rms_norm_fp16_cuda(hidden_state, self.weight, self.eps)

    ## EXL3 order: (x*w)*rstd, all FP32. Weight upcast to FP32.
    let input_dtype = hidden_state.scalarType()
    let x = hidden_state.to(kFloat32)
    let w = self.weight.to(kFloat32)
    let variance = x.square().mean(axis = -1, keepdim = true)
    let rstd = variance.add(Scalar(self.eps)).rsqrt()
    return ((x * w) * rstd).to(input_dtype)
  of qBF16:
    # HF Qwen3RMSNorm order: (x*rstd).cast * w (w in input dtype)
    # sqrt().reciprocal() may be used instead of rsqrt() because:
    #   - On CPU, rsqrt() = sqrt().reciprocal() exactly (no HW rsqrt)
    #   - sqrt().reciprocal() is ~25% faster on CPU (bench_rmsnorm) (to be checked on GPU)
    let input_dtype = hidden_state.scalarType()
    let x = hidden_state.to(kFloat32)
    let w = self.weight.to(kFloat32)
    let variance = x.square().mean(axis = -1, keepdim = true)
    let rstd = variance.add(Scalar(self.eps)).rsqrt()
    return (x * rstd).to(input_dtype) * self.weight

proc forward_with_residual(self: RmsNorm, hidden_state, residual: Tensor): (Tensor, Tensor) =
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
