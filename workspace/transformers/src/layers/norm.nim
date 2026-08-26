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

type
  GemmaRmsNorm* = ref object
    ## RMSNorm with the weight applied as `1 + w` (Gemma-style), used for
    ## Qwen3.5 qk-norm.
    ##
    ## Forward computes, all in f32:
    ##   `output = (x / sqrt(mean(x^2) + eps)) * (1 + w)`,
    ## then casts back to the input dtype. The weight is stored
    ## as-is (BF16 [head_dim] in the shard). The `1 + w` scaling happens in
    ## f32. This matches the vendored Qwen3_5RMSNorm
    ## (`_norm(x.float()) * (1.0 + weight.float())`, `.type_as(x)`).
    weight*: Tensor
    eps*: float64
    hidden_size*: int

## Build GemmaRMSNorm from a `[head_dim]` weight (applied as `1 + w`). eps defaults to 1e-6.
func init*(_: type GemmaRmsNorm, weight: Tensor, eps: SomeFloat = 1e-6): GemmaRmsNorm =
  let hidden_size = weight.size(0)
  GemmaRmsNorm(
    weight: weight,
    eps: float64(eps),
    hidden_size: hidden_size,
  )

proc forward*(self: GemmaRmsNorm, x: Tensor): Tensor =
  ## GemmaRMSNorm over the last dimension, f32 math, cast back to x.dtype.
  let input_dtype = x.scalarType()
  let x32 = x.to(kFloat32)
  let variance = x32.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(Scalar(self.eps)).rsqrt()
  let normed = x32 * rstd
  let w32 = self.weight.to(kFloat32)
  result = (normed * (1.0 + w32)).to(input_dtype)

template `()`*(layer: GemmaRmsNorm, x: Tensor): untyped =
  forward(layer, x)

type
  RmsNormGated* = ref object
    ## RMSNorm with a SiLU-gated multiplier over the last dimension,
    ## the Gated DeltaNet output norm (vendored `Qwen3_5RMSNormGated`).
    ##
    ## The weight is applied as a REGULAR multiply (not `1 + w`): the shard
    ## stores the norm weight directly (F32 [128], mean near 1). The gate
    ## tensor carries the z projection reshaped to the normed shape.
    ##
    ## Forward, matching the vendored op order:
    ##   normed = x.f32 * rsqrt(mean(x.f32^2) + eps)     (f32)
    ##   normed = normed.to(x.dtype)                      (bf16)
    ##   gated  = weight * normed                         (f32 weight × bf16 → f32)
    ##   gated  = gated * silu(gate.f32)                  (f32)
    ##   output = gated.to(x.dtype)                       (bf16)
    weight*: Tensor
    eps*: float64
    hidden_size*: int

## Build RmsNormGated from a `[head_v_dim]` weight. eps defaults to 1e-6.
func init*(_: type RmsNormGated, weight: Tensor, eps: SomeFloat = 1e-6): RmsNormGated =
  let hidden_size = weight.size(0)
  RmsNormGated(
    weight: weight,
    eps: float64(eps),
    hidden_size: hidden_size,
  )

proc forward*(self: RmsNormGated, x: Tensor, gate: Tensor): Tensor =
  ## RmsNormGated over the last dimension of `x`, gated by `silu(gate)`.
  ##
  ## Args:
  ##   x: (…, hidden_size) tensor to normalize (bf16)
  ##   gate: same leading shape as `x`, last dim `hidden_size` (bf16)
  ##
  ## Returns:
  ##   (…, hidden_size) in x.dtype
  let input_dtype = x.scalarType()
  let x32 = x.to(kFloat32)
  let variance = x32.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(Scalar(self.eps)).rsqrt()
  let normed = (x32 * rstd).to(input_dtype)
  let weighted = self.weight * normed
  let gated = weighted * F.silu(gate.to(kFloat32))
  result = gated.to(input_dtype)

template `()`*(layer: RmsNormGated, x, gate: Tensor): untyped =
  forward(layer, x, gate)
