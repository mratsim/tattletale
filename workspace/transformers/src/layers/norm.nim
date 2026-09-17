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
    ## Root-mean-square layer norm with a learned per-dimension scale.
    ##   `output = (x * rsqrt(mean(x^2) + eps)) * w`
    ## The bias-one variant is RmsNormOne.
    weight*: Tensor
    eps*: float64
    hidden_size*: int
    quant_format*: QuantFormatKind

func init*(_: type RmsNorm, weight: Tensor, quant_format: QuantFormatKind = qBF16,
           eps: SomeFloat = 1e-6): RmsNorm =
  ## Build one RMSNorm from a `[width]` weight.
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
  ## The multiply order differs by quantization format:
  ##   qExl3: (x*w)*rstd → cast, weight-first, all FP32, matches ext.rms_norm
  ##   qBF16: (x*rstd).to(dtype)*w, rstd-first, matches HF Qwen3RMSNorm
  ##
  ## The multiplication order (weight-first vs rstd-first) is the dominant
  ## factor in reproducing the ext.rms_norm and HF Qwen3RMSNorm numbers
  ## (0.000244 vs 0.0 CPU diff).
  ## FP16/BF16 intermediates are significantly worse (0.125+ diff).
  case self.quant_format
  of qExl3:
    # We emulate warp-shuffle reduction
    # TODO: optimized kernel
    # See
    #  - tattletale/workspace/transformers/tests/rounding_rmsnorm/t_exl3_rms_norm.nim
    #  - tattletale/workspace/transformers/tests/rounding_rmsnorm/rmsnorm_common.nim
    when defined(cuda):
      if hidden_state.deviceType() == kCuda:
        return pkl_rms_norm_fp16_cuda(hidden_state, self.weight.to(kFloat16), self.eps)

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
    return (x * rstd).to(input_dtype) * self.weight.to(input_dtype)

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
  RmsNormOne* = ref object
    ## RMSNorm with a weight bias of one:
    ##   `output = (x * rsqrt(mean(x^2) + eps)) * (1 + w)`
    ## Computed in f32 and cast back to the input dtype.
    ##
    ## The checkpoint stores the offset `w`.
    ## A stored zero leaves the norm unscaled.
    weight*: Tensor
    eps*: float64
    hidden_size*: int
    quant_format*: QuantFormatKind

func init*(_: type RmsNormOne, weight: Tensor, quant_format: QuantFormatKind = qBF16,
           eps: SomeFloat = 1e-6): RmsNormOne =
  ## Build one bias-one RMSNorm from a `[width]` weight offset.
  let hidden_size = weight.size(0)
  RmsNormOne(
    weight: weight, eps: float64(eps),
    hidden_size: hidden_size,
    quant_format: quant_format,
  )

proc forward*(self: RmsNormOne, hidden_state: Tensor): Tensor =
  ## Bias-one RMSNorm over the last dimension, FP32 intermediate:
  ##   `output = (x * rsqrt(mean(x^2) + eps)) * (1 + w)`
  let input_dtype = hidden_state.scalarType()
  let x = hidden_state.to(kFloat32)
  let w = self.weight.to(kFloat32)
  let variance = x.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(Scalar(self.eps)).rsqrt()
  return (x * rstd * (Scalar(1.0) + w)).to(input_dtype)

proc forward_with_residual(self: RmsNormOne, hidden_state, residual: Tensor): (Tensor, Tensor) =
  ## Residual addition + bias-one RMSNorm.
  let new_residual = hidden_state + residual
  (self.forward(new_residual), new_residual)

## Call operator overloads:
## - norm(x)              → forward(x)           → Tensor
## - norm(x, residual)    → forward_with_residual(x, residual) → (Tensor, Tensor)
template `()`*(layer: RmsNormOne, x: Tensor): untyped =
  forward(layer, x)

template `()`*(layer: RmsNormOne, x, residual: Tensor): untyped =
  forward_with_residual(layer, x, residual)

type
  RmsNormGated* = ref object
    ## RMSNorm with a SiLU gate on the last dimension, the Gated DeltaNet output norm.
    ##   normed = x.f32 * rsqrt(mean(x.f32^2) + eps)
    ##   output = (w * normed.to(x.dtype) * silu(gate.f32)).to(x.dtype)
    ## The gate multiplication runs in f32 and the result is cast back to the input dtype.
    weight: Tensor
    eps: float64
    hidden_size: int

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

type
  RmsNormGatedSigmoid* = ref object
    ## RMSNorm with a sigmoid gate on the last dimension, the KDA output
    ## norm (Kimi and Ling checkpoints, o_norm key).
    ##   normed = x.f32 * rsqrt(mean(x.f32^2) + eps)
    ##   output = (w * normed.to(x.dtype) * sigmoid(gate.f32)).to(x.dtype)
    ## Same arithmetic and rounding as RmsNormGated with the sigmoid
    ## activation the KDA checkpoints carry. The gate multiplication runs
    ## in f32 and the result is cast back to the input dtype.
    weight: Tensor
    eps: float64
    hidden_size: int

## Build RmsNormGatedSigmoid from a `[head_v_dim]` weight with eps
## defaulting to 1e-6, Ling's rms_norm_eps, pass eps explicitly
## for checkpoints carrying another value (Kimi is 1e-5)
func init*(_: type RmsNormGatedSigmoid, weight: Tensor,
    eps: SomeFloat = 1e-6): RmsNormGatedSigmoid =
  let hidden_size = weight.size(0)
  RmsNormGatedSigmoid(
    weight: weight,
    eps: float64(eps),
    hidden_size: hidden_size,
  )

proc forward*(self: RmsNormGatedSigmoid, x: Tensor, gate: Tensor): Tensor =
  ## RmsNormGatedSigmoid over the last dimension of `x`, sigmoid-gated
  ## by `sigmoid(gate)`.
  ##
  ## Args:
  ##   x: (…, head_v_dim) tensor to normalize (bf16)
  ##   gate: same leading shape as `x`, last dim `head_v_dim` (bf16)
  ##
  ## Returns:
  ##   (…, head_v_dim) in x.dtype
  let input_dtype = x.scalarType()
  let x32 = x.to(kFloat32)
  let variance = x32.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(Scalar(self.eps)).rsqrt()
  let normed = (x32 * rstd).to(input_dtype)
  let weighted = self.weight * normed
  let gated = weighted * F.sigmoid(gate.to(kFloat32))
  result = gated.to(input_dtype)

template `()`*(layer: RmsNormGatedSigmoid, x, gate: Tensor): untyped =
  forward(layer, x, gate)

type
  FusedRmsNormGatedSigmoid* = ref object
    ## RMSNorm with a sigmoid gate on the last dimension, the
    ## flash-linear-attention fused kernel's single-rounding form
    ## (the o_norm checkpoints): normalization, weight multiply and
    ## gate multiply all run in f32, and the result rounds back to the
    ## input dtype once, at the end.
    ##   rstd   = 1 / sqrt(mean(x.f32^2) + eps)
    ##   output = (x.f32 * rstd * w.f32 * sigmoid(gate.f32)).to(x.dtype)
    ## Reciprocal written 1/sqrt like the fused kernel, never rsqrt.
    ## The two-rounding form: RmsNormGatedSigmoid, a distinct type.
    weight: Tensor
    eps: float64
    hidden_size: int

## Build FusedRmsNormGatedSigmoid from a `[head_dim]` weight. Pass eps
## explicitly at every construction site, the reference stack passes
## config.rms_norm_eps.
func init*(_: type FusedRmsNormGatedSigmoid, weight: Tensor,
    eps: SomeFloat): FusedRmsNormGatedSigmoid =
  let hidden_size = weight.size(0)
  FusedRmsNormGatedSigmoid(
    weight: weight,
    eps: float64(eps),
    hidden_size: hidden_size,
  )

proc forward*(self: FusedRmsNormGatedSigmoid, x: Tensor, gate: Tensor): Tensor =
  ## The flash-linear-attention fused single-rounding form over the last
  ## dimension of `x`, sigmoid-gated.
  ##
  ## Args:
  ##   x: (…, head_dim) tensor to normalize
  ##   gate: same leading shape as `x`, last dim `head_dim`
  ##
  ## Returns:
  ##   (…, head_dim) in x.dtype, one rounding at the end
  let input_dtype = x.scalarType()
  let x32 = x.to(kFloat32)
  let variance = x32.square().mean(axis = -1, keepdim = true)
  # 1/sqrt as the two-op division form, reciprocal after sqrt,
  # one rounding, never the fused rsqrt.
  let rstd = variance.add(Scalar(self.eps)).sqrt().reciprocal()
  let weighted = x32 * rstd * self.weight.to(kFloat32)
  let gated = weighted * F.sigmoid(gate.to(kFloat32))
  result = gated.to(input_dtype)

template `()`*(layer: FusedRmsNormGatedSigmoid, x, gate: Tensor): untyped =
  forward(layer, x, gate)
