## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared RMSNorm function implementations and reporting for rounding research tests.
##
## Both test_exl3_rms_norm.nim and test_hf_rms_norm.nim import this module.

import
  std/strformat,
  workspace/libtorch as F,
  workspace/libpositron_cuda

export F

const MaxDiffTol* = 1e-4

# ── RMSNorm function implementations ──────────────────────────────────────

export pkl_rms_norm_fp16_cuda

proc rmsNormManualFP32*(x, weight: Tensor, eps: float64): Tensor =
  ## Reference: (x*x).mean().sqrt().reciprocal(), then x * rstd * weight.
  ## Weight upcast to FP32.
  let x_fp = x.to(kFloat32)
  let w_fp = weight.to(kFloat32)
  let variance = (x_fp * x_fp).mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).sqrt().reciprocal()
  let normed = (x_fp * rstd).to(x.scalarType())
  normed * w_fp

proc rmsNormOptSqrRsqrtFP32*(x, weight: Tensor, eps: float64): Tensor =
  ## x.square().mean().rsqrt(), then x * rstd * weight.
  ## Weight upcast to FP32.
  let x_fp = x.to(kFloat32)
  let w_fp = weight.to(kFloat32)
  let variance = x_fp.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).rsqrt()
  let normed = (x_fp * rstd).to(x.scalarType())
  normed * w_fp

proc rmsNormWeightFirstFP32*(x, weight: Tensor, eps: float64): Tensor =
  ## EXL3 order: (x*w)*rstd, all FP32. Weight upcast to FP32.
  let x_fp = x.to(kFloat32)
  let w_fp = weight.to(kFloat32)
  let variance = x_fp.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).rsqrt()
  ((x_fp * w_fp) * rstd).to(x.scalarType())

proc rmsNormFusedTorch*(x, weight: Tensor, eps: float64): Tensor =
  ## Fused torch::rms_norm, weight in original dtype.
  F.rms_norm(x, weight.size(0), weight, eps)

proc rmsNormFusedTorchFP32*(x, weight: Tensor, eps: float64): Tensor =
  ## Fused torch::rms_norm with FP32 upcast on both input and weight.
  let x_fp = x.to(kFloat32)
  let w_fp = weight.to(kFloat32)
  F.rms_norm(x_fp, w_fp.size(0), w_fp, eps).to(x.scalarType())

proc rmsNormWarpShuffleFP32*(x, weight: Tensor, eps: float64): Tensor =
  ## EXL3 strategy: weight-first (x*w)*rstd, per-column 4-element sums.
  ## Weight upcast to FP32.
  let x_fp = x.to(kFloat32)
  let w_fp = weight.to(kFloat32)
  let dim = x_fp.size(-1).int
  let columns = dim div 4
  let x2 = (x_fp * x_fp).reshape(x_fp.shape[0..^2] & @[columns, 4])
  let partials = x2.sum(-1)
  let total = partials.sum(-1, keepdim = true)
  let variance = total / float64(dim)
  let rstd = (variance + float64(eps)).rsqrt()
  ((x_fp * w_fp) * rstd).to(x.scalarType())

# ── HF-specific: weight NOT upcast (matches qBF16 layer exactly) ──────────

proc rmsNormHFPath*(x, weight: Tensor, eps: float64): Tensor =
  ## Exact match for HF qBF16 RMSNorm layer:
  ## (x*fp32 → square → mean → sqrt → reciprocal → *x → to input dtype) * weight(original dtype)
  let x_fp = x.to(kFloat32)
  let variance = (x_fp * x_fp).mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).sqrt().reciprocal()
  let normed = (x_fp * rstd).to(x.scalarType())
  normed * weight

proc rmsNormHFSqrRsqrt*(x, weight: Tensor, eps: float64): Tensor =
  ## HF path using .square() and .rsqrt(), weight NOT upcast.
  let x_fp = x.to(kFloat32)
  let variance = x_fp.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).rsqrt()
  let normed = (x_fp * rstd).to(x.scalarType())
  normed * weight

# ── Diff reporting ────────────────────────────────────────────────────────

proc report*(implName: string, a, b: Tensor) =
  let diff = (a.to(kFloat32) - b.to(kFloat32)).abs().max().item(float)
  let mark = if diff < 1e-6: "✅"
             elif diff < MaxDiffTol: "⚠️"
             else: "❌"
  echo &"  {mark} {implName:<32} max_diff={diff:.8f}"
