## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 Quantization — Fast Walsh-Hadamard Transform
##
## 128-block FWHT for input/output incoherence processing.
## All operations via libtorch tensor ops.
##
## Reference: exl3.py forward(), hadamard_inner.cuh

import
  std/options,
  std/sequtils,
  std/math,
  workspace/libtorch as F

const
  HADAMARD_DIM* = 128
  INV_SQRT_128* = 0.088388347648'f32  # 1/sqrt(128)

# ─── Core butterfly ────────────────────────────────────────────────

proc fwht_128(x: var F.Tensor) =
  ## In-place fast Walsh-Hadamard transform, length 128 on last dim.
  ## Operates in fp32 (caller must pass fp32 tensor).
  ##
  ## 7 butterfly stages:
  ##   step=1: pairs (0,1), (2,3), ..., (126,127)
  ##   step=2: pairs (0,2), (1,3), (4,6), (5,7), ...
  ##   ...
  ##   step=64: pairs (0,64), (1,65), ..., (63,127)
  ##
  ## Each pair: out_i = a+b, out_{i+step} = a-b
  var step = 1
  while step < HADAMARD_DIM:
    let half = step
    let pairs = HADAMARD_DIM div (step * 2)
    for p in 0 ..< pairs:
      let offset = p * step * 2
      let a_view = x.narrow(-1, offset, half)
      let a = a_view.clone()
      let b = x.narrow(-1, offset + half, half)
      let sum_ab = a + b
      let diff_ab = a - b
      F.copyFrom(a_view, sum_ab)
      F.copyFrom(b, diff_ab)
    step *= 2

# ─── Block-wise transform ──────────────────────────────────────────

proc hadamard_rotate_128*(
    x: F.Tensor,
    pre_scale: Option[F.Tensor],
    post_scale: Option[F.Tensor],
    norm = INV_SQRT_128
  ): F.Tensor =
  ## Apply 128-block Walsh-Hadamard transform on the last dimension.
  ##
  ## Matches ``ext.had_r_128(input, output, pre_scale, post_scale, norm)``.
  ## CUDA kernel: ``output = FWHT(input * pre_scale) * post_scale / sqrt(128)``.
  ##
  ## Args:
  ##   x: [batch, dim] where dim is a multiple of 128.
  ##   pre_scale: optional [dim] element-wise scale, applied before FWHT.
  ##   post_scale: optional [dim] element-wise scale, applied after FWHT + norm.
  ##   norm: post-transform normalization factor (1/sqrt(128) by default).
  ##
  ## Returns:
  ##   Transformed tensor.
  result = x.clone()
  let dim = x.size(-1)
  if dim <= 0 or dim mod HADAMARD_DIM != 0:
    raise newException(ValueError,
      "[ttt] hadamard_rotate_128 expects last dimension to be a positive multiple of 128, got dim=" & $dim)

  for blk_start in countup(0, dim - 1, HADAMARD_DIM):
    var blk = result.narrow(-1, blk_start, HADAMARD_DIM)
    # Pre-scale in fp16 (matching CUDA kernel's __hmul2)
    if pre_scale.isSome:
      let s = pre_scale.unsafeGet().narrow(0, blk_start, HADAMARD_DIM)
      F.copyFrom(blk, blk * s)
    # Convert to fp32 for FWHT + norm
    var blk_f32 = blk.to(kFloat32)
    fwht_128(blk_f32)
    F.copyFrom(blk, blk_f32 * norm)
    # Post-scale in fp16 (matching CUDA kernel's __hmul2)
    if post_scale.isSome:
      let s = post_scale.unsafeGet().narrow(0, blk_start, HADAMARD_DIM)
      F.copyFrom(blk, blk * s)
