## EXL3 Quantization — Fast Walsh-Hadamard Transform
##
## 128-block FWHT for input/output incoherence processing.
## All operations via libtorch tensor ops.
##
## Reference: exl3.py forward(), hadamard_inner.cuh

import
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

proc hadamard_rotate_128*(x: F.Tensor, scale: F.Tensor = nil,
                norm: float32 = 1.0'f32,
                pre_scale: bool = true): F.Tensor =
  ## Apply 128-block Walsh-Hadamard transform on the last dimension.
  ##
  ## Matches ``ext.had_r_128(input, output, pre_scale, post_scale, norm)``.
  ## CUDA kernel: ``output = FWHT(input * pre_scale) * post_scale / sqrt(128)``.
  ##
  ## Args:
  ##   x: [batch, dim] where dim is a multiple of 128.
  ##   scale: [dim] optional element-wise scale (pre_scale or post_scale).
  ##   norm: post-transform normalization factor.
  ##   pre_scale: If true, apply scale before FWHT; if false, after.
  ##
  ## Returns:
  ##   Transformed tensor.
  result = x.clone()
  let dim = x.size(-1)

  for blk_start in countup(0, dim - 1, HADAMARD_DIM):
    var blk = result.narrow(-1, blk_start, HADAMARD_DIM)
    # Apply pre-scale in fp16 (matching CUDA kernel: __hmul2 in half precision)
    if not scale.isNil and pre_scale:
      let s = scale.narrow(0, blk_start, HADAMARD_DIM)
      F.copyFrom(blk, blk * s)
    # Convert to fp32 for FWHT + norm
    var blk_f32 = blk.to(kFloat32)
    fwht_128(blk_f32)
    F.copyFrom(blk, blk_f32 * norm)
    # Apply post-scale in fp16 (matching CUDA kernel)
    if not scale.isNil and not pre_scale:
      let s = scale.narrow(0, blk_start, HADAMARD_DIM)
      F.copyFrom(blk, blk * s)
