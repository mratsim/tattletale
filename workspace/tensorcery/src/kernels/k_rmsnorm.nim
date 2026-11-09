# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## RMSNorm kernel over the erased tensor face, on ceramic's `k_tile_rmsnorm`.
##
## - `k_rmsnorm`: x * rsqrt(mean(x²) + eps) per axis strip, weighted or not
## - one warp per axis strip, grid-stride over the strip list
## - elements upcast to float32 inside the norm kernel, one rounding
##   at the store through the output's element type
##   (bfloat16 round-to-nearest-even)

import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic
import workspace/tensorcery/src/device_view
import workspace/ceramic/src/kernels/k_tile_rmsnorm

proc k_rmsnorm*(dst, src: distinct DeviceView, eps: float32, axis: static int) =
  ## Root-mean-square normalization over the input axis, fully overwriting dst.
  ##
  ## Expected input:
  ## - src: the source view, `axis` selects the normalized dimension
  ## - axis: the normalized dimension, a static index
  ##
  ## Output:
  ## - dst fully written with x * rsqrt(mean(x²) + eps)
  ##
  ## The launch threadgroup must be exactly one warp of threads, see `rmsnorm`.
  let
    dv = make_view(dst.data +% dst.offset, dst.layout)
    sv = make_view(src.data +% src.offset, src.layout)
  rmsnorm(dv, sv, eps, axis)

proc k_rmsnorm*(dst, src, weight: distinct DeviceView, eps: float32, axis: static int) =
  ## Root-mean-square normalization over the input axis, per-axis weight,
  ## fully overwriting dst.
  ##
  ## Expected input:
  ## - src: the source view, `axis` selects the normalized dimension
  ## - weight: rank 1, one slot per axis position, shape (axis size,)
  ## - axis: the normalized dimension, a static index
  ##
  ## Output:
  ## - dst fully written with x * rsqrt(mean(x²) + eps) * weight
  ##
  ## The launch threadgroup must be exactly one warp of threads, see `rmsnorm`.
  let
    dv = make_view(dst.data +% dst.offset, dst.layout)
    sv = make_view(src.data +% src.offset, src.layout)
    wv = make_view(weight.data +% weight.offset, weight.layout)
  rmsnorm(dv, sv, wv, eps, axis)
