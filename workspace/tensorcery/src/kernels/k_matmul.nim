# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Matrix multiply kernel over the erased tensor face, on the ceramic tile algebra.
##
## - `k_matmul`: o = a·b over the matmul shape (o: (M, N), a: (M, K), b: (K, N))
## - one squadron per (32, 32) output tile, K steps in chunks of 16
## - bfloat16 operands accumulate in float32 and round once at the store
##   (round-to-nearest-even), k_tile_gemm's f32 accumulator

import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic
import workspace/tensorcery/src/device_view
import workspace/tensorcery/src/tensor_datatypes
import workspace/ceramic/src/kernels/k_tile_gemm

proc k_matmul*(o, a, b: distinct DeviceView, mma: static TiledMma) =
  ## o = a·b over the erased tensor face.
  ##
  ## Expected input:
  ## - o: (M, N) output view, any strides
  ## - a: (M, K) operand view, any strides
  ## - b: (K, N) operand view, any strides
  ##
  ## Output:
  ## - o fully written with a·b, accumulated in float32,
  ##   one round-to-nearest-even rounding into o's element type
  ##
  ## - mma: the tile schedule, a trailing static constructed at the launch
  ##   site (the jagged_attn_fwd pattern). One squadron covers
  ##   an (MT, NT) output tile, ragged edges go through k_tile_gemm's
  ##   paddedLoad / guardedStore
  let
    ov = make_view(o.data +% o.offset, o.layout)
    av = make_view(a.data +% a.offset, a.layout)
    bv = make_view(b.data +% b.offset, b.layout)
  # k_tile_gemm consumes B in BLAS B^T form (N, K): b's strides swap over the same cells
  let bT = make_view(bv, (bv.shape[1], bv.shape[0]),
                     (bv.layout.stride[1], bv.layout.stride[0]))
  matmul(mma, ov, av, bT)
