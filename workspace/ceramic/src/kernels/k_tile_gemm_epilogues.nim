## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#   Tile fused gemm+epilogue kernels: D = f(A·B), one shared core
#
# ############################################################



import workspace/crucible
import ../int_tuples
import ../layouts
import ../layout_constructors
import ../layout_indexing
import ../tensors
import ../ptr_arithmetic
import ../atoms
import ../tile_algebra/tile_config
import ../tile_algebra/tile_views
import ../tile_algebra/tile_io
import ../tile_algebra/tile_ops
import ../tile_algebra/tile_mma
import ../tile_algebra/tile_epilogues
import ../tile_algebra/tile_epilogues_backend

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_config, tile_views, tile_io, tile_ops, tile_mma,
       tile_epilogues, tile_epilogues_backend

#  Bias is a per-output-column vector broadcast over the tile rows.
#  The operand view's stride-0 row carries the broadcast.
#  axpby's C operand row stride is the padded column count 32.

proc gemm_with_epilogue*[TIn, TOut; Epi](
    D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
    N, K, M: int32; epi: Epi, buf1: ptr UncheckedArray[float32]) {.device.} =
  ## D = f(A·B): the gemm core fused with the epilogue's `apply`.
  ## Expected input:
  ##   A row-major (N, K), B row-major (K, M), D row-major (N, M)
  ## The epilogue's operand views carry their own gmem buffers.
  let gl_a = makeGl(A, 0, 0, N, K)
  let gl_b = makeGlStrided(B, 0, 0, 1, M)
  let gl_d = makeGl(D, 0, 0, N, M)

  const TileDim = 32
  const tileK = 16
  var a_rtl: rt_l(TIn, TileDim, tileK)
  var b_rtr: rt_r(TIn, tileK, TileDim)
  var d_rtl: rt_l(float32, TileDim, TileDim)

  d_rtl.zero()

  let OUTPUT_Y = threadgroup_position_in_grid.y
  let OUTPUT_X = threadgroup_position_in_grid.x

  for k in 0'i32 ..< K div tileK:
    a_rtl.load(gl_a, (0, 0, OUTPUT_Y, k))
    b_rtr.load(gl_b, (0, 0, OUTPUT_X, k))
    d_rtl.mma_AB(a_rtl, b_rtr)

  var o = shard(epi, buf1, (0, 0, OUTPUT_Y, OUTPUT_X), d_rtl)
  o.apply(d_rtl, d_rtl)
  d_rtl.store(gl_d, (0, 0, OUTPUT_Y, OUTPUT_X))

proc gemm_with_epilogue*[TIn, TOut; Epi](
    D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
    N, K, M: int32; epi: Epi) {.device.} =
  ## D = f(A·B) for an epilogue with no gmem operands.
  # The no-gmem form reuses the D buffer as the epilogue's fp32
  # operand slot. Any other TOut would silently instantiate with a
  # mismatched pointer, so it is rejected here instead.
  static:
    doAssert TOut is float32,
      "gemm_with_epilogue: the no-gmem form requires an fp32 output (the D buffer doubles as the epilogue's fp32 operand buffer)"
  gemm_with_epilogue(D, A, B, N, K, M, epi, D)

proc matmul*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                        N, K, M: int32) {.device.} =
  ## D = A·B.
  gemm_with_epilogue(D, A, B, N, K, M, EpiIdentity())

proc gemm_relu*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                           N, K, M: int32) {.device.} =
  ## D = max(0, A·B).
  gemm_with_epilogue(D, A, B, N, K, M, EpiReLU())

proc linear*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                        Bias: ptr UncheckedArray[float32], N, K, M: int32) {.device.} =
  ## D = A·B + bias, a per-output-column vector broadcast over the tile rows.
  gemm_with_epilogue(D, A, B, N, K, M,
    initEpiAddBias(biasView[float32, 32, 32](Bias)),
    Bias)

proc linear_relu*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                             Bias: ptr UncheckedArray[float32], N, K, M: int32) {.device.} =
  ## D = max(0, A·B + bias).
  gemm_with_epilogue(D, A, B, N, K, M,
    initEpiLinearBiasReLU(biasView[float32, 32, 32](Bias)),
    Bias)

proc gemm_axpby*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                            C: ptr UncheckedArray[float32], Alpha, Beta: float32,
                            N, K, M: int32) {.device.} =
  ## D = α·A·B + β·C.
  gemm_with_epilogue(D, A, B, N, K, M,
    initEpiAXPBY(Alpha, Beta, cView[float32, 32, 32](C)),
    C)
