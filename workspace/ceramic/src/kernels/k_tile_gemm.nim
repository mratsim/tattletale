## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#   Tile gemm kernels: D = f(A·B), one shared fused core
#
# ############################################################

import workspace/crucible
import ../int_tuples
import ../layouts
import ../layout_constructors
import ../layout_indexing
import ../tensors
import ../ptr_arithmetic
import ../tile_algebra
import ../tile_algebra/tile_epilogues_backend

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra, tile_epilogues_backend

# TODO: ragged support

proc gemm_with_epilogue*[TIn, TOut; Epi](
    D: ptr UncheckedArray[TOut], rsd, csd: int32,
    A: ptr UncheckedArray[TIn], rsa, csa: int32,
    B: ptr UncheckedArray[TIn], rsb, csb: int32,
    N, K, M: int32; epi: Epi, buf1: ptr UncheckedArray[float32]) {.device.} =
  ## D = f(A·B), A (N, K), B (K, M), D (N, M), explicit row/col strides.
  ##
  ## K contract (ragged-K): K is the ALLOCATED extent and must be a
  ## multiple of the tile K (16) — the loop below iterates `K div 16` full
  ## 16-slice blocks and a ragged K would silently drop the tail. For a
  ## ragged logical K, pass the padded extent Kp (multiple of 16) and
  ## ZERO-FILL the K..Kp-1 extent: the kernel reads it inside the K loop,
  ## so garbage there leaks finite wrong values into the accumulator
  ## (0xDEAD fp16 is a normal finite number, not NaN). An unpadded K fails
  ## loudly: the kernel returns before writing D, so a host-side value
  ## check sees the untouched output instead of a truncated result.
  const tileK = 16
  if K mod tileK != 0:
    return
  let gd_a = gd(A, shape = (1, 1, N, K), stride = (0, 0, rsa, csa))
  let gd_b = gd(B, shape = (1, 1, M, K), stride = (0, 0, csb, rsb))
  let gd_d = gd(D, shape = (1, 1, N, M), stride = (0, 0, rsd, csd))

  const TileDim = 32
  var a_rtl: rt_l(TIn, TileDim, tileK)
  var b_rtr: rt_r(TIn, tileK, TileDim)
  var d_rtl: rt_l(float32, TileDim, TileDim, getTileConfig(float32, TIn))

  d_rtl.zero()

  let OUTPUT_Y = threadgroup_position_in_grid.y
  let OUTPUT_X = threadgroup_position_in_grid.x

  for k in 0'i32 ..< K div tileK:
    loadTile(a_rtl, gd_a, (0, 0, OUTPUT_Y, k))
    loadTile(b_rtr, gd_b, (0, 0, OUTPUT_X, k))
    d_rtl.mma_AB(a_rtl, b_rtr)

  var o = shard(epi, buf1, (0, 0, OUTPUT_Y, OUTPUT_X), d_rtl)
  o.apply(d_rtl, d_rtl)
  storeTile(gd_d, d_rtl, (0, 0, OUTPUT_Y, OUTPUT_X))

proc gemm_with_epilogue*[TIn, TOut; Epi](
    D: ptr UncheckedArray[TOut], rsd, csd: int32,
    A: ptr UncheckedArray[TIn], rsa, csa: int32,
    B: ptr UncheckedArray[TIn], rsb, csb: int32,
    N, K, M: int32; epi: Epi) {.device.} =
  ## D = f(A·B) for an epilogue with no gmem operands.
  static:
    doAssert TOut is float32,
      "gemm_with_epilogue: the no-gmem form requires an fp32 output (the D buffer doubles as the epilogue's fp32 operand buffer)"
  gemm_with_epilogue(D, rsd, csd, A, rsa, csa, B, rsb, csb, N, K, M, epi, D)

proc matmul*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                        N, K, M: int32) {.device.} =
  ## D = A·B (row-major layouts).
  gemm_with_epilogue(D, M, 1, A, K, 1, B, M, 1, N, K, M, EpiIdentity())

proc gemm_relu*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                           N, K, M: int32) {.device.} =
  ## D = max(0, A·B) (row-major layouts).
  gemm_with_epilogue(D, M, 1, A, K, 1, B, M, 1, N, K, M, EpiReLU())

proc linear*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                        Bias: ptr UncheckedArray[float32], N, K, M: int32) {.device.} =
  ## D = A·B + bias.
  gemm_with_epilogue(D, M, 1, A, K, 1, B, M, 1, N, K, M,
    initEpiAddBias(biasView(float32, 32, 32, Bias)),
    Bias)

proc linear_relu*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                             Bias: ptr UncheckedArray[float32], N, K, M: int32) {.device.} =
  ## D = max(0, A·B + bias).
  gemm_with_epilogue(D, M, 1, A, K, 1, B, M, 1, N, K, M,
    initEpiLinearBiasReLU(biasView(float32, 32, 32, Bias)),
    Bias)

proc gemm*[TIn, TOut](D: ptr UncheckedArray[TOut],
                      M, N, K: int32, alpha: float32,
                      A: ptr UncheckedArray[TIn], rsa, csa: int32,
                      B: ptr UncheckedArray[TIn], rsb, csb: int32,
                      beta: float32,
                      C: ptr UncheckedArray[float32], rsc, csc: int32) {.device.} =
  ## D = α·A·B + β·C.
  gemm_with_epilogue(D, N, 1, A, rsa, csa, B, rsb, csb, M, K, N,
    initEpiAXPBY(alpha, beta, C, rsc, csc),
    C)
