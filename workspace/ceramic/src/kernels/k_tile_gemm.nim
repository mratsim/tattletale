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
import ../tile_algebra/tile_io_bounded

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra, tile_epilogues_backend, tile_io_bounded

proc gemm_with_epilogue*[TIn, TOut; Epi](
    D: ptr UncheckedArray[TOut], rsd, csd: int32,
    A: ptr UncheckedArray[TIn], rsa, csa: int32,
    B: ptr UncheckedArray[TIn], rsb, csb: int32,
    N, K, M: int32; epi: Epi, buf1: ptr UncheckedArray[float32]) {.device.} =
  ## D = f(A·B), A (N, K), B (K, M), D (N, M), explicit row/col strides.
  ##
  ## Expected input:
  ##   - A of shape (N, K) at (rsa, csa) strides and B of shape (K, M)
  ##     presented as an (M, K) view at (csb, rsb) strides
  ##   - D of shape (N, M) at (rsd, csd) strides, raw runtime dims,
  ##     no caller padding
  ##   - epi, a fused epilogue, buf1 the fp32 buffer its gmem operands
  ##     shard-read from (the D buffer for an in-place epilogue)
  ##
  ## Output:
  ##   D tile = f(AB tile), written masked at the real (N, M) extent.
  ##
  ## Ragged-native:
  ##   - the k loop runs ceil(K / tileK) slices
  ##   - the A and B tile loads are bounded by the raw dims, out-of-range
  ##     lanes hold the zero fill in-register and never touch memory
  ##   - the final store is masked at the real extent, so D's padding
  ##     lanes are untouched
  ##
  ## Operand contract:
  ##   - an epilogue's gmem operands are shard-read over the full 32×32 tile extent
  ##   - on a region-edge tile the out-of-range lanes read past the operand's real region
  ##   - those lanes' values are dropped by the masked store, but the operand buffer must tolerate reads at the tile extent
  ##
  ## Ragged shapes with gmem-operand epilogues go through the bounded procs
  ## below (gemm, linear, linear_relu), which load their operands into
  ## bounded register tiles.
  let gd_a = gd(A, shape = (1, 1, N, K), stride = (0, 0, rsa, csa))
  let gd_b = gd(B, shape = (1, 1, M, K), stride = (0, 0, csb, rsb))
  let gd_d = gd(D, shape = (1, 1, N, M), stride = (0, 0, rsd, csd))

  const TileDim = 32
  const tileK = 16
  var a_rtl: rt_l(TIn, TileDim, tileK)
  var b_rtr: rt_r(TIn, tileK, TileDim)
  var d_rtl: rt_l(float32, TileDim, TileDim, getTileConfig(float32, TIn))

  d_rtl.zero()

  let OUTPUT_Y = threadgroup_position_in_grid.y
  let OUTPUT_X = threadgroup_position_in_grid.x
  let validM = min(N - int32(OUTPUT_Y) * int32(TileDim), int32(TileDim))
  let validN = min(M - int32(OUTPUT_X) * int32(TileDim), int32(TileDim))

  let kTiles = (K + int32(tileK) - 1) div int32(tileK)
  for k in 0'i32 ..< kTiles:
    loadTileBounded(a_rtl, gd_a, (0, 0, OUTPUT_Y, k), N, K)
    loadTileBounded(b_rtr, gd_b, (0, 0, OUTPUT_X, k), M, K)
    d_rtl.mma_AB(a_rtl, b_rtr)

  var o = shard(epi, buf1, (0, 0, OUTPUT_Y, OUTPUT_X), d_rtl)
  o.apply(d_rtl, d_rtl)
  storeTileMasked(gd_d, d_rtl, (0, 0, OUTPUT_Y, OUTPUT_X), validM, validN)

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

proc gemm_with_bias_epilogue*[TIn, TOut; Epi](
    D: ptr UncheckedArray[TOut],
    A, B: ptr UncheckedArray[TIn],
    Bias: ptr UncheckedArray[float32],
    N, K, M: int32; epi: Epi) {.device.} =
  ## D = f(A·B + bias), one 32×32 output tile per threadgroup, row-major.
  ##
  ## Expected input:
  ##   - A of shape (N, K), B of shape (K, M) and D of shape (N, M)
  ##     at row-major strides (K, 1), (1, M) and (M, 1), raw runtime dims,
  ##     no caller padding
  ##   - Bias, one fp32 per output column
  ##   - epi, a bias epilogue, EpiAddBias or EpiLinearBiasReLU
  ##
  ## Output:
  ##   D tile = f(AB tile + bias), written masked at row stride M.
  ##
  ## Bias reads:
  ##   - the bias loads through loadTileBounded over a row-stride-0 view
  ##   - a lane reads Bias[col] only with its (row, col) inside the real region
  ##   - out-of-range columns hold zeros in-register and never touch memory
  ##     past the column count, and the masked store drops those lanes anyway
  const TileDim = 32
  const tileK = 16
  let gd_a = gd(A, shape = (1, 1, N, K), stride = (0, 0, K, 1))
  let gd_b = gd(B, shape = (1, 1, M, K), stride = (0, 0, 1, M))
  let gd_d = gd(D, shape = (1, 1, N, M), stride = (0, 0, M, 1))
  let gd_bias = gd(Bias, shape = (1, 1, N, M), stride = (0, 0, 0, 1))

  var a_rtl: rt_l(TIn, TileDim, tileK)
  var b_rtr: rt_r(TIn, tileK, TileDim)
  var d_rtl: rt_l(float32, TileDim, TileDim, getTileConfig(float32, TIn))
  d_rtl.zero()

  let OUTPUT_Y = threadgroup_position_in_grid.y
  let OUTPUT_X = threadgroup_position_in_grid.x
  let validM = min(N - int32(OUTPUT_Y) * int32(TileDim), int32(TileDim))
  let validN = min(M - int32(OUTPUT_X) * int32(TileDim), int32(TileDim))

  let kTiles = (K + int32(tileK) - 1) div int32(tileK)
  for k in 0'i32 ..< kTiles:
    loadTileBounded(a_rtl, gd_a, (0, 0, OUTPUT_Y, k), N, K)
    loadTileBounded(b_rtr, gd_b, (0, 0, OUTPUT_X, k), M, K)
    d_rtl.mma_AB(a_rtl, b_rtr)

  var bias_rtl: rt_l(float32, TileDim, TileDim, getTileConfig(float32, TIn))
  loadTileBounded(bias_rtl, gd_bias, (0, 0, OUTPUT_Y, OUTPUT_X), N, M)

  # The bias shard of `epi` is a type-compatibility placeholder, the bounded
  # register tile feeds the apply, so bias_gmem is never dereferenced.
  var o = shard(epi, Bias, (0, 0, OUTPUT_Y, OUTPUT_X), d_rtl)
  o.apply(d_rtl, d_rtl, bias_rtl)
  storeTileMasked(gd_d, d_rtl, (0, 0, OUTPUT_Y, OUTPUT_X), validM, validN)

proc linear*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                        Bias: ptr UncheckedArray[float32], N, K, M: int32) {.device.} =
  ## D = A·B + bias, one fp32 bias lane per output column, the
  ## gemm_with_bias_epilogue ragged path.
  gemm_with_bias_epilogue(D, A, B, Bias, N, K, M,
    initEpiAddBias(biasView(float32, 32, 32, Bias)))

proc linear_relu*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                             Bias: ptr UncheckedArray[float32], N, K, M: int32) {.device.} =
  ## D = max(0, A·B + bias), one fp32 bias lane per output column, the
  ## gemm_with_bias_epilogue ragged path with ReLU.
  gemm_with_bias_epilogue(D, A, B, Bias, N, K, M,
    initEpiLinearBiasReLU(biasView(float32, 32, 32, Bias)))

proc gemm*[TIn, TOut](D: ptr UncheckedArray[TOut],
                      M, N, K: int32, alpha: float32,
                      A: ptr UncheckedArray[TIn], rsa, csa: int32,
                      B: ptr UncheckedArray[TIn], rsb, csb: int32,
                      beta: float32,
                      C: ptr UncheckedArray[float32], rsc, csc: int32) {.device.} =
  ## D = α·A·B + β·C, A (M, K), B (K, N), C/D (M, N), the BLIS strided
  ## surface. D is row-major at row stride N.
  ##
  ## Ragged-native contract:
  ##   - raw runtime dims, no caller padding or staging
  ##   - the A, B and C tile loads are bounded by the raw dims,
  ##     out-of-range lanes hold zeros in-register and never touch memory
  ##   - the k loop runs ceil(K / tileK) slices and the final D store is
  ##     masked at the real M×N extent, β = 0 skips the C load
  const TileDim = 32
  const tileK = 16
  let gd_a = gd(A, shape = (1, 1, M, K), stride = (0, 0, rsa, csa))
  let gd_b = gd(B, shape = (1, 1, N, K), stride = (0, 0, csb, rsb))
  let gd_c = gd(C, shape = (1, 1, M, N), stride = (0, 0, rsc, csc))
  let gd_d = gd(D, shape = (1, 1, M, N), stride = (0, 0, N, 1))

  var a_rtl: rt_l(TIn, TileDim, tileK)
  var b_rtr: rt_r(TIn, tileK, TileDim)
  var d_rtl: rt_l(float32, TileDim, TileDim, getTileConfig(float32, TIn))
  d_rtl.zero()

  let OUTPUT_Y = threadgroup_position_in_grid.y
  let OUTPUT_X = threadgroup_position_in_grid.x
  let validM = min(M - int32(OUTPUT_Y) * int32(TileDim), int32(TileDim))
  let validN = min(N - int32(OUTPUT_X) * int32(TileDim), int32(TileDim))

  let kTiles = (K + int32(tileK) - 1) div int32(tileK)
  for k in 0'i32 ..< kTiles:
    loadTileBounded(a_rtl, gd_a, (0, 0, OUTPUT_Y, k), M, K)
    loadTileBounded(b_rtr, gd_b, (0, 0, OUTPUT_X, k), N, K)
    d_rtl.mma_AB(a_rtl, b_rtr)

  var c_rtl: rt_l(float32, TileDim, TileDim, getTileConfig(float32, TIn))
  if beta != 0.0'f32:
    loadTileBounded(c_rtl, gd_c, (0, 0, OUTPUT_Y, OUTPUT_X), M, N)

  var o = shard(initEpiAXPBY(alpha, beta, C, rsc, csc), C, (0, 0, OUTPUT_Y, OUTPUT_X), d_rtl)
  o.apply(d_rtl, d_rtl, c_rtl)
  storeTileMasked(gd_d, d_rtl, (0, 0, OUTPUT_Y, OUTPUT_X), validM, validN)
