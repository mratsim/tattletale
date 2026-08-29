## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile GEMM kernels: D = f(A·B) on 32×32 output tiles, fp32 accumulate.
##
## Contract: A (M, K), B (K, N), D (M, N) at explicit row/col strides.
## M, N and K are raw runtime dims, no caller padding.
## Edge tiles load bounded and store masked. Memory past the real region
## is never touched.
##
## Dataflow per output tile:
##
##   A (M, K) ── bounded load ──┐
##                              ├─ mma_AB (k-slices of 16) ── AB tile
##   B (K, N) ── bounded load ──┘
##
##   AB tile ───────────────────┐
##                              ├─ apply ── D tile ── finalStore ── D (M, N)
##   C (M, N) ── bounded load ──┘

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
  ## D = f(A·B), one 32×32 output tile per threadgroup.
  ##
  ## Expected input:
  ##   - A (N, K) at (rsa, csa), B (K, M) at (rsb, csb),
  ##     D (N, M) at (rsd, csd): element strides
  ##   - N, K, M: raw runtime dims. Edge tiles straddle the real region.
  ##   - epi: epilogue with its gmem operands and storeMask
  ##   - buf1: gmem buffer for the shard's TensorView operands
  ##     (C for EpiAXPBY, Bias for the bias epilogues)
  ##
  ## Output: D tile = f(AB tile), written masked at (rsd, csd).
  ## Bounded loads zero-fill out-of-range lanes in-register.
  ## The mma stays tile-shaped. Memory past the real region is never touched.
  const TileDim = 32
  const tileK = 16
  let gd_a = gd(A, shape = (1, 1, N, K), stride = (0, 0, rsa, csa))
  let gd_b = gd(B, shape = (1, 1, M, K), stride = (0, 0, csb, rsb))
  let gd_d = gd(D, shape = (1, 1, N, M), stride = (0, 0, rsd, csd))

  var a_rtl: rt_l(TIn, TileDim, tileK)
  var b_rtr: rt_r(TIn, tileK, TileDim)
  var d_rtl: rt_l(float32, TileDim, TileDim, getTileConfig(float32, TIn))

  d_rtl.zero()

  let OUTPUT_Y = threadgroup_position_in_grid.y
  let OUTPUT_X = threadgroup_position_in_grid.x
  let m0 = int32(OUTPUT_Y) * int32(TileDim)
  let n0 = int32(OUTPUT_X) * int32(TileDim)
  let validM = min(N - m0, int32(TileDim))
  let validN = min(M - n0, int32(TileDim))

  let kTiles = (K + int32(tileK) - 1) div int32(tileK)
  for k in 0'i32 ..< kTiles:
    loadTileBounded(a_rtl, gd_a, (0, 0, OUTPUT_Y, k), N, K)
    loadTileBounded(b_rtr, gd_b, (0, 0, OUTPUT_X, k), M, K)
    d_rtl.mma_AB(a_rtl, b_rtr)

  var o = shard(epi, buf1, (0, 0, OUTPUT_Y, OUTPUT_X), d_rtl)
  o.storeMask = tileStoreMask(d_rtl, validM, validN)
  o.apply(d_rtl, d_rtl)
  o.finalStore(gd_d, d_rtl, (0, 0, int(OUTPUT_Y), int(OUTPUT_X)))

proc gemm_with_epilogue*[TIn, TOut; Epi](
    D: ptr UncheckedArray[TOut], rsd, csd: int32,
    A: ptr UncheckedArray[TIn], rsa, csa: int32,
    B: ptr UncheckedArray[TIn], rsb, csb: int32,
    N, K, M: int32; epi: Epi) {.device.} =
  ## D = f(A·B) for an epilogue with no gmem operands.
  ## The full form is called with the output buffer as `buf1`.
  static:
    doAssert TOut is float32,
      "gemm_with_epilogue: the no-gmem form requires an fp32 output " &
      "(the epilogue's operand buffer is fp32)"
  gemm_with_epilogue(D, rsd, csd, A, rsa, csa, B, rsb, csb, N, K, M, epi, D)

proc matmul*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                        N, K, M: int32) {.device.} =
  ## D = A·B (row-major layouts). The identity epilogue has no gmem
  ## operands.
  gemm_with_epilogue(D, M, 1, A, K, 1, B, M, 1, N, K, M, EpiIdentity())

proc gemm_relu*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                           N, K, M: int32) {.device.} =
  ## D = max(0, A·B) (row-major layouts). The ReLU epilogue has no gmem
  ## operands.
  gemm_with_epilogue(D, M, 1, A, K, 1, B, M, 1, N, K, M, EpiReLU())

proc linear*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                        Bias: ptr UncheckedArray[float32], N, K, M: int32) {.device.} =
  ## D = A·B + bias.
  ## The bias is a 32-wide column broadcast, always in-bounds.
  ## The full form receives the bias buffer as `buf1`.
  gemm_with_epilogue(D, M, 1, A, K, 1, B, M, 1, N, K, M,
    initEpiAddBias(biasView(float32, 32, 32, Bias)),
    Bias)

proc linear_relu*[TIn, TOut](D: ptr UncheckedArray[TOut], A, B: ptr UncheckedArray[TIn],
                             Bias: ptr UncheckedArray[float32], N, K, M: int32) {.device.} =
  ## D = max(0, A·B + bias).
  ## The bias is a 32-wide column broadcast, always in-bounds.
  ## The full form receives the bias buffer as `buf1`.
  gemm_with_epilogue(D, M, 1, A, K, 1, B, M, 1, N, K, M,
    initEpiLinearBiasReLU(biasView(float32, 32, 32, Bias)),
    Bias)

proc gemm*[TIn, TOut](D: ptr UncheckedArray[TOut],
                      rsd, csd: int32,
                      M, N, K: int32, alpha: float32,
                      A: ptr UncheckedArray[TIn], rsa, csa: int32,
                      B: ptr UncheckedArray[TIn], rsb, csb: int32,
                      beta: float32,
                      C: ptr UncheckedArray[float32], rsc, csc: int32) {.device.} =
  ## D = α·A·B + β·C, A (M, K), B (K, N), C/D (M, N), runtime strides.
  ##
  ## C loads bounded into a register tile. Out-of-range lanes hold zeros
  ## in-register, never touching C memory.
  ## The plain EpiAXPBY apply folds α·AB + β·C in-register.
  ## β = 0 skips the C load.
  ## finalStore writes D masked. Out-of-range lanes of D stay untouched.
  const TileDim = 32
  const tileK = 16
  let gd_a = gd(A, shape = (1, 1, M, K), stride = (0, 0, rsa, csa))
  let gd_b = gd(B, shape = (1, 1, N, K), stride = (0, 0, csb, rsb))
  let gd_c = gd(C, shape = (1, 1, M, N), stride = (0, 0, rsc, csc))
  let gd_d = gd(D, shape = (1, 1, M, N), stride = (0, 0, rsd, csd))

  var a_rtl: rt_l(TIn, TileDim, tileK)
  var b_rtr: rt_r(TIn, tileK, TileDim)
  var d_rtl: rt_l(float32, TileDim, TileDim, getTileConfig(float32, TIn))
  d_rtl.zero()

  let OUTPUT_Y = threadgroup_position_in_grid.y
  let OUTPUT_X = threadgroup_position_in_grid.x
  let m0 = int32(OUTPUT_Y) * int32(TileDim)
  let n0 = int32(OUTPUT_X) * int32(TileDim)
  let validM = min(M - m0, int32(TileDim))
  let validN = min(N - n0, int32(TileDim))

  let kTiles = (K + int32(tileK) - 1) div int32(tileK)
  for k in 0'i32 ..< kTiles:
    loadTileBounded(a_rtl, gd_a, (0, 0, OUTPUT_Y, k), M, K)
    loadTileBounded(b_rtr, gd_b, (0, 0, OUTPUT_X, k), N, K)
    d_rtl.mma_AB(a_rtl, b_rtr)

  var c_rtl: rt_l(float32, TileDim, TileDim, getTileConfig(float32, TIn))
  if beta != 0'f32:
    loadTileBounded(c_rtl, gd_c, (0, 0, OUTPUT_Y, OUTPUT_X), M, N)

  # The C view is a type-compat placeholder.
  # The in-register apply below never reads C_gmem (C comes via c_rtl).
  # Strides and shape are the tile's, not the real C's.
  var o = initEpiAXPBY(alpha, beta, cView(float32, TileDim, TileDim, C))
  o.storeMask = tileStoreMask(d_rtl, validM, validN)
  o.apply(d_rtl, d_rtl, c_rtl)
  o.finalStore(gd_d, d_rtl, (0, 0, int(OUTPUT_Y), int(OUTPUT_X)))
