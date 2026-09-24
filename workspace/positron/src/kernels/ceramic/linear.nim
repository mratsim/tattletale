# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ──────────────────────  linear (the F.linear projection, GEMV and GEMM)  ──────────────────────

## Dense linear projection on the ceramic Tile API.
##
## Regime:
## - one device entry per element dtype
## - the projection shape (N, K, M) travels as runtime int32 arguments
## - one body per (El, TileC) instantiation serves every projection shape
##
## | contract     | value                                                                                                                                                |
## | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
## | projection   | Out[m][n] = sum_k X[m][k] * W[n][k], the torch F.linear contract over the row-major (N, K) weight layout the checkpoints store                       |
## | serves       | one body for every dense projection of a model, the qkv/o projections, the MoE router, the shared-expert projections and lm_head                     |
## | decode GEMV  | M = 1, grid.y = 1, the weight columns parallelize over grid.x                                                                                        |
## | prefill GEMM | grid.y tiles M in 32-row A tiles, the B tiles reload per (M-tile, N-tile) threadgroup                                                                |
## | rounding     | fp32 mma accumulation over 16-wide K chunks, one RNE round to the storage element at the output store, that store is the eager matmul's output round |
## | referee      | torch eager bf16 (fp32 opmath, its own accumulation order), parity inside the 2-bf16-ulp-at-max band, not bit equality                               |

##
##   per threadgroup:  X rows (32) × W cols (TileC) → 16-wide mma chunks → fp32 acc → RNE → El store
##
## Shape contract, the launcher computes the grid from these, a violated shape
## under-covers the grid or mis-tiles the K walk:
## - K a multiple of the 16-wide mma K step
## - N a multiple of TileC (64, the tile config's default tile width)
## - M unconstrained, rows past M load zero-filled and are never stored, a tail
##   M-tile needs no host padding
##
## Known production gaps (documented, not fixed):
## - no B-tile reuse across grid.y, each (M-tile, N-tile) threadgroup re-walks
##   its weight column block, decode-shaped M re-reads the weights once per launch,
##   prefill reads them once per M-tile
## - no K-split for very wide K, the 16-wide chunk walk stays serial within a threadgroup
##
import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Row-bounded tile io ─────────────────────────────────────────────
# tile_io_rows' loadTileRows/storeTileRows cover every 16-bit element:
# the load guards each element before the access, rows past `rowLimit`
# are zero-filled and never read.
# ─── Local device extensions ─────────────────────────────────────────

# ─── The kernel ───────────────────────────────────────────────────────

proc dense_linear_tile_core*[El; TileC: static int](
    Out: ptr UncheckedArray[El], X: ptr UncheckedArray[El],
    W: ptr UncheckedArray[El], N, K, M: int32, tx, ty: int32) {.device.} =
  ## Out[m][n] = sum_k X[m][k] * W[n][k] over the row-major (N, K) weights,
  ## the torch F.linear contract, one (32, TileC) output tile per threadgroup.
  ##
  ## Expected input:
  ##   - Out of shape (M, N), X of shape (M, K), W of shape (N, K) row-major,
  ##     the F.linear weight layout the checkpoints store, all El element size
  ##   - N, K, M runtime int32, the projection shape
  ##   - tx the weight column block, ty the 32-row input block, grid
  ##     (N div TileC, ceil(M/32)) at 32 lanes
  ##
  ## per threadgroup, X rows (32) × W cols (TileC) → 16-wide mma chunks → fp32 acc → RNE → El store
  ##
  ## Output:
  ##   - Out rows [ty*32, ty*32 + rem), rem = min(M - ty*32, 32), each element
  ##     one RNE round of the fp32 mma accumulator to El
  ##   - rows past M zero-fill on load and never store, a tail M-tile needs
  ##     no host padding, the decode GEMV is the same body at M = 1
  ##
  ## Caller contract, the launcher computes the grid from these, a violated
  ## shape under-covers the grid or mis-tiles the K walk:
  ##   - K a multiple of the 16-wide mma K step
  ##   - N a multiple of TileC
  ##
  ## Instantiation regime:
  ##   - El infers from the pointer arguments
  ##   - the projection shape is runtime, one body per (El, TileC)
  ##     instantiation serves every projection shape
  static:
    doAssert TileC mod 8 == 0,
      "dense_linear_tile_core: TileC must be an atom-column multiple"
  let r0 = ty * 32
  let rem = min(M - r0, 32'i32)
  let gdX = X.gd(shape = (-1, -1, -1, -1), stride = (K, 0, K, 1))
  let gdW = W.gd(shape = (-1, -1, -1, -1), stride = (1, 0, K, 1))
  let gdOut = Out.gd(shape = (-1, -1, -1, -1), stride = (N, 0, N, 1))
  var acc: rt_l(float32, 32, TileC, getTileConfig(float32, El))
  var outT: rt_l(El, 32, TileC)
  var a: rt_l(El, 32, 16)
  var b: rt_r(El, 16, TileC)
  acc.zero()
  for kk in 0'i32 ..< K div 16:
    a.loadTileRows(gdX, (r0, 0, 0, kk), rem)
    b.loadTile(gdW, (0, 0, tx, kk))
    acc.mma_AB(a, b)
  outT.map(acc, roundToNearestEven[El](x))
  gdOut.storeTileRows(outT, (r0, 0, 0, tx), rem)

proc dense_linear_tile_fwd*[El](
    Out: ptr UncheckedArray[El], X: ptr UncheckedArray[El],
    W: ptr UncheckedArray[El], N, K, M: int32, tx, ty: int32) {.device.} =
  ## F.linear at the default tile width.
  ##
  ## Parameters, pointers naming their dtypes, shapes bound at the call, grid
  ## coordinates arriving from the grid:
  ##
  ## | parameter | shape, dtype, layout                                                                                                                     | producer                         | unit     |
  ## | --------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | -------- |
  ## | Out       | (M, N) El, row-major, the projection output, produced by the kernel, one round-to-nearest-even per element from the fp32 mma accumulator | this kernel                      | El       |
  ## | X         | (M, K) El, row-major, the projection input                                                                                               | host-computed                    | El       |
  ## | W         | (N, K) El, row-major, the F.linear weight layout the checkpoints store                                                                   | host-computed                    | El       |
  ## | N, K, M   | the runtime projection shape (N output columns, K reduce length, M input rows)                                                           | host-derived                     | elements |
  ## | tx, ty    | the weight column block (N div TileC grid) and the 32-row input block (ceil(M/32) grid)                                                  | device-computed grid coordinates | elements |
  ## Contract:
  ##   - TileC = 64, the 8 atom columns of the tile config's atoms
  ##   - El infers from the pointer arguments
  ##   - the projection shape (N, K, M) is runtime
  dense_linear_tile_core[El, 64](Out, X, W, N, K, M, tx, ty)

proc dense_linear_tile32_fwd*[El](
    Out: ptr UncheckedArray[El], X: ptr UncheckedArray[El],
    W: ptr UncheckedArray[El], N, K, M: int32, tx, ty: int32) {.device.} =
  ## F.linear at the narrow tile width.
  ##
  ## Parameters, pointers naming their dtypes, shapes bound at the call, grid
  ## coordinates arriving from the grid:
  ##
  ## | parameter | shape, dtype, layout                                                                                                                     | producer                         | unit     |
  ## | --------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | -------- |
  ## | Out       | (M, N) El, row-major, the projection output, produced by the kernel, one round-to-nearest-even per element from the fp32 mma accumulator | this kernel                      | El       |
  ## | X         | (M, K) El, row-major, the projection input                                                                                               | host-computed                    | El       |
  ## | W         | (N, K) El, row-major, the F.linear weight layout the checkpoints store                                                                   | host-computed                    | El       |
  ## | N, K, M   | the runtime projection shape (N output columns, K reduce length, M input rows)                                                           | host-derived                     | elements |
  ## | tx, ty    | the weight column block (N div TileC grid) and the 32-row input block (ceil(M/32) grid)                                                  | device-computed grid coordinates | elements |
  ## Contract:
  ##   - TileC = 32, the stage-4 a/b decay and beta GEMV binding of the mega composition,
  ##     its projection rows N = 32 sit below the default width
  ##   - El infers from the pointer arguments
  ##   - the projection shape (N, K, M) is runtime
  dense_linear_tile_core[El, 32](Out, X, W, N, K, M, tx, ty)
