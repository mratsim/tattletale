# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ──────────────────────  linear (the F.linear projection, GEMV and GEMM)  ──────────────────────

## Dense linear projection on the ceramic Tile API, the MSL block instantiates one entry
## per projection shape, shapes static, the contract checks and the K-walk bound resolve
## at compile time.
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
## Shape contract, a violated shape under-covers the grid or mis-tiles the K walk:
## - K a multiple of the 16-wide mma K step
## - N a multiple of TileC (64)
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

proc roundStoreElem[El; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[El, R, C, A],
    src: RtLeft[float32, R, C, A]) {.device.} =
  ## `dst[r][c] = El(src[r][c])`, one RNE round per storage write, the eager matmul's
  ## output round. The frag walk follows the loadTile lane→element mapping, the tiles
  ## agree element for element.
  ##
  ## Atom contract:
  ## - the atom A is shared by both operands
  ## - `rt_l`'s default atom is `getTileConfig(float32, T)` for every element T (tile_algebra/tiles.nim)
  ## - so the fp32 accumulator tile and the storage tile always share one
  ##   geometry class and the frag indices agree
  ##
  ## A mismatched-atom call site fails to infer A and does not compile.
  static:
    doAssert R mod A.getM() == 0 and C mod A.getN() == 0,
      "roundStoreElem: the tile geometry must cover whole atom tiles"
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        when El is bfloat16:
          dst.frags[n][m].frag[v] = src.frags[n][m].frag[v].bfloat16
        else:
          dst.frags[n][m].frag[v] = src.frags[n][m].frag[v].to(float16)

# ─── The kernel ───────────────────────────────────────────────────────

proc dense_linear_tile_fwd*[El; N, K, TileC: static int](
    Out: ptr UncheckedArray[El],  # (M, N) out, the projection rows
    X: ptr UncheckedArray[El],    # (M, K) input rows
    W: ptr UncheckedArray[El],    # (N, K) row-major weights, the F.linear layout
    M: int32,
    tx, ty: int32) {.device.} =
  ## | tile        | one (32, TileC) output tile at the caller's coordinates, grid (N div TileC, ceil(M/32))          |
  ## | ----------- | ------------------------------------------------------------------------------------------------ |
  ## | tx, ty      | the weight column block and the 32-row input block                                               |
  ## | K walk      | 16-wide mma chunks, fp32 accumulation over the El operands                                       |
  ## | output      | the fp32 accumulator rounds once (RNE) to El at the store                                        |
  ## | tail rows   | rows past M are zero-filled on load and skipped on store, partial-M batches need no host padding |
  ## | decode GEMV | the same body at M = 1                                                                           |
  ##
  ## Instantiation contract:
  ## - every static binding set of this core needs a distinct call-site line
  ## - the engine's monomorphization key erases generic static bindings, calls
  ##   that share a call-site line collapse into the first binding set's body
  static:
    doAssert TileC mod 8 == 0,
      "dense_linear_tile_fwd: TileC must be an atom-column multiple"
    doAssert K mod 16 == 0,
      "dense_linear_tile_fwd: K must be a multiple of the 16-wide K step"
    doAssert N mod TileC == 0,
      "dense_linear_tile_fwd: N must be a multiple of TileC"
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
  roundStoreElem(outT, acc)
  gdOut.storeTileRows(outT, (r0, 0, 0, tx), rem)
