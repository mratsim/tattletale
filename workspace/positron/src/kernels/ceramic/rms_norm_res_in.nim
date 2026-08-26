## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Fused residual-add RMSNorm (rms_norm_res_in): Tile API port
#
# ############################################################

## Fused residual-add RMSNorm on the ceramic Tile API:
## `Out[r][c] = rms_norm(x + res_in)[r][c]` over fp16 (M, C) input and
## residual buffers and an fp16 (M, C) output, the residual add inside
## the kernel (one launch, no host pre-add).
## Experimental: not a production kernel, known gaps below, not fixed.
##
## Per-element semantics:
##   - the sum s = x + res rounds to fp16 (RNE) and re-promotes to
##     fp32, the exllamav3 RES_IN `add_resid_in` write shape: the
##     value the norm consumes and the residual stream would carry
##   - the norm sequence is the standalone `rms_norm` arithmetic:
##     s², row_sum, ·1/C, +ε, rsqrt, mul_row, γ mul, one fp16 rounding
##     at the store
##   - with a zero residual the output is bit-identical to the
##     standalone kernel on the same input
##
## Tile geometry:
##   - one 8-row × TileC-col tile per threadgroup, grid (1, ceil(M/8)),
##     the atom's 32 lanes, TileC the static tile width (128)
##   - each threadgroup loops the C div TileC column blocks twice:
##     pass 1 accumulates the row sums of s² over all blocks, pass 2
##     normalizes and stores
##   - the views carry the rmsnorm swapped strides (row axis = the
##     contiguous hidden dim, stride 1 for X/R/Out, col stride C = the
##     token row), so the origin's row component holds the column-block
##     index and its col component the row-block index
##   - γ broadcasts through a (0, 0, 1, 0) view
##   - rows ≥ M are zero-filled on load and skipped on store, so
##     partial-M batches need no host padding
##
## Known production gaps (documented, not fixed):
##   - C must be a multiple of TileC. A partial last column block is
##     out of contract
##   - the updated residual stream is not written. Production packs or
##     multi-buffers it with the output (the engine reads back only
##     binding 0)
##   - concrete fp16 in / fp16 out only, no fp32 path

import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ═════════════════════════════════════════════════════════════════════
#  Local device extensions: the tile API gaps the fusion needs
#  ═════════════════════════════════════════════════════════════════════

proc addResidF16[R, C: static int; A: static MmaAtom](
    dst: var RtRight[float32, R, C, A],
    x, res: RtRight[float32, R, C, A]) {.device.} =
  ## Per-element residual add over one register tile: `dst[r][c]` = the
  ## fp16-rounded sum (x[r][c] + res[r][c]) re-promoted to fp32, the
  ## exllamav3 RES_IN `add_resid_in` write shape (the fp32 add of the
  ## fp16-exact operands is exact, one RNE fp16 round). The frag walk
  ## follows the loadTile lane→element mapping, so x, res and dst agree
  ## elementwise.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for m in 0 ..< colTiles:
    for n in 0 ..< rowTiles:
      for v in 0 ..< vpt:
        let s = x.frags[m][n].frag[v] + res.frags[m][n].frag[v]
        dst.frags[m][n].frag[v] = s.to(float16).to(float32)

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc rms_norm_res_in_fwd*(
    Out: ptr UncheckedArray[float16],    # (M, C) fp16: the normed output
    X: ptr UncheckedArray[float16],      # (M, C) fp16: the input rows
    R: ptr UncheckedArray[float16],      # (M, C) fp16: the residual stream before the add
    G: ptr UncheckedArray[float16],      # (C,) fp16: the norm weight
    M, C: int32,
    eps: float32,
    TileC: static int) {.device.} =
  ## Computes one fused residual-norm over the 8-row tile block
  ## `threadgroup_position_in_grid.y`: the module doc's two-pass
  ## sequence, `C div TileC` column blocks per pass. `TileC` is the
  ## static tile width (128) and must divide C. Rows ≥ M are
  ## zero-filled on load and skipped on store. The grid is
  ## (1, ceil(M/8)) with a 32-lane threadgroup per tile.
  let gid = int32(threadgroup_position_in_grid.y)
  let colBlocks = C div int32(TileC)
  let glX = gd(X, shape = (-1, -1, -1, -1), stride = (8 * C, 0, 1, C))
  let glR = gd(R, shape = (-1, -1, -1, -1), stride = (8 * C, 0, 1, C))
  let glG = gd(G, shape = (-1, -1, -1, -1), stride = (0, 0, 1, 0))
  let glOut = gd(Out, shape = (-1, -1, -1, -1), stride = (8 * C, 0, 1, C))
  var x_rtr: rt_r(float32, 8, TileC)
  var res_rtr: rt_r(float32, 8, TileC)
  var s_rtr: rt_r(float32, 8, TileC)
  var sq_rtr: rt_r(float32, 8, TileC)
  var gamma_rtr: rt_r(float32, 8, TileC)
  var ss: rv(float32, 8, TileC)
  ss.zero()
  for cb in 0'i32 ..< colBlocks:
    loadTileRows(x_rtr, glX, (0, 0, cb, gid), M)
    loadTileRows(res_rtr, glR, (0, 0, cb, gid), M)
    addResidF16(s_rtr, x_rtr, res_rtr)
    sq_rtr.mul(s_rtr, s_rtr)
    ss.row_sum(sq_rtr, ss)
  let invC = 1.0'f32 / float32(C)
  ss.mul(ss, invC)
  ss.add(ss, eps)
  ss.rsqrt(ss)
  for cb in 0'i32 ..< colBlocks:
    loadTileRows(x_rtr, glX, (0, 0, cb, gid), M)
    loadTileRows(res_rtr, glR, (0, 0, cb, gid), M)
    addResidF16(s_rtr, x_rtr, res_rtr)
    s_rtr.mul_row(s_rtr, ss)
    loadTile(gamma_rtr, glG, (0, 0, cb, 0))
    s_rtr.mul(s_rtr, gamma_rtr)
    storeTileRows(glOut, s_rtr, (0, 0, cb, gid), M)
