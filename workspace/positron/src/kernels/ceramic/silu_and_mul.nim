## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Fused SiLU × Mul (silu_and_mul): Tile API port
#
# ############################################################

## Fused GatedMLP activation on the ceramic Tile API:
## `Out[r][c] = silu(X[r][c]) · X[r][N + c]` over one fp16 input
## buffer X of shape (M, 2N) holding the gate in cols [0, N) and the
## up in cols [N, 2N), one fp16 output (M, N). The FlashInfer
## `silu_and_mul` convention.
## Experimental: not a production kernel, known gaps below, not fixed.
##
## Per-element semantics:
##   - silu computed in fp32 from the fp16 gate (exact promotion) as
##     `g / (1 + exp2(−g·log2e))` with log2e = 1.4426950408889634'f32
##     and the backend's native exp2 (1-ulp class, not a bit-exact
##     Taylor form)
##   - the silu result rounds to fp16, then the optional `actLimit`
##     clamp applies in fp32 arithmetic (skipped when 0). The final
##     fp16×fp16 multiply performs one rounding (fp32 product of the
##     fp16-rounded operands, one RNE round)
##
## Tile geometry:
##   - one 8-row × TileC-col tile per threadgroup, grid
##     (N div TileC, ceil(M/8)), the atom's 32 lanes
##   - TileC: the static tile width (launchers instantiate 64) and
##     must divide N
##   - one gd view over X with row stride 2N carries both operands:
##     the gate tile loads at X col origin tx, the up tile at col
##     origin tx + N div TileC (the view's element (r, c) is
##     X[r·2N + c], so the col origin N + tx·TileC selects the up half)
##   - the store writes the out tile at col origin tx into a (M, N)
##     row-major view
##   - rows beyond M are zero-filled on load and skipped on store, so
##     the batch needs no host padding
##
## Known production gaps (documented, not fixed):
##   - N must be a multiple of TileC. A partial last column block is
##     out of contract
##   - the shared tile_algebra has no silu, elementwise tile add or
##     elementwise tile div, so the fusion walks the register frags in
##     a module-local device proc instead of tile-op maps
##   - concrete fp16 in / fp16 out only, no fp32 path

import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ═════════════════════════════════════════════════════════════════════
#  Local device extensions: the tile API gaps the fusion needs
#  ═════════════════════════════════════════════════════════════════════

proc siluAndMulElem*[R, C: static int; A: static MmaAtom](
    dst: var RtLeft[float16, R, C, A],
    gate, up: RtLeft[float16, R, C, A],
    actLimit: float32) {.device.} =
  ## Per-element fused silu·mul over one register tile: `dst[r][c] =
  ## silu(gate[r][c]) · up[r][c]` per the module doc's semantics. The
  ## frag walk follows the loadTile lane→element mapping, so gate, up
  ## and dst agree elementwise.
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let g32 = gate.frags[n][m].frag[v].to(float32)
        let u32 = up.frags[n][m].frag[v].to(float32)
        var s = g32 / (1.0'f32 + exp2(-g32 * 1.4426950408889634'f32))
        var u = u32
        if actLimit != 0.0'f32:
          s = min(s, actLimit)
          u = min(max(u, -actLimit), actLimit)
        let s16 = s.to(float16)
        let u16 = u.to(float16)
        dst.frags[n][m].frag[v] =
          (s16.to(float32) * u16.to(float32)).to(float16)

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc silu_and_mul_fwd*(
    Out: ptr UncheckedArray[float16],    # (M, N): the silu(gate)·up output
    X: ptr UncheckedArray[float16],      # (M, 2N): gate cols [0, N), up cols [N, 2N)
    M, N: int32,
    actLimit: float32,
    TileC: static int) {.device.} =
  ## Computes one 8×TileC silu-mul tile per threadgroup: grid
  ## (N div TileC, ceil(M/8)), `tx` the TileC-col block, `ty` the
  ## 8-row block. The gate tile loads at X col origin tx, the up tile
  ## at col origin tx + N div TileC, the store writes the out tile at
  ## col origin tx (see the module doc). Rows ≥ M are zero-filled on
  ## load and skipped on store, so partial-M batches need no host
  ## padding. `TileC` must divide N.
  let tx = int32(threadgroup_position_in_grid.x)
  let ty = int32(threadgroup_position_in_grid.y)
  let glX = gd(X, shape = (-1, -1, -1, -1), stride = (8 * 2 * N, 0, 2 * N, 1))
  let glOut = gd(Out, shape = (-1, -1, -1, -1), stride = (8 * N, 0, N, 1))
  var gate: rt_l(float16, 8, TileC)
  var up: rt_l(float16, 8, TileC)
  var outT: rt_l(float16, 8, TileC)
  loadTileRows(gate, glX, (0, 0, ty, tx), M)
  loadTileRows(up, glX, (0, 0, ty, tx + N div TileC), M)
  siluAndMulElem(outT, gate, up, actLimit)
  storeTileRows(glOut, outT, (0, 0, ty, tx), M)
