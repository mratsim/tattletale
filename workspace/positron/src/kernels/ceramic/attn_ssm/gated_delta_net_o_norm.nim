# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ──────────────────────  gated_delta_net_o_norm (SiLU-gated RMSNorm, the GDN output norm)  ──────────────────────

## SiLU-gated RMSNorm on the ceramic Tile API, the Gated DeltaNet output norm.
##
##   out = El(El(w · El(x · rstd)) · silu(g))    silu(g) = g / (1 + exp2(−g·log2e))
##   chain:  x → x·rstd → El → ·w → El → ·silu(g) → El
##   El, the family dtype (bf16 or fp16), one round-to-nearest-even at each El step
##
## | contract       | value                                                                                                          |
## | -------------- | -------------------------------------------------------------------------------------------------------------- |
## | tensors        | x, gate, out (M, Dv) family dtype, row-major; w (Dv); eps runtime f32, must be > 0 (the recorded layer's 1e-6) |
## | M              | runtime arg, the layer tensors (b, T, Hv, Dv) flatten to rows, the layout permutation stays host-side          |
## | rstd           | rsqrt(mean(x²) + eps) over the row                                                                             |
## | tail rows      | rows >= M store zero-skipped, the load reads padded rows, backing storage covers ceil(M / TileR)·TileR         |
## | Dv             | static (128), equal to the tile width, one row_sum spans the tile                                              |
## | geometry       | grid (1, ceil(M div TileR)) at 32 lanes, one TileR-row x Dv-col tile per threadgroup                           |
## | rounding chain | normed, weighted and output each round to the family dtype, every multiply's operands stay f32 in between      |
## | silu form      | f32 over the widened gated operand, the same 1-ulp-class exponential form as silu_and_mul                      |
##
## Fusion contract (the inline-tile property):
## - {.device.} tile procs `rowRstd`, `rmsWeightElem`, `siluMulElem`, `rmsNormGatedElem`
##   inline into any kernel that keeps the epilogue tiles in threadgroup registers
## - the mega kernel composes the tile core `rmsNormGatedTileAt` inline, the fused
##   entry `rmsNormGatedTile` computing the whole chain in one launch
## - the per-head variant `rmsNormGatedTilePerHeadAt` serves (Hv, Dv) per-head weight
##   layouts. `rmsWeightElem` + `siluMulElem` splits bit-exactly at the weighted value,
##   so the f32 and family-dtype round-trip is exact

import ../math_consts
import workspace/crucible
import workspace/ceramic
import ../tile_widen
import ../tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Inline tile procs (the fusion contract) ─────────────────────────

proc rowRstd[El; R, C: static int; A: static MmaAtom](
    y: RtLeft[El, R, C, A], eps: float32): float32 {.device.} =
  ## - Per-lane rstd = rsqrt(mean over the tile row of y² + eps).
  ## - Each lane's fragments share one tile row, the atom's lane→element mapping,
  ##   and C is the norm width, so the row reduction is one row_sum inside the tile.
  ## - eps must be > 0, an all-zero row makes the mean 0 and rsqrt(0) = +Inf,
  ##   the +Inf the store writes silently
  static:
    doAssert R == A.getM(),
      "rowRstd: one scalar rstd per lane requires TileR == atom M"
  var y32: rt_l(float32, R, C)
  y32.widen(y)
  var sq: rt_l(float32, R, C)
  sq.mul(y32, y32)
  var sumVec: rv(float32, R, C)
  sumVec.row_sum(sq)
  result = rsqrt(sumVec.rowScalar() / float32(C) + eps)

proc rmsWeightElem*[El; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[El, R, C, A],
    y, w: RtLeft[El, R, C, A],
    eps: float32) {.device.} =
  ## - Epilogue first half, recorded chain's first two rounds:
  ## - `dst = bf16(w · bf16(y · rstd))`, the weighted RMSNorm output the silu stage multiplies.
  let rstd = rowRstd(y, eps)
  var y32: rt_l(float32, R, C, A)
  y32.widen(y)
  var w32: rt_l(float32, R, C, A)
  w32.widen(w)
  var normed: rt_l(float32, R, C, A)
  normed.map(y32, roundToNearestEven[El](x * rstd).float32)
  dst.map2(w32, normed, roundToNearestEven[El](x * y))

proc siluMulElem*[El; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[El, R, C, A],
    x, gate: RtLeft[El, R, C, A]) {.device.} =
  ## - Epilogue second half, recorded chain's final round:
  ## - `dst = bf16(x · silu(g))`, the silu in f32 over the widened gated operand, no intermediate bf16 round on the silu.
  var x32: rt_l(float32, R, C, A)
  x32.widen(x)
  var g32: rt_l(float32, R, C, A)
  g32.widen(gate)
  var silu32: rt_l(float32, R, C, A)
  silu32.map(g32, x / (1.0'f32 + exp2((-x) * Log2e)))
  dst.map2(x32, silu32, roundToNearestEven[El](x * y))

proc rmsNormGatedElem*[El; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[El, R, C, A],
    y, gate, w: RtLeft[El, R, C, A],
    eps: float32) {.device.} =
  ## - Whole epilogue in one register-tile walk:
  ## - `dst = bf16(bf16(w · bf16(y · rstd)) · silu(g))`, the fused composition
  ##   of rmsWeightElem and siluMulElem, the bf16 `weighted` intermediate
  ##   held in registers with no memory round-trip.
  let rstd = rowRstd(y, eps)
  var g32: rt_l(float32, R, C, A)
  g32.widen(gate)
  var y32: rt_l(float32, R, C, A)
  y32.widen(y)
  var w32: rt_l(float32, R, C, A)
  w32.widen(w)
  var normed: rt_l(float32, R, C, A)
  normed.map(y32, roundToNearestEven[El](x * rstd).float32)
  var weighted: rt_l(float32, R, C, A)
  weighted.map2(w32, normed, roundToNearestEven[El](x * y).float32)
  var silu32: rt_l(float32, R, C, A)
  silu32.map(g32, x / (1.0'f32 + exp2((-x) * Log2e)))
  dst.map2(weighted, silu32, roundToNearestEven[El](x * y))

# ─── Core tile procs (inline-tile property) ──────────────────────────

proc rmsNormGatedTileCoreAt[El](
    outp: ptr UncheckedArray[El],
    x: ptr UncheckedArray[El],
    gate: ptr UncheckedArray[El],
    w: ptr UncheckedArray[El],
    M: int32,
    eps: float32,
    rowBlk: int32,
    Dv, TileR, WRowStride: static int) {.device.} =
  ## Shared epilogue tile walk, parameterized over the weight view's row stride.
  ##
  ##   WRowStride = 0  ─────  one (Dv) weight row broadcast over the tile
  ##   WRowStride = Dv ─────  one weight row per tile row
  ##
  ## Public wrappers below fix the binding. The generic never needs a direct call site.
  ##
  ## Tile contract:
  ##   - One (TileR-row) tile of epilogue rows at the caller's row block, the full Dv-wide
  ##     row in the tile.
  ##   - Rows at or above M load zero-filled and stay unwritten on store.
  ##
  ## Instantiation contract:
  ##   - Each static binding set of this core needs its own call-site line.
  ##   - The engine's monomorphization key erases generic static bindings, so calls sharing
  ##     one call-site line all collapse into the first binding set's body.
  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dv, 1))
  let glG = gate.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dv, 1))
  let glW = w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, WRowStride, 1))
  let glO = outp.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dv, 1))

  var xT: rt_l(El, TileR, Dv)
  var gT: rt_l(El, TileR, Dv)
  var wT: rt_l(El, TileR, Dv)
  xT.loadTileRowsZeroPadded(glX, (0, 0, rowBlk, 0), M)
  gT.loadTileRowsZeroPadded(glG, (0, 0, rowBlk, 0), M)
  wT.loadTileRowsZeroPadded(glW, (0, 0, rowBlk, 0), M)

  var oT: rt_l(El, TileR, Dv)
  rmsNormGatedElem(oT, xT, gT, wT, eps)
  glO.storeTileRows(oT, (0, 0, rowBlk, 0), M)

proc rmsNormGatedTileAt*[El](
    outp: ptr UncheckedArray[El],  # (M, Dv) family-dtype out
    x: ptr UncheckedArray[El],     # (M, Dv), the norm input
    gate: ptr UncheckedArray[El],  # (M, Dv), the silu-gated operand
    w: ptr UncheckedArray[El],     # (Dv), the norm weight, one row broadcast over the tile
    M: int32,
    eps: float32,
    rowBlk: int32,
    Dv, TileR: static int) {.device.} =
  ## Tile core, one (Dv) weight row broadcast over the tile's rows. The megakernel composes this core inline. `rmsNormGatedTile`
  ## is the grid-driven wrapper.
  ##
  ## Returns the weighted, silu-gated, normalized rows through `outp`.
  ##
  ## Example, at TileR = 8: `rmsNormGatedTileAt(outp, x, gate, w, 32, eps, rowBlk, 128, 8)`
  ## computes rows `rowBlk·8 ..< rowBlk·8 + 8` of a 32-row epilogue, every row
  ## weighted by the same `w[0 ..< 128]`.
  rmsNormGatedTileCoreAt[El](outp, x, gate, w, M, eps, rowBlk, Dv, TileR, 0)

proc rmsNormGatedTilePerHeadAt*[El](
    outp: ptr UncheckedArray[El],  # (M, Dv) family-dtype out
    x: ptr UncheckedArray[El],     # (M, Dv), the norm input
    gate: ptr UncheckedArray[El],  # (M, Dv), the silu-gated operand
    w: ptr UncheckedArray[El],     # (M, Dv), one weight row per output row
    M: int32,
    eps: float32,
    rowBlk: int32,
    Dv, TileR: static int) {.device.} =
  ## Tile core, one (Dv) weight row per output row. Serves the (M, Dv) per-head weight layout of a per-head output norm.
  ## Row `rowBlk·TileR + r` weights with `w[(rowBlk·TileR + r)·Dv ..< (rowBlk·TileR + r + 1)·Dv]`.
  ##
  ## Returns the weighted, silu-gated, normalized rows through `outp`.
  ##
  ## Example, at TileR = 8: `rmsNormGatedTilePerHeadAt(outp, x, gate, w, 32, eps, rowBlk, 128, 8)`
  ## computes rows `rowBlk·8 ..< rowBlk·8 + 8` of a 32-row epilogue, each row
  ## weighted by its own head's weight row.
  rmsNormGatedTileCoreAt[El](outp, x, gate, w, M, eps, rowBlk, Dv, TileR, Dv)

proc rmsNormGatedTile*[El](
    outp: ptr UncheckedArray[El],  # (M, Dv) family-dtype out
    x: ptr UncheckedArray[El],     # (M, Dv), the norm input
    gate: ptr UncheckedArray[El],  # (M, Dv), the silu-gated operand
    w: ptr UncheckedArray[El],     # (Dv), the norm weight, one row broadcast over the tile
    M: int32,
    eps: float32,
    Dv, TileR: static int) {.device.} =
  ##  Grid-driven form of `rmsNormGatedTileAt`. Grid (1, ceil(M div tileR)),
  ## one (TileR-row) tile per threadgroup.
  let rowBlk = int32(threadgroup_position_in_grid.y)
  rmsNormGatedTileAt(outp, x, gate, w, M, eps, rowBlk, Dv, TileR)

proc rmsWeightTile*[El](
    midp: ptr UncheckedArray[El],  # (M, Dv) out, El(w·El(y·rstd))
    x: ptr UncheckedArray[El],     # (M, Dv), the norm input
    w: ptr UncheckedArray[El],     # (Dv), the norm weight
    M: int32,
    eps: float32,
    Dv, TileR: static int) {.device.} =
  ## Composed pair, first launch.
  ## RMSNorm + weight half, rows >= M bounded on load and store.
  let rowBlk = int32(threadgroup_position_in_grid.y)
  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dv, 1))
  let glW = w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glM = midp.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dv, 1))

  var xT: rt_l(El, TileR, Dv)
  var wT: rt_l(El, TileR, Dv)
  xT.loadTileRowsZeroPadded(glX, (0, 0, rowBlk, 0), M)
  wT.loadTileRowsZeroPadded(glW, (0, 0, rowBlk, 0), M)

  var mT: rt_l(El, TileR, Dv)
  rmsWeightElem(mT, xT, wT, eps)
  glM.storeTileRows(mT, (0, 0, rowBlk, 0), M)

proc siluMulTile*[El](
    outp: ptr UncheckedArray[El],  # (M, Dv) family-dtype out
    mid: ptr UncheckedArray[El],   # (M, Dv), the weighted input
    gate: ptr UncheckedArray[El],  # (M, Dv), the silu-gated operand
    M: int32,
    Dv, TileR: static int) {.device.} =
  ## Composed pair, second launch.
  ## Multiplies the weighted RMSNorm by the f32 silu of the second operand.
  ## Rows >= M bound both the load and the store.
  ##
  ## Returns the silu-gated weighted rows through `outp`.
  let rowBlk = int32(threadgroup_position_in_grid.y)
  let glM = mid.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dv, 1))
  let glG = gate.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dv, 1))
  let glO = outp.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dv, 1))

  var mT: rt_l(El, TileR, Dv)
  var gT: rt_l(El, TileR, Dv)
  mT.loadTileRowsZeroPadded(glM, (0, 0, rowBlk, 0), M)
  gT.loadTileRowsZeroPadded(glG, (0, 0, rowBlk, 0), M)

  var oT: rt_l(El, TileR, Dv)
  siluMulElem(oT, mT, gT)
  glO.storeTileRows(oT, (0, 0, rowBlk, 0), M)
