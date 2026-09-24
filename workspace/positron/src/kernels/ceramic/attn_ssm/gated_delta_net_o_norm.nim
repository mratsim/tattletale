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
## - the mega kernel composes the tile core `rmsNormGatedTileCoreAt` inline,
##   WRowStride = Dv, the per-head (Hv, Dv) weight layout
## - the fused entry `rmsNormGatedTile` binds WRowStride = 0, one broadcast
##   weight row, and computes the whole chain in one launch
## - `rmsWeightElem` + `siluMulElem` splits bit-exactly at the weighted value,
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
  ## Epilogue first half, the weighted RMSNorm half of the chain.
  ## `dst = El(w · El(y · rstd))`, the value the silu stage multiplies.
  ##
  ## Contract:
  ##
  ## | parameter | shape, dtype, layout                                                                                                                                     | producer          | unit |
  ## | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---- |
  ## | dst       | (R, C) El register tile (the tile types carry R, C and the atom), produced by this proc, the El round of the weighted value per element                  | this proc         | El   |
  ## | y         | (R, C) El register tile, the norm input, the kernel's x tile                                                                                             | the caller's load | El   |
  ## | w         | (R, C) El register tile, the norm weight                                                                                                                 | host-computed     | El   |
  ## | eps       | f32, the rstd epsilon, host-computed, must be > 0 (the recorded layer's 1e-6), an all-zero row makes rsqrt(0) = +Inf, the +Inf the store writes silently | host-computed     | f32  |
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
  ## Epilogue second half, the chain's final El round.
  ## `dst = El(x · silu(g))`, the silu in f32 over the widened gated operand,
  ## no intermediate El round on the silu.
  ##
  ## Contract:
  ##
  ## | parameter | shape, dtype, layout                                                                                                                         | producer          | unit |
  ## | --------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---- |
  ## | dst       | (R, C) El register tile (the tile types carry R, C and the atom), produced by this proc, the final El round of the gated product per element | this proc         | El   |
  ## | x         | (R, C) El register tile, the weighted RMSNorm value (rmsWeightElem's output)                                                                 | the caller        | El   |
  ## | gate      | (R, C) El register tile, the silu-gated operand                                                                                              | the caller's load | El   |
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
  ## Whole epilogue in one register-tile walk.
  ## `dst = El(El(w · El(y · rstd)) · silu(g))` = rmsWeightElem fused with siluMulElem,
  ## the El `weighted` intermediate held in registers, no memory round-trip.
  ##
  ## Contract:
  ##
  ## | parameter | shape, dtype, layout                                                                                                                                     | producer          | unit |
  ## | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---- |
  ## | dst       | (R, C) El register tile (the tile types carry R, C and the atom), produced by this proc, the final El round of the gated product per element             | this proc         | El   |
  ## | y         | (R, C) El register tile, the norm input, the kernel's x tile                                                                                             | the caller's load | El   |
  ## | gate      | (R, C) El register tile, the silu-gated operand                                                                                                          | the caller's load | El   |
  ## | w         | (R, C) El register tile, the norm weight                                                                                                                 | host-computed     | El   |
  ## | eps       | f32, the rstd epsilon, host-computed, must be > 0 (the recorded layer's 1e-6), an all-zero row makes rsqrt(0) = +Inf, the +Inf the store writes silently | host-computed     | f32  |
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

proc rmsNormGatedTileCoreAt*[El](
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
  ## The megakernel composes the per-head binding inline, the fused entry
  ## `rmsNormGatedTile` binds the broadcast form.
  ##
  ## Parameters, pointers naming their dtypes, shapes bound at the call:
  ##
  ## | parameter             | shape, dtype, layout                                                                                                                | producer                        | unit     |
  ## | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | -------- |
  ## | outp                  | (M, Dv) El, row-major, the epilogue rows, produced by this proc, the El round of the gated product per element, rows >= M unwritten | this proc                       | El       |
  ## | x                     | (M, Dv) El, row-major, the norm input                                                                                               | host-computed                   | El       |
  ## | gate                  | (M, Dv) El, row-major, the silu-gated operand                                                                                       | host-computed                   | El       |
  ## | w                     | (M, Dv) with WRowStride = Dv, or (Dv) broadcast with WRowStride = 0, El, the norm weight                                            | host-computed                   | El       |
  ## | M                     | the runtime row count (the layer tensors (b, T, Hv, Dv) flatten to rows, the layout permutation host-side)                          | host-derived                    | rows     |
  ## | eps                   | f32, the rstd epsilon, host-computed, must be > 0 (the recorded layer's 1e-6)                                                       | host-computed                   | f32      |
  ## | rowBlk                | the M div TileR row-block index                                                                                                     | device-computed grid coordinate | rows     |
  ## | Dv, TileR, WRowStride | static tile geometry (norm width, the row block height, the weight view's row stride)                                               | compile-time                    | elements |
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

proc rmsNormGatedTile*[El](
    outp: ptr UncheckedArray[El],  # (M, Dv) family-dtype out
    x: ptr UncheckedArray[El],     # (M, Dv), the norm input
    gate: ptr UncheckedArray[El],  # (M, Dv), the silu-gated operand
    w: ptr UncheckedArray[El],     # (Dv), the norm weight, one row broadcast over the tile
    M: int32,
    eps: float32,
    Dv, TileR: static int) {.device.} =
  ## Grid-driven broadcast-weight form of `rmsNormGatedTileCoreAt`, WRowStride = 0.
  ## Grid (1, ceil(M div TileR)), one (TileR-row) tile per threadgroup.
  ##
  ## Returns the weighted, silu-gated, normalized rows through `outp`.
  ##
  ## | parameter | shape, dtype, layout                                                                                                                | producer      | unit     |
  ## | --------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------- | -------- |
  ## | outp      | (M, Dv) El, row-major, the epilogue rows, produced by this proc, the El round of the gated product per element, rows >= M unwritten | this proc     | El       |
  ## | x         | (M, Dv) El, row-major, the norm input                                                                                               | host-computed | El       |
  ## | gate      | (M, Dv) El, row-major, the silu-gated operand                                                                                       | host-computed | El       |
  ## | w         | (Dv) El, row-major, the norm weight, one row broadcast over the tile                                                                | host-computed | El       |
  ## | M         | the runtime row count (the layer tensors (b, T, Hv, Dv) flatten to rows, the layout permutation host-side)                          | host-derived  | rows     |
  ## | eps       | f32, the rstd epsilon, host-computed, must be > 0 (the recorded layer's 1e-6)                                                       | host-computed | f32      |
  ## | Dv, TileR | static tile geometry (norm width, the row block height)                                                                             | compile-time  | elements |
  let rowBlk = int32(threadgroup_position_in_grid.y)
  rmsNormGatedTileCoreAt[El](outp, x, gate, w, M, eps, rowBlk, Dv, TileR, 0)

proc rmsWeightTile*[El](
    midp: ptr UncheckedArray[El],  # (M, Dv) out, El(w·El(y·rstd))
    x: ptr UncheckedArray[El],     # (M, Dv), the norm input
    w: ptr UncheckedArray[El],     # (Dv), the norm weight
    M: int32,
    eps: float32,
    Dv, TileR: static int) {.device.} =
  ## Composed pair, first launch.
  ## RMSNorm + weight half, rows >= M bounded on load and store.
  ##
  ## | parameter | shape, dtype, layout                                                                                       | producer                        | unit     |
  ## | --------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------- | -------- |
  ## | midp      | (M, Dv) El, row-major, the weighted RMSNorm rows, produced by this kernel, rows >= M unwritten             | this kernel                     | El       |
  ## | x         | (M, Dv) El, row-major, the norm input                                                                      | host-computed                   | El       |
  ## | w         | (Dv) El, row-major, the norm weight, one row broadcast over the tile                                       | host-computed                   | El       |
  ## | M         | the runtime row count (the layer tensors (b, T, Hv, Dv) flatten to rows, the layout permutation host-side) | host-derived                    | rows     |
  ## | eps       | f32, the rstd epsilon, host-computed, must be > 0 (the recorded layer's 1e-6)                              | host-computed                   | f32      |
  ## | rowBlk    | the M div TileR row-block index                                                                            | device-computed grid coordinate | rows     |
  ## | Dv, TileR | static tile geometry (norm width, the row block height)                                                    | compile-time                    | elements |
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
  ## | parameter | shape, dtype, layout                                                                              | producer      | unit     |
  ## | --------- | ------------------------------------------------------------------------------------------------- | ------------- | -------- |
  ## | outp      | (M, Dv) El, row-major, the silu-gated weighted rows, produced by this kernel, rows >= M unwritten | this kernel   | El       |
  ## | mid       | (M, Dv) El, row-major, the weighted RMSNorm value (rmsWeightTile's output)                        | host-computed | El       |
  ## | gate      | (M, Dv) El, row-major, the silu-gated operand                                                     | host-computed | El       |
  ## | M         | the runtime row count, host-derived                                                               | host-derived  | rows     |
  ## | Dv, TileR | static tile geometry (norm width, the row block height)                                           | compile-time  | elements |
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
