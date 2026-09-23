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

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Local device helpers ────────────────────────────────────────────
proc zeroTailRows[R, C: static int; A: static MmaAtom; T](
    tile: var RtLeft[T, R, C, A], r0, rowLimit: int32) {.device.} =
  ## Zeroes the tile's rows from the `rowLimit` boundary up. Argument `r0` is the tile's
  ## first plane row.
  const M = A.getM()
  const rowTiles = R div M
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  for n in 0 ..< rowTiles:
    if r0 + int32(n * M + row) >= rowLimit:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          when T is bfloat16:
            tile.frags[n][m].frag[v] = (0.0'f32).bfloat16
          else:
            tile.frags[n][m].frag[v] = 0'f32.to(T)

proc loadTileRowsGated[R, C: static int; A: static MmaAtom; T](
    tile: var RtLeft[T, R, C, A],
    gl: GlView[T],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded loadTile over a 2D view.
  ## - the underlying tile load reads the full (R, C) plane unconditionally
  ## - rows at or above `rowLimit` are zeroed in registers afterwards
  ## - a straddling tile does read its tail rows past the logical row count,
  ##   the view's backing storage must cover the padded tile rows,
  ##   ceil(M / TileR)·TileR
  tile.loadTile(gl, origin)
  let r0 = int32(origin[2]) * int32(R)
  if r0 + int32(R) > rowLimit:
    zeroTailRows(tile, r0, rowLimit)

proc storeTailRows[R, C: static int; A: static MmaAtom; T](
    gl: GlView[T],
    tile: RtLeft[T, R, C, A],
    origin: tuple,
    r0, rowLimit: int32) {.device.} =
  ## Row-wise store of the in-range rows of a straddling tile.
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  let o = (int(origin[0]), int(origin[1]), int(origin[2]), int(origin[3]))
  var dst = local_tile_dyn(gl, R, C, o)
  for n in 0 ..< rowTiles:
    if r0 + int32(n * M + row) < rowLimit:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          dst[row + n * M, col + m * N + v] = tile.frags[n][m].frag[v]

proc storeTileRowsGated[R, C: static int; A: static MmaAtom; T](
    gl: GlView[T],
    tile: RtLeft[T, R, C, A],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded storeTile over a 2D view.
  ## Rows at or above `rowLimit` are not written. The in-limit tile stores
  ## through the facility and a straddling tile stores row-wise.
  let r0 = int32(origin[2]) * int32(R)
  if r0 + int32(R) <= rowLimit:
    gl.storeTile(tile, origin)
  else:
    storeTailRows(gl, tile, origin, r0, rowLimit)

# ─── Inline tile procs (the fusion contract) ─────────────────────────

proc rowRstd[El; R, C: static int; A: static MmaAtom](
    y: RtLeft[El, R, C, A], eps: float32): float32 {.device.} =
  ## - Per-lane rstd = rsqrt(mean over the tile row of y² + eps).
  ## - Each lane's fragments share one tile row, the atom's lane→element mapping,
  ##   and C is the norm width, so the row reduction is one row_sum inside the tile.
  ## - eps must be > 0, an all-zero row makes the mean 0 and rsqrt(0) = +Inf,
  ##   the +Inf the store writes silently
  var y32: rt_l(float32, R, C)
  y32.widen(y)
  var sq: rt_l(float32, R, C)
  sq.mul(y32, y32)
  var sumVec: rv(float32, R, C)
  sumVec.row_sum(sq)
  result = rsqrt(sumVec.data[0] / float32(C) + eps)

proc rmsWeightElem*[El; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[El, R, C, A],
    y, w: RtLeft[El, R, C, A],
    eps: float32) {.device.} =
  ## - Epilogue first half, recorded chain's first two rounds:
  ## - `dst = bf16(w · bf16(y · rstd))`, the weighted RMSNorm output the silu stage multiplies.
  let rstd = rowRstd(y, eps)
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let normed = roundToRne[El](y.frags[n][m].frag[v].float32 * rstd)
        dst.frags[n][m].frag[v] =
          roundToRne[El](w.frags[n][m].frag[v].float32 * normed.float32)

proc siluMulElem*[El; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[El, R, C, A],
    x, gate: RtLeft[El, R, C, A]) {.device.} =
  ## - Epilogue second half, recorded chain's final round:
  ## - `dst = bf16(x · silu(g))`, the silu in f32 over the widened gated operand, no intermediate bf16 round on the silu.
  var x32: rt_l(float32, R, C)
  x32.widen(x)
  var g32: rt_l(float32, R, C)
  g32.widen(gate)
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let g = g32.frags[n][m].frag[v]
        let silu32 = g / (1.0'f32 + exp2((-g) * Log2e))
        dst.frags[n][m].frag[v] =
          roundToRne[El](x32.frags[n][m].frag[v] * silu32)

proc rmsNormGatedElem*[El; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[El, R, C, A],
    y, gate, w: RtLeft[El, R, C, A],
    eps: float32) {.device.} =
  ## - Whole epilogue in one register-tile walk:
  ## - `dst = bf16(bf16(w · bf16(y · rstd)) · silu(g))`, the fused composition
  ##   of rmsWeightElem and siluMulElem, the bf16 `weighted` intermediate
  ##   held in registers with no memory round-trip.
  let rstd = rowRstd(y, eps)
  var g32: rt_l(float32, R, C)
  g32.widen(gate)
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let normed = roundToRne[El](y.frags[n][m].frag[v].float32 * rstd)
        let weighted =
          roundToRne[El](w.frags[n][m].frag[v].float32 * normed.float32)
        let g = g32.frags[n][m].frag[v]
        let silu32 = g / (1.0'f32 + exp2((-g) * Log2e))
        dst.frags[n][m].frag[v] =
          roundToRne[El](weighted.float32 * silu32)

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
  xT.loadTileRowsGated(glX, (0, 0, rowBlk, 0), M)
  gT.loadTileRowsGated(glG, (0, 0, rowBlk, 0), M)
  wT.loadTileRowsGated(glW, (0, 0, rowBlk, 0), M)

  var oT: rt_l(El, TileR, Dv)
  rmsNormGatedElem(oT, xT, gT, wT, eps)
  glO.storeTileRowsGated(oT, (0, 0, rowBlk, 0), M)

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
  xT.loadTileRowsGated(glX, (0, 0, rowBlk, 0), M)
  wT.loadTileRowsGated(glW, (0, 0, rowBlk, 0), M)

  var mT: rt_l(El, TileR, Dv)
  rmsWeightElem(mT, xT, wT, eps)
  glM.storeTileRowsGated(mT, (0, 0, rowBlk, 0), M)

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
  mT.loadTileRowsGated(glM, (0, 0, rowBlk, 0), M)
  gT.loadTileRowsGated(glG, (0, 0, rowBlk, 0), M)

  var oT: rt_l(El, TileR, Dv)
  siluMulElem(oT, mT, gT)
  glO.storeTileRowsGated(oT, (0, 0, rowBlk, 0), M)
