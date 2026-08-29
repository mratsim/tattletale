## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Causal conv1d, depthwise, bf16: forward + decode update
#
# ############################################################

## Depthwise causal conv1d (K = 3, groups = H, no bias) on the ceramic
## Tile API: bf16 data, fp32 accumulate. Channels are tile ROWS, time
## is tile COLUMNS. The decode variant carries the last K-1 input
## columns as state between calls.
##
## Per-element semantics:
##   - `Out[r, t] = Σ_{k=0}^{K-1} W[r, k] · X[r, t-(K-1)+k]`
##     (causal: negative column indices count as 0)
##   - fwd state-out: `StateOut[r, k] = X[r, T-(K-1)+k]` for k < K-1
##   - decode: `Out[r, 0] = Σ_k W[r, k] · Win[r, k]`
##     `StateOut[r, k] = Win[r, k+1]` for k < K-1
##
## Tile geometry:
##   - one 8-row × TileC-column tile per threadgroup, grid
##     (T div TileC, ceil(H/8)). Tile row r is channel `ty·8 + r`,
##     tile column c is time `tx·TileC + c`
##   - the K taps are lane-local shifted loads. Tap k views the base
##     pointer advanced by `t0-(K-1)+k` elements and loads at column
##     origin 0. Tile column c of tap k is then `X[r, t0-(K-1)+k+c]`
##   - the causal left pad is the first block's negative columns.
##     Those alias the previous row's tail in a row-major buffer,
##     so the first column block zeroes them after the load
##   - the per-channel weight is a row scalar, tap k contributes
##     `tap[r,c] · W[ty·8+r, k]`
##   - each bf16 product is exact in fp32, and the K products accumulate
##     in fp32
##   - rows beyond H are zero-filled on load. The row-bounded store skips them
##     (forward conv output). The column-bounded stores carry no row bound.
##     Over-H rows reach state-out (both kernels), and the decode output
##     receives them as zeros. Metal discards those writes, so partial-H batches
##     need no host padding
##
## Contract:
##   - T is a multiple of TileC and T >= TileC
##   - decode requires K >= 2 (the shifted state store)

import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ═════════════════════════════════════════════════════════════════════
#  Local device extensions
#  ═════════════════════════════════════════════════════════════════════

proc convAccumTap[R, C: static int; A: static MmaAtom](
    acc: var RtLeft[float32, R, C, A],
    tap: RtLeft[bfloat16, R, C, A],
    wgt: ptr UncheckedArray[bfloat16],
    K, k, wRow0: int32) {.device.} =
  ## `acc[r][c] += tap[r][c] · wgt[wRow0 + r, k]`, fp32 accumulate.
  ## `wRow0` is the threadgroup's first channel row (`ty·8`),
  ## because the weight buffer is indexed by global channel row,
  ## not the tile-local row. Tap and acc tiles share one MMA atom,
  ## so the fragment walk covers matching (row, column) cells in both tiles.
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  for n in 0 ..< rowTiles:
    let w = wgt[(wRow0 + int32(n * M + row)) * K + k].to(float32)
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        acc.frags[n][m].frag[v] += tap.frags[n][m].frag[v].to(float32) * w

proc zeroCols[T; R, C: static int; A: static MmaAtom](
    tile: var RtLeft[T, R, C, A],
    nCols: int32) {.device.} =
  ## Zeroes the tile's first `nCols` columns. Column-axis twin of zeroRows
  ## (tile_io_rows). The first column block's negative tap columns
  ## alias the previous row's tail, so they are zeroed after the load
  ## instead of relying on out-of-bounds reads.
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  let zeroVal =
    when T is float16: 0'u16.asFp16()
    elif T is bfloat16: toBf16(0.0'f32)
    else: {.error: "zeroCols: unsupported tile element type " & $T.}
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let c = col + m * N + v
        if int32(c) < nCols:
          tile.frags[n][m].frag[v] = zeroVal

proc storeCols[TIn, TOut; R, C: static int; A: static MmaAtom](
    gl: GlView[TOut],
    tile: RtLeft[TIn, R, C, A],
    origin: tuple,
    nCols: int32) {.device.} =
  ## Stores only the tile's first `nCols` columns, leaving the rest
  ## unwritten. Column-axis twin of storeRows (tile_io_rows).
  ## Stores narrower than the 8-column tile width:
  ## the (H, 1) decode output and the (H, K-1) state stores.
  ## No row bound. Over-H rows hold the zeros from the bounded load
  ## and store them, relying on Metal discarding out-of-bounds writes.
  ## TODO: take a `rowLimit` like storeTileRows. Removal criterion:
  ## every call site passes H and no store relies on discarded writes.
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
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let c = col + m * N + v
        if int32(c) < nCols:
          dst[row + n * M, c] = tile.frags[n][m].frag[v].to(TOut)

# ═════════════════════════════════════════════════════════════════════
#  Kernels
#  ═════════════════════════════════════════════════════════════════════

proc causal_conv1d_fwd*(
    Out: ptr UncheckedArray[bfloat16],      # (H, T) conv output
    X: ptr UncheckedArray[bfloat16],        # (H, T) pre-conv input
    StateOut: ptr UncheckedArray[bfloat16], # (H, K-1) last K-1 input columns
    W: ptr UncheckedArray[bfloat16],        # (H, K) depthwise weight
    H, T: int32,
    TileC: static int,
    K: static int) {.device.} =
  ## Chunked prefill: `Out[r, t] = Σ_k W[r, k] · X[r, t-(K-1)+k]`.
  ## `StateOut` holds the last K-1 input columns for the decode step,
  ## written only by the last column block. Grid (T div TileC, ceil(H/8)),
  ## `tx` the TileC-column block, `ty` the 8-row block.
  let tx = int32(threadgroup_position_in_grid.x)
  let ty = int32(threadgroup_position_in_grid.y)
  let t0 = tx * TileC
  let gdOut = Out.gd(shape = (-1, -1, -1, -1), stride = (8 * T, 0, T, 1))
  let gdStateOut = StateOut.gd(shape = (-1, -1, -1, -1), stride = (8 * (K - 1), 0, K - 1, 1))
  var acc: rt_l(float32, 8, TileC, getTileConfig(float32, bfloat16))
  acc.zero()
  for k in 0 ..< K:
    var tap: rt_l(bfloat16, 8, TileC)
    let gdTap = (X +% (t0 - int32(K - 1) + k)).gd(
      shape = (-1, -1, -1, -1), stride = (8 * T, 0, T, 1))
    tap.loadTileRows(gdTap, (0, 0, ty, 0), H)
    if tx == 0:
      zeroCols(tap, int32(K - 1 - k))
    convAccumTap(acc, tap, W, int32(K), int32(k), int32(ty) * 8)
  gdOut.storeTileRows(acc, (0, 0, ty, tx), H)
  if tx == T div TileC - 1:
    var st: rt_l(bfloat16, 8, TileC)
    let gdSt = (X +% int(T - int32(K - 1))).gd(
      shape = (-1, -1, -1, -1), stride = (8 * T, 0, T, 1))
    st.loadTileRows(gdSt, (0, 0, ty, 0), H)
    gdStateOut.storeCols(st, (0, 0, ty, 0), int32(K - 1))

proc causal_conv1d_update*(
    Out: ptr UncheckedArray[bfloat16],      # (H, 1) decode output
    Win: ptr UncheckedArray[bfloat16],      # (H, K) window = [state | new token]
    StateOut: ptr UncheckedArray[bfloat16], # (H, K-1) updated state = Win cols 1..K-1
    W: ptr UncheckedArray[bfloat16],        # (H, K) depthwise weight
    H: int32,
    TileC: static int,
    K: static int) {.device.} =
  ## One decode step:
  ## `Out[r, 0] = Σ_k W[r, k] · Win[r, k]`
  ## `StateOut[r, k] = Win[r, k+1]`. Grid (1, ceil(H/8)).
  ## Tap k views the base pointer advanced by k elements,
  ## so its column 0 is Win[r, k]. Columns past K-1 alias the next row's head,
  ## and the column bound never stores them.
  static: doAssert K >= 2
  let ty = int32(threadgroup_position_in_grid.y)
  let gdOut = Out.gd(shape = (-1, -1, -1, -1), stride = (8 * 1, 0, 1, 1))
  let gdStateOut = StateOut.gd(shape = (-1, -1, -1, -1), stride = (8 * (K - 1), 0, K - 1, 1))
  var acc: rt_l(float32, 8, TileC, getTileConfig(float32, bfloat16))
  acc.zero()
  var taps: array[K, rt_l(bfloat16, 8, TileC)]
  for k in 0 ..< K:
    let gdTap = (Win +% int32(k)).gd(
      shape = (-1, -1, -1, -1), stride = (8 * K, 0, K, 1))
    taps[k].loadTileRows(gdTap, (0, 0, ty, 0), H)
    convAccumTap(acc, taps[k], W, int32(K), int32(k), int32(ty) * 8)
  gdOut.storeCols(acc, (0, 0, ty, 0), 1)
  gdStateOut.storeCols(taps[1], (0, 0, ty, 0), int32(K - 1))
