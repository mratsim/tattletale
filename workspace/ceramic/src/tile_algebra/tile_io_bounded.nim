## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Bounded tile I/O:
##   the ragged edge of a register tile against the real region.
##
## A register tile straddles the region edge when its plane origin plus extent
## crosses the raw runtime (row, col) extents. The per-lane contract:
##
## | lane          | loadTileBounded | tileStoreMask | storeTileMasked           |
## | ------------- | --------------- | ------------- | ------------------------- |
## | inside limits | reads gmem      | bit set       | writes                    |
## | outside       | zero fill       | bit clear     | leaves the cell untouched |
##
## Out-of-range lanes never touch memory. A tile fully inside both limits
## takes the plain `loadTile`/`storeTile` path with no guard cost, so
## grid-aligned shapes keep the aligned path bit-for-bit.

import workspace/crucible
import ../int_tuples
import ../layout_indexing
import ../tensors
import ./tiles
import ./tile_config
import ./tile_io

# ═════════════════════════════════════════════════════════════════════════
#  The zero fill value (fp16-safe)
# ═════════════════════════════════════════════════════════════════════════

func zeroTileValue*(T: typedesc): T =
  ## Zero fill for an out-of-range lane.
  ## fp16 has no `T(0)` constructor (the distinct-type gap), its zero is the bit pattern `0'u16.asFp16()`.
  when T is float16: 0'u16.asFp16()
  elif T is bfloat16: 0'u16.asBf16()
  else: T(0)

# ═════════════════════════════════════════════════════════════════════════
#  Loads
# ═════════════════════════════════════════════════════════════════════════

proc loadTileBounded*[TIn; TOut; R, C: static int; A: static MmaAtom](
    tile: var RtLeft[TOut, R, C, A],
    gl: GlView[TIn],
    origin: tuple,
    limitRows, limitCols: int32) =
  ## Bounded load for a LayoutLeft tile.
  ##
  ## A lane reads its element only when both of its plane coordinates sit
  ## inside the real region, out-of-range lanes hold the zero fill, and no
  ## lane is ever read past the caller's allocation.
  ##
  ## Expected input:
  ##   - tile, an unowned (R, C) register tile, loaded lane by lane
  ##   - gl, the (rows, cols) global-memory view, the tile's plane at plane
  ##     origin `origin[2]·R` (rows) and `origin[3]·C` (cols)
  ##   - limitRows and limitCols, the raw extents along the view's dims
  ##
  ## Output:
  ##   tile.frags holds the loaded values for the in-range lanes and `T(0)`
  ##   everywhere else, in `loadTile`'s lane→cell mapping and order.
  ##
  ## A tile fully inside both limits loads through `loadTile` (no guard).
  let r0 = int32(origin[2]) * int32(R)
  let c0 = int32(origin[3]) * int32(C)
  if r0 + int32(R) <= limitRows and c0 + int32(C) <= limitCols:
    loadTile(tile, gl, origin)
  else:
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
    let src = local_tile_dyn(gl, R, C, o)
    for n in 0 ..< rowTiles:
      for m in 0 ..< colTiles:
        for vptI in 0 ..< vpt:
          if r0 + int32(n * M + row) < limitRows and
             c0 + int32(m * N + col + vptI) < limitCols:
            tile.frags[n][m].frag[vptI] = src[row + n * M, col + m * N + vptI].to(TOut)
          else:
            tile.frags[n][m].frag[vptI] = zeroTileValue(TOut)

proc loadTileBounded*[TIn; TOut; R, C: static int; A: static MmaAtom](
    tile: var RtRight[TOut, R, C, A],
    gl: GlView[TIn],
    origin: tuple,
    limitRows, limitCols: int32) =
  ## Bounded load for a LayoutRight (swapped-operand) tile.
  ##
  ## Same contract as the RtLeft overload, over the RtRight lane→cell
  ## mapping (col-tile index m outer, row-tile index n inner) and plane
  ## origin `origin[2]·C` (rows), `origin[3]·R` (cols).
  let r0 = int32(origin[2]) * int32(C)
  let c0 = int32(origin[3]) * int32(R)
  if r0 + int32(C) <= limitRows and c0 + int32(R) <= limitCols:
    loadTile(tile, gl, origin)
  else:
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
    let src = local_tile_dyn(gl, C, R, o)
    for m in 0 ..< colTiles:
      for n in 0 ..< rowTiles:
        for vptI in 0 ..< vpt:
          if r0 + int32(col + m * N + vptI) < limitRows and
             c0 + int32(row + n * M) < limitCols:
            tile.frags[m][n].frag[vptI] = src[col + m * N + vptI, row + n * M].to(TOut)
          else:
            tile.frags[m][n].frag[vptI] = zeroTileValue(TOut)

# ═════════════════════════════════════════════════════════════════════════
#  Stores
# ═════════════════════════════════════════════════════════════════════════
#
#  Predication contract (keep in lockstep):
#    tileStoreMask sets bit
#  ((n·colTiles + m)·vpt + v) exactly when storeTileMasked's guard
#  `row + n·M < validM and col + m·N + v < validN` holds for the same lane cell, in the same iteration
#  order. A drift between the bitmask and the guard silently mis-predicates stores.

proc tileStoreMask*[T; R, C: static int; A: static MmaAtom](
    tile: RtLeft[T, R, C, A],
    validM, validN: int32): int =
  ## Per-lane store predication for a LayoutLeft tile's valid (M, N) range,
  ## measured from the tile's plane origin.
  ##
  ## Contract:
  ##   - bit ((n·colTiles + m)·vpt + v) is set when the lane's cell
  ##     (row + n·M, col + m·N + v) is inside the valid range
  ##   - the bit order matches `storeTile`/`storeTileMasked`'s iteration order
  ##   - the `tile` value carries the (R, C, atom) geometry as type params
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  result = 0
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt:
        if row + n * M < int(validM) and col + m * N + vptI < int(validN):
          result = result or (1 shl ((n * colTiles + m) * vpt + vptI))

proc storeTileMasked*[TIn; TOut; R, C: static int; A: static MmaAtom](
    gl: GlView[TOut],
    tile: RtLeft[TIn, R, C, A],
    origin: tuple,
    validM, validN: int32) =
  ## Masked store for a LayoutLeft tile, writes only the tile's cells inside
  ## (validM, validN), the valid counts measured from the tile's plane origin.
  ##
  ## Contract:
  ##   - out-of-range cells leave the destination untouched, the destination buffer may be larger than the real region and keep the padding
  ##   - lane→cell mapping and iteration order match `storeTile`
  ##   - a tile fully inside both limits stores through `storeTile` (no guard)
  if validM >= int32(R) and validN >= int32(C):
    storeTile(gl, tile, origin)
  else:
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
        for vptI in 0 ..< vpt:
          if row + n * M < int(validM) and col + m * N + vptI < int(validN):
            dst[row + n * M, col + m * N + vptI] = tile.frags[n][m].frag[vptI].to(TOut)
