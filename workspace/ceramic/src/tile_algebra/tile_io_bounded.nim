## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Bounded tile I/O: per-lane guarded loads and the store predication mask.
##
## A tile is bounded when its plane origin plus extent crosses the real region (raw runtime M/N/K, no caller padding).
## The per-lane rule:
##
##   lane inside both limits      lane outside either limit
##   loadTileBounded: read gmem             zero fill in-register
##   tileStoreMask:   bit set               bit clear
##   finalStore:      write D               leave D untouched
##
## Out-of-range lanes never touch memory. A tile fully inside both limits
## takes the plain `loadTile` path with no guard cost.
##
## Plane origins: `origin[2]·R` (RtLeft) or `origin[2]·C` (RtRight).
## `origin[3]·C` / `origin[3]·R` give the second plane axis.
## `limitRows`/`limitCols` are the real extents along the view's dims.

import workspace/crucible
import ../int_tuples
import ../layout_indexing
import ../tensors
import ./tiles
import ./tile_config
import ./tile_io
import ./tile_ops_unary

# ═════════════════════════════════════════════════════════════════════════
#  The zero fill value (fp16-safe)
# ═════════════════════════════════════════════════════════════════════════

func zeroTileValue*(T: typedesc): T {.inline.} =
  ## The zero fill for an out-of-range lane. fp16 has no `T(0)`
  ## constructor (the distinct-type gap): its zero is the bit pattern
  ## `0'u16.asFp16()`.
  when T is float16: 0'u16.asFp16()
  else: T(0)

# ═════════════════════════════════════════════════════════════════════════
#  RtLeft variants (unswapped views)
#  ═════════════════════════════════════════════════════════════════════════
#
#  loadTile RtLeft indexes src[row + n·M, col + m·N + v] over a (R, C) view:
#  - the frag row-tile index n spans the view's first dim (plane rows, origin[2]·R)
#  - the frag col-tile index m spans the view's second dim (plane cols, origin[3]·C)

proc loadTileBounded[TIn; TOut; R, C: static int; A: static MmaAtom](
    tile: var RtLeft[TOut, R, C, A],
    gl: GlView[TIn],
    origin: tuple,
    r0, c0, limitRows, limitCols: int32) {.inline.} =
  ## Per-lane guarded load for a straddling RtLeft tile.
  ## An owned lane reads its element only when both plane coordinates
  ## are inside the real region. Out-of-range lanes hold the zero fill.
  ## Same lane→cell mapping and `to` chokepoint as `loadTile`.
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
      for v in 0 ..< vpt:
        if r0 + int32(n * M + row) < limitRows and
           c0 + int32(m * N + col + v) < limitCols:
          tile.frags[n][m].frag[v] = src[row + n * M, col + m * N + v].to(TOut)
        else:
          tile.frags[n][m].frag[v] = zeroTileValue(TOut)

proc loadTileBounded*[TIn; TOut; R, C: static int; A: static MmaAtom](
    tile: var RtLeft[TOut, R, C, A],
    gl: GlView[TIn],
    origin: tuple,
    limitRows, limitCols: int32) {.inline.} =
  ## Bounded loadTile. Lanes whose plane row >= limitRows or plane col >= limitCols read zeros instead of memory.
  ## The mma accumulates exact zeros there.
  ## No lane is ever read past the caller's allocation.
  ## A tile fully inside both limits loads through the facility with zero guard cost.
  let r0 = int32(origin[2]) * int32(R)
  let c0 = int32(origin[3]) * int32(C)
  if r0 + int32(R) <= limitRows and c0 + int32(C) <= limitCols:
    loadTile(tile, gl, origin)
  else:
    loadTileBounded(tile, gl, origin, r0, c0, limitRows, limitCols)

# ═════════════════════════════════════════════════════════════════════════
#  RtRight variants (swapped views)
#  ═════════════════════════════════════════════════════════════════════════
#
#  loadTile RtRight indexes src[col + m·N + v, row + n·M] over a (C, R) view:
#  - the frag col-tile index m spans the view's first dim (plane rows, origin[2]·C)
#  - the frag row-tile index n spans the view's second dim (plane cols, origin[3]·R)

proc loadTileBounded[TIn; TOut; R, C: static int; A: static MmaAtom](
    tile: var RtRight[TOut, R, C, A],
    gl: GlView[TIn],
    origin: tuple,
    r0, c0, limitRows, limitCols: int32) {.inline.} =
  ## Per-lane guarded load for a straddling RtRight (B-operand) tile.
  ## Follows the RtRight frag ordering (col-tile m outer, row-tile n inner).
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
      for v in 0 ..< vpt:
        if r0 + int32(col + m * N + v) < limitRows and
           c0 + int32(row + n * M) < limitCols:
          tile.frags[m][n].frag[v] = src[col + m * N + v, row + n * M].to(TOut)
        else:
          tile.frags[m][n].frag[v] = zeroTileValue(TOut)

proc loadTileBounded*[TIn; TOut; R, C: static int; A: static MmaAtom](
    tile: var RtRight[TOut, R, C, A],
    gl: GlView[TIn],
    origin: tuple,
    limitRows, limitCols: int32) {.inline.} =
  ## Bounded loadTile for the swapped (B-operand) views.
  ## Lanes whose plane row >= limitRows or plane col >= limitCols read zeros instead of memory.
  ## A tile fully inside both limits loads through the facility.
  let r0 = int32(origin[2]) * int32(C)
  let c0 = int32(origin[3]) * int32(R)
  if r0 + int32(C) <= limitRows and c0 + int32(R) <= limitCols:
    loadTile(tile, gl, origin)
  else:
    loadTileBounded(tile, gl, origin, r0, c0, limitRows, limitCols)

# ═════════════════════════════════════════════════════════════════════════
#  tileStoreMask: per-lane store predication
#  ═════════════════════════════════════════════════════════════════════════

proc tileStoreMask*[T; R, C: static int; A: static MmaAtom](
    tile: RtLeft[T, R, C, A], validM, validN: int): int {.inline.} =
  ## Per-lane store predication for the tile's valid (M, N) range.
  ## Bit ((n·colTiles + m)·vpt + v) is set when the lane's cell
  ## (row + n·M, col + m·N + v) is inside the valid range. The bit order
  ## matches the storeTile/finalStore iteration order. The `tile` value
  ## carries the (R, C, atom) geometry as type params.
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
      for v in 0 ..< vpt:
        if row + n * M < validM and col + m * N + v < validN:
          result = result or (1 shl ((n * colTiles + m) * vpt + v))
