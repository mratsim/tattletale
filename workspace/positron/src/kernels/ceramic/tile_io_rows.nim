## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Row-bounded tile load/store (positron local extension)
#
# ############################################################

## Row-bounded variants of the tile_io `loadTile`/`storeTile` for the positron
## kernels' partial-row-block contract (a decode q block, a partial last batch
## tile). Elements whose tile-plane row is at or above `rowLimit` are
## zero-filled on load and not written on store.
##
## The bodies delegate to the tile_io facilities, keeping the lane-to-cell mapping and the `to` conversion chokepoint.
## The guard lives in the epilogue position. A straddling tile
## zeroes its out-of-range rows after the load (`zeroRows`).
## The store writes only the in-range rows (`storeRows`). Each lane
## evaluates the guard once per sub-tile row, since the guard is
## uniform across the lane's columns and values.
##
## Plane-row convention: the tile's first plane row is `origin[2]·R`
## for the RtLeft variants and `origin[3]·R` for the RtRight
## variants, the same side convention loadTile uses
## (`local_tile_dyn(gl, R, C, o)` versus `(gl, C, R, o)`).
##
## The row-bounded load reads the whole tile, out-of-range rows
## included, then zeroes those rows. Metal performs bounds checking
## on buffer accesses: out-of-bounds reads return 0 and writes are discarded,
## so the extra reads stay invisible.
##
## Local to positron: the shared tile_algebra (tile_io.nim) is owned
## elsewhere, so the bounded variants live here.

import workspace/crucible
import workspace/ceramic

# ═════════════════════════════════════════════════════════════════════
#  RtLeft variants (unswapped views, guard on origin[2])
#  ═════════════════════════════════════════════════════════════════════

proc zeroRows[R, C: static int; A: static MmaAtom](
    tile: var RtLeft[float16, R, C, A],
    r0, rowLimit: int32) {.device.} =
  ## Zeroes the tile's rows with plane row >= rowLimit. `r0` is the tile's
  ## first plane row (origin[2]·R).
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  for n in 0 ..< rowTiles:
    if r0 + int32(n * M + row) >= rowLimit:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          tile.frags[n][m].frag[v] = 0'u16.asFp16()

proc loadTileRows*[R, C: static int; A: static MmaAtom](
    tile: var RtLeft[float16, R, C, A],
    gl: GlView[float16],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded loadTile: tile-plane rows origin[2]·R + r at or above
  ## `rowLimit` are zero-filled instead of read.
  tile.loadTile(gl, origin)
  let r0 = int32(origin[2]) * int32(R)
  if r0 + int32(R) > rowLimit:
    zeroRows(tile, r0, rowLimit)

proc storeRows[TIn; R, C: static int; A: static MmaAtom](
    gl: GlView[float16],
    tile: RtLeft[TIn, R, C, A],
    origin: tuple,
    r0, rowLimit: int32) {.device.} =
  ## Stores only the tile's in-range rows (plane row < rowLimit).
  ## The fp32 tile quantizes to fp16 (RNE) via the `to` chokepoint.
  ## An fp16 tile round-trips unchanged.
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
          dst[row + n * M, col + m * N + v] =
            tile.frags[n][m].frag[v].to(float16)

proc storeTileRows*[R, C: static int; A: static MmaAtom](
    gl: GlView[float16],
    tile: RtLeft[float32, R, C, A],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded storeTile: rows at or above `rowLimit` are not written.
  ## A tile fully inside the limit stores through the facility. A straddling tile writes only its in-range rows.
  let r0 = int32(origin[2]) * int32(R)
  if r0 + int32(R) <= rowLimit:
    gl.storeTile(tile, origin)
  else:
    storeRows(gl, tile, origin, r0, rowLimit)

proc storeTileRows*[R, C: static int; A: static MmaAtom](
    gl: GlView[float16],
    tile: RtLeft[float16, R, C, A],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded storeTile for fp16 tiles. The `to` round trip is the identity.
  let r0 = int32(origin[2]) * int32(R)
  if r0 + int32(R) <= rowLimit:
    gl.storeTile(tile, origin)
  else:
    storeRows(gl, tile, origin, r0, rowLimit)

# ═════════════════════════════════════════════════════════════════════
#  RtRight variants (swapped views, guard on origin[3])
#  ═════════════════════════════════════════════════════════════════════

proc zeroRows[T; R, C: static int; A: static MmaAtom](
    tile: var RtRight[T, R, C, A],
    r0, rowLimit: int32) {.device.} =
  ## Zeroes the tile's rows with plane row >= rowLimit, following the RtRight
  ## frag ordering (col-tile m outer, row-tile n inner).
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  for n in 0 ..< rowTiles:
    if r0 + int32(n * M + row) >= rowLimit:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          tile.frags[m][n].frag[v] =
            when T is float16: 0'u16.asFp16()
            else: T(0)

proc loadTileRows*[TIn, TOut; R, C: static int; A: static MmaAtom](
    tile: var RtRight[TOut, R, C, A],
    gl: GlView[TIn],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded loadTile for the swapped rmsnorm views: tile-plane
  ## rows origin[3]·R + r at or above `rowLimit` are zero-filled instead of read.
  tile.loadTile(gl, origin)
  let r0 = int32(origin[3]) * int32(R)
  if r0 + int32(R) > rowLimit:
    zeroRows(tile, r0, rowLimit)

proc storeRows[TIn; R, C: static int; A: static MmaAtom](
    gl: GlView[float16],
    tile: RtRight[TIn, R, C, A],
    origin: tuple,
    r0, rowLimit: int32) {.device.} =
  ## Stores only the tile's in-range rows (plane row < rowLimit),
  ## with the RtRight frag ordering and the swapped-view indexing.
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
  var dst = local_tile_dyn(gl, C, R, o)
  for n in 0 ..< rowTiles:
    if r0 + int32(n * M + row) < rowLimit:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          dst[col + m * N + v, row + n * M] =
            tile.frags[m][n].frag[v].to(float16)

proc storeTileRows*[R, C: static int; A: static MmaAtom](
    gl: GlView[float16],
    tile: RtRight[float32, R, C, A],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded storeTile for the swapped rmsnorm views: rows at or above `rowLimit` are not written.
  let r0 = int32(origin[3]) * int32(R)
  if r0 + int32(R) <= rowLimit:
    gl.storeTile(tile, origin)
  else:
    storeRows(gl, tile, origin, r0, rowLimit)
