## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#          Tile I/O: the load/store dataflow
#
# ############################################################
#
# RtLeft tiles address a row-major view, RtRight the transposed-B view.
# The stores convert the fp32 accumulator to the storage type when writing.
# No per-element guards: the caller pads the source to the tile width.

import ../int_tuples
import ../layout_indexing
import ../atoms
import ./tiles
import ./tile_config
import ./tile_views
import ./tile_fma_partition
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  The loads: one proc per map, selected by the tile's layout
# ═════════════════════════════════════════════════════════════════════════

proc loadTile*[T; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    tile: var RtLeft[T, R, C, A, TL]; gl: GlView[T];
    origin: tuple) =
  ## Loads this thread's rows of the R×C tile from a row-major view at origin.
  ## The caller pads the source to the tile width.
  const rowTiles = R div A.mnk.m
  const colTiles = C div A.mnk.n
  const vpt = toIntVal(A.valuesPerThread(opA))
  let thr = fmaSlice[A, TL]()
  let lane = int(thread_index_in_threadgroup)
  let baseOff = baseOffset(gl, origin, R, C,
                           uint32(laneFm[A](lane)), uint32(laneFn[A](lane)),
                           colTile = false)
  let src = bufferView[T, rowTiles, colTiles, vpt, A.mnk.m, A.mnk.n](gl, baseOff, colTile = false)
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt:
        tile.frags[n][m].frag[vptI] = src(n, m, vptI)

proc loadTile*[T; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    tile: var RtRight[T, R, C, A, TL]; gl: GlView[T];
    origin: tuple) =
  ## Loads this thread's columns of the R×C tile from the transposed-B
  ## view at origin. The caller pads the source to the tile width.
  const rowTiles = R div A.mnk.m
  const colTiles = C div A.mnk.n
  const vpt = toIntVal(A.valuesPerThread(opA))
  let thr = fmaSlice[A, TL]()
  let lane = int(thread_index_in_threadgroup)
  let baseOff = baseOffset(gl, origin, R, C,
                           uint32(laneFm[A](lane)), uint32(laneFn[A](lane)),
                           colTile = true)
  let src = bufferView[T, rowTiles, colTiles, vpt, A.mnk.m, A.mnk.n](gl, baseOff, colTile = true)
  for m in countup(thr.tn, colTiles - 1, TL.thrN):
    for n in 0 ..< rowTiles:
      for vptI in 0 ..< vpt:
        tile.frags[m][n].frag[vptI] = src(n, m, vptI)

proc loadTileFp32*[TIn; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    tile: var RtRight[float32, R, C, A, TL]; gl: GlView[TIn];
    origin: tuple) =
  ## Loads this thread's columns into an fp32 tile, promoting each TIn element to fp32.
  ## fp16 storage widens in registers, so a fused chain needs no store-load-convert round-trip.
  const rowTiles = R div A.mnk.m
  const colTiles = C div A.mnk.n
  const vpt = toIntVal(A.valuesPerThread(opA))
  let thr = fmaSlice[A, TL]()
  let lane = int(thread_index_in_threadgroup)
  let baseOff = baseOffset(gl, origin, R, C,
                           uint32(laneFm[A](lane)), uint32(laneFn[A](lane)),
                           colTile = true)
  let src = bufferView[TIn, rowTiles, colTiles, vpt, A.mnk.m, A.mnk.n](gl, baseOff, colTile = true)
  for m in countup(thr.tn, colTiles - 1, TL.thrN):
    for n in 0 ..< rowTiles:
      for vptI in 0 ..< vpt:
        tile.frags[m][n].frag[vptI] = toFp32(src(n, m, vptI))

# ═════════════════════════════════════════════════════════════════════════
#  The stores
# ═════════════════════════════════════════════════════════════════════════

proc storeTile*[TOut; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    gl: GlView[TOut]; tile: RtLeft[float32, R, C, A, TL];
    origin: tuple) =
  ## Writes the accumulator's owned cells to a row-major view at origin.
  ## TOut: fp32 copies, fp16 rounds (RNE), any other type fails at compile time.
  const rowTiles = R div A.mnk.m
  const colTiles = C div A.mnk.n
  const vpt = toIntVal(A.valuesPerThread(opA))
  let thr = fmaSlice[A, TL]()
  let lane = int(thread_index_in_threadgroup)
  let baseOff = baseOffset(gl, origin, R, C,
                           uint32(laneFm[A](lane)), uint32(laneFn[A](lane)),
                           colTile = false)
  var dst = bufferView[TOut, rowTiles, colTiles, vpt, A.mnk.m, A.mnk.n](gl, baseOff, colTile = false)
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in countup(thr.tn, colTiles - 1, TL.thrN):
      for vptI in 0 ..< vpt:
        when TOut is float32:
          dst[n, m, vptI] = tile.frags[n][m].frag[vptI]
        elif TOut is float16:
          dst[n, m, vptI] = toFp16(tile.frags[n][m].frag[vptI])
        else:
          {.error: "storeTile: unsupported output storage type".}

proc storeTile*[TOut; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    gl: GlView[TOut]; tile: RtRight[float32, R, C, A, TL];
    origin: tuple) =
  ## Same as the RtLeft storeTile, on the transposed-B map.
  ## TOut: fp32 copies, fp16 rounds (RNE), any other type fails at
  ## compile time.
  const rowTiles = R div A.mnk.m
  const colTiles = C div A.mnk.n
  const vpt = toIntVal(A.valuesPerThread(opA))
  let thr = fmaSlice[A, TL]()
  let lane = int(thread_index_in_threadgroup)
  let baseOff = baseOffset(gl, origin, R, C,
                           uint32(laneFm[A](lane)), uint32(laneFn[A](lane)),
                           colTile = true)
  var dst = bufferView[TOut, rowTiles, colTiles, vpt, A.mnk.m, A.mnk.n](gl, baseOff, colTile = true)
  for m in countup(thr.tn, colTiles - 1, TL.thrN):
    for n in 0 ..< rowTiles:
      for vptI in 0 ..< vpt:
        when TOut is float32:
          dst[n, m, vptI] = tile.frags[m][n].frag[vptI]
        elif TOut is float16:
          dst[n, m, vptI] = toFp16(tile.frags[m][n].frag[vptI])
        else:
          {.error: "storeTile: unsupported output storage type".}

# ═════════════════════════════════════════════════════════════════════════
#  The kernel spellings: `tile.load(gl, origin)` / `tile.store(gl, origin)`
# ═════════════════════════════════════════════════════════════════════════

proc load*[T; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    tile: var RtLeft[T, R, C, A, TL]; gl: GlView[T];
    origin: tuple) =
  ## `tile.load(gl, origin)`: loads this thread's rows.
  loadTile(tile, gl, origin)

proc load*[T; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    tile: var RtRight[T, R, C, A, TL]; gl: GlView[T];
    origin: tuple) =
  ## `tile.load(gl, origin)`: loads this thread's columns.
  loadTile(tile, gl, origin)

proc load*[TIn; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    tile: var RtRight[float32, R, C, A, TL]; gl: GlView[TIn];
    origin: tuple) =
  ## `tile.load(gl, origin)`: loads this thread's columns, each TIn
  ## element promoted to fp32.
  loadTileFp32(tile, gl, origin)

proc store*[TOut; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    tile: RtLeft[float32, R, C, A, TL]; gl: GlView[TOut];
    origin: tuple) =
  ## `tile.store(gl, origin)`: stores the accumulator's owned cells,
  ## converted to the TOut storage type.
  storeTile(gl, tile, origin)

proc store*[TOut; R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    tile: RtRight[float32, R, C, A, TL]; gl: GlView[TOut];
    origin: tuple) =
  ## `tile.store(gl, origin)`: stores the tile's columns, converted to
  ## the TOut storage type.
  storeTile(gl, tile, origin)
