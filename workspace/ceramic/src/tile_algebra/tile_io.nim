## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../int_tuples
import ../layout_indexing
import ../layout_algebra
import ../tensors
import ./tiles
import ./tile_config
import ./tile_ops_unary
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  Loads
# ═════════════════════════════════════════════════════════════════════════

proc loadTile*[TIn; TOut; R, C: static int; A: static MmaAtom](
    tile: var RtLeft[TOut, R, C, A],
    gl: GlView[TIn],
    origin: tuple) =
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
        tile.frags[n][m].frag[vptI] = src[row + n * M, col + m * N + vptI].to(TOut)

proc loadTile*[TIn; TOut; R, C: static int; A: static MmaAtom](
    tile: var RtRight[TOut, R, C, A],
    gl: GlView[TIn],
    origin: tuple) =
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
        tile.frags[m][n].frag[vptI] = src[col + m * N + vptI, row + n * M].to(TOut)

# ═════════════════════════════════════════════════════════════════════════
#  Stores
# ═════════════════════════════════════════════════════════════════════════

proc storeTile*[TIn; TOut; R, C: static int; A: static MmaAtom](
    gl: GlView[TOut],
    tile: RtLeft[TIn, R, C, A],
    origin: tuple) =
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
        dst[row + n * M, col + m * N + vptI] = tile.frags[n][m].frag[vptI].to(TOut)

proc storeTile*[TIn; TOut; R, C: static int; A: static MmaAtom](
    gl: GlView[TOut],
    tile: RtRight[TIn, R, C, A],
    origin: tuple) =
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
  for m in 0 ..< colTiles:
    for n in 0 ..< rowTiles:
      for vptI in 0 ..< vpt:
        dst[col + m * N + vptI, row + n * M] = tile.frags[m][n].frag[vptI].to(TOut)
