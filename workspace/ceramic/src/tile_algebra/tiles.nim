# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../layout_indexing
import ../tensors
import ../ptr_arithmetic
import ./tile_config
import workspace/crucible

export tile_config

type
  RtLeft*[T; R, C: static int; A: static MmaAtom] = object
    ## LayoutLeft (col-major) register tile of an R×C matrix (rows × columns).
    frags*: array[R div A.getM(), array[C div A.getN(), SubTile[A, T]]]

  RtRight*[T; R, C: static int; A: static MmaAtom] = object
    ## LayoutRight (row-major) register tile of an R×C matrix (rows × columns).
    frags*: array[C div A.getN(), array[R div A.getM(), SubTile[A, T]]]

template rt_l*(T: typedesc, R, C: static int, A: untyped = getTileConfig(float32, T)): typedesc =
  ## LayoutLeft (col-major) register tile for an R×C matrix.
  RtLeft[T, R, C, A]

template rt_r*(T: typedesc, R, C: static int, A: untyped = getTileConfig(float32, T)): typedesc =
  ## LayoutRight (row-major) register tile for an R×C matrix.
  RtRight[T, R, C, A]

func zero*[T; R, C: static int; A: static MmaAtom](
    tile: var RtLeft[T, R, C, A]) =
  ## `tile.zero()`: zeroes the accumulator's owned cells.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for vptI in 0 ..< vpt:
        tile.frags[n][m].frag[vptI] = T(0)

func zero*[T; R, C: static int; A: static MmaAtom](
    tile: var RtRight[T, R, C, A]) =
  ## `tile.zero()`: zeroes the accumulator's owned cells.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for m in 0 ..< colTiles:
    for n in 0 ..< rowTiles:
      for vptI in 0 ..< vpt:
        tile.frags[m][n].frag[vptI] = T(0)

# ═════════════════════════════════════════════════════════════════════════
#  rv: the row-reduction col-vec
# ═════════════════════════════════════════════════════════════════════════

template rv*(T: typedesc, R, C: static int, A: untyped = getTileConfig(float32, T)): typedesc =
  ## Column vector of shape (rows, values per thread).
  ## This stores the result of
  ## a row-reduction operation f a tile like max or sum
  Tensor[T,
         (Int[R div A.getM()], Int[A.getVpt()]),
         (Int[A.getVpt()], Int[1])]

# ═════════════════════════════════════════════════════════════════════════
#  The lane→cell decomposition and the single-scalar slots
# ═════════════════════════════════════════════════════════════════════════

template laneCellOf*(A: untyped): untyped =
  ## Returns the atom's A-fragment cell index this lane owns.
  ##
  ## Contract:
  ## - the shared lane→cell decomposition, `crd2idx(A.getLayoutA(), (lane, 0))`
  ## - a template, the atom's layout folds into the call site's device body
  ## - runs in device context, `thread_index_in_threadgroup` valid
  crd2idx(A.getLayoutA(), (int(thread_index_in_threadgroup), 0)).toIntVal()

template laneRowOf*(A: untyped): untyped =
  ## Returns the fragment row this lane owns, `laneCellOf(A) mod A.getM()`.
  laneCellOf(A) mod A.getM()

template laneColOf*(A: untyped): untyped =
  ## Returns the fragment column this lane owns, `laneCellOf(A) div A.getM()`.
  laneCellOf(A) div A.getM()

func laneScalar*[T; R, C: static int; A: static MmaAtom](
    tile: RtLeft[T, R, C, A]): T =
  ## Returns the single-value slot of a tile, `tile.frags[0][0].frag[0]`.
  ##
  ## Contract:
  ## - the tile's one useful element carries the (0, 0) fragment of every lane
  ## - serves the v operand's scalar read and one-element broadcast loads
  tile.frags[0][0].frag[0]

func rowScalar*[T; rowTiles, vpt: static int](
    vec: Tensor[T, (Int[rowTiles], Int[vpt]), (Int[vpt], Int[1])]): T =
  ## Returns the calling lane's row-0 slot of the row-reduction col-vec,
  ## `vec.data[0]`, the scalar its fragment row just reduced.
  vec.data[0]

# ═════════════════════════════════════════════════════════════════════════
#  GlView: Global Views / Data Descriptors
# ═════════════════════════════════════════════════════════════════════════

type
  GlView*[T] = TensorView[T, (int, int, int, int), (int, int, int, int)]
    ## View over a runtime global buffer of shape (batch, depth, rows, cols)

func gd*[T](buf: ptr UncheckedArray[T], batch, depth, rows, cols: SomeInteger): GlView[T] {.inline.}=
  ## Builds a descriptor over a global view with shape (batch, depth, row, col)
  block:
    let b = int(batch)
    let d = int(depth)
    let r = int(rows)
    let c = int(cols)
    let one = 1
    GlView[T](data: buf,
              layout: make_layout((b, d, r, c), (d * r * c, r * c, c, one)))

func gd*[T](buf: ptr UncheckedArray[T],
                shape, stride: tuple): GlView[T] {.inline.} =
  ## Builds a view from explicit (batch, depth, rows, cols) pairs of shape and stride
  block:
    let sh = shape
    let st = stride
    let b = int(sh[0])
    let d = int(sh[1])
    let r = int(sh[2])
    let c = int(sh[3])
    let sb = int(st[0])
    let sd = int(st[1])
    let sr = int(st[2])
    let sc = int(st[3])
    GlView[T](data: buf,
              layout: make_layout((b, d, r, c), (sb, sd, sr, sc)))

func local_tile_dyn*[T](
        gl: GlView[T],
        R, C: static int,
        origin: tuple): TensorView[T, (int, int), (int, int)] {.inline.} =
  ## `local_tile` for dynamic layouts.
  ## Workaround for local_tile's template nesting generated kilometers of code
  ## and slowing compilation and triggering VM max limits
  block:
    let o0 = int(origin[0])
    let o1 = int(origin[1])
    let o2 = int(origin[2])
    let o3 = int(origin[3])
    let sb = int(gl.layout.stride[0])
    let sd = int(gl.layout.stride[1])
    let sr = int(gl.layout.stride[2])
    let sc = int(gl.layout.stride[3])
    let base = o0 * sb + o1 * sd + o2 * sr * R + o3 * sc * C
    let r = int(R)
    let c = int(C)
    TensorView[T, (int, int), (int, int)](
      data: gl.data +% base,
      layout: Layout[(int, int), (int, int)](
        shape: (r, c),
        stride: (sr, sc)))
