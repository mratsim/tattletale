## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#          Tile views: GlView + the tile-plane addressing
#
# ############################################################
#
# The runtime addressing of a global tile view: the GlView stride
# record, the per-lane storage partition layout (n, m, vpt), the view
# base offset, and the tile-plane view the loads and stores index.
# The loads and stores iterate the storage partition via idx2crd,
# reading or writing the buffer element through the tile-plane view.

import ../int_tuples
import ../layouts
import ../layout_constructors
import ../layout_indexing
import ../tensors
import ../ptr_arithmetic

# ═════════════════════════════════════════════════════════════════════════
#  GlView: the runtime addressing of a global tile view
# ═════════════════════════════════════════════════════════════════════════

type
  GlView*[T] = object
    ## The global view: the buffer pointer plus the batch/depth/row/col strides of the view axes.
    base*: ptr UncheckedArray[T]
    strideBatch*: int32
    strideDepth*: int32
    strideRow*: int32
    strideCol*: int32

proc makeGl*[T](buf: ptr UncheckedArray[T];
                aBatch, aDepth, aRows, aCols: SomeInteger): GlView[T] =
  ## Builds a view from the dims form: strideRow = cols, strideDepth = rows·cols,
  ## strideBatch = depth·rows·cols, strideCol = 1.
  ## The base pointer keeps the caller's address space
  ## (the kernel buffer params are device pointers), so the proc form
  ## lowers unchanged.
  GlView[T](base: buf,
            strideBatch: int32(int64(aBatch) * int64(aDepth) * int64(aRows) * int64(aCols)),
            strideDepth: int32(int64(aRows) * int64(aCols)),
            strideRow: int32(aCols),
            strideCol: 1'i32)

proc makeGlStrided*[T](buf: ptr UncheckedArray[T];
                       aStrideBatch, aStrideDepth, aStrideRow,
                       aStrideCol: SomeInteger): GlView[T] =
  ## Builds a view with explicit strides, for the views the dims form
  ## cannot express (the rmsnorm per-tile views, stride-0 broadcast rows).
  ## See makeGl on the address space.
  GlView[T](base: buf, strideBatch: int32(aStrideBatch),
            strideDepth: int32(aStrideDepth),
            strideRow: int32(aStrideRow), strideCol: int32(aStrideCol))

# ═════════════════════════════════════════════════════════════════════════
#  The tile-plane addressing
# ═════════════════════════════════════════════════════════════════════════

func baseOffset*[T; O: tuple](gl: GlView[T]; origin: O;
                              R, C: static int; fm, fn: uint32;
                              colTile: static bool): int64 {.inline.} =
  ## The view base: the origin slice offsets (batch·strideBatch + depth·strideDepth)
  ## plus the tile offsets. Signed: negative strides must address backwards,
  ## and `+%` wraps an int64 offset correctly (a negative int32 would
  ## zero-extend). `colTile` selects the transposed-B scaling: the tile row scales
  ## by C, the tile col by R, and the lane's fm/fn swap axes.
  when colTile:
    int64(int(origin[0]) * int(gl.strideBatch) +
          int(origin[1]) * int(gl.strideDepth) +
          (int(origin[2]) * int(C) + int(fn)) * int(gl.strideRow) +
          (int(origin[3]) * int(R) + int(fm)) * int(gl.strideCol))
  else:
    int64(int(origin[0]) * int(gl.strideBatch) +
          int(origin[1]) * int(gl.strideDepth) +
          (int(origin[2]) * int(R) + int(fm)) * int(gl.strideRow) +
          (int(origin[3]) * int(C) + int(fn)) * int(gl.strideCol))

proc storageLayout*[rowTiles, colTiles, vpt: static int]: auto =
  ## The per-lane storage partition layout of one tile, with the vpt
  ## axis fastest: slot (n, m, vptI) maps to the flat index
  ## (n·colTiles + m)·vpt + vptI, the slot formula of the tile's fragment
  ## grid.
  make_layout((Int[rowTiles](), Int[colTiles](), Int[vpt]()),
              (Int[vpt * colTiles](), Int[vpt](), Int[1]()))

proc bufferView*[T; rowTiles, colTiles, vpt, subtileM, subtileN: static int](
    gl: GlView[T]; baseOff: int64;
    colTile: static bool): TensorView[T, (Int[rowTiles], Int[colTiles], Int[vpt]), (int, int, int)] =
  ## A TensorView over the buffer at `baseOff` with the tile-plane
  ## strides: (n, m, vptI) = (subtileM·strideRow, subtileN·strideCol,
  ## strideCol) for the natural tile, swapped (subtileN·strideCol,
  ## subtileM·strideRow, strideRow) for the transposed-B col tile (the vptI axis
  ## steps the row there). The subtile dims are the atom's mnk,
  ## supplied at the call site. Every atom addresses its own tile plane.
  when colTile:
    make_view(gl.base +% baseOff,
              make_layout((Int[rowTiles](), Int[colTiles](), Int[vpt]()),
                          (int(subtileM) * int(gl.strideCol),
                           int(subtileN) * int(gl.strideRow),
                           int(gl.strideRow))))
  else:
    make_view(gl.base +% baseOff,
              make_layout((Int[rowTiles](), Int[colTiles](), Int[vpt]()),
                          (int(subtileM) * int(gl.strideRow),
                           int(subtileN) * int(gl.strideCol),
                           int(gl.strideCol))))
