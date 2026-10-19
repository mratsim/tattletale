# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Register tiles on the layout algebra.
##
## - `rt`: the unified register tile, an owned n-d tensor with a compile-time layout
## - `zero`: the per-type zero for register cells and pad values
## - `getLocalTileID` / `getLocalThreadID`: launch mapping, the squadron's tile
##   and the thread's lane inside it

import std/macros
import std/typetraits
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layouts_datatypes
import workspace/ceramic/src/tensors
import ./tile_matmul

# ═══════════════════════════════════════════════════════════════
#  Tile datatype
# ═══════════════════════════════════════════════════════════════

type Tile*[T, Sh, St] = TensorOwned[T, Sh, St]

macro rt*(T: typedesc, L: typed): typedesc =
  ## Register tile type from a layout, e.g. an atom's thread->value map.
  ##
  ## Contract:
  ## - the Tile over T with the layout's static shape and stride
  let ty = getTypeInst(L)
  doAssert ty.kind == nnkBracketExpr and ty.len == 3,
    "rt: expected a Layout[Shape, Stride] type, got " & ty.repr
  result = nnkBracketExpr.newTree(bindSym"Tile", T, ty[1], ty[2])

func staticShape*(n: NimNode): seq[int] {.compileTime.} =
  ## Static ints of a shape node: a tuple literal, a tuple type, or a typedesc wrapping one
  var nn = n
  if nn.kind == nnkStmtListExpr:
    nn = nn[^1]
  if nn.kind in {nnkTupleConstr, nnkPar}:
    for e in nn:
      result.add e.getStaticInt()
    return
  var ty = nn.getTypeInst()
  if ty.kind == nnkBracketExpr and ty[0].kind == nnkSym and $ty[0] == "typeDesc":
    ty = ty[1]
  if ty.isTupleTy():
    for i in 0 ..< ty.len:
      result.add ty[i].getStaticInt()
  else:
    result.add ty.getStaticInt()

template rt*(T: typedesc, dims: IntOrIntTuple): untyped =
  ## Register tile type from a static shape, LayoutLeft: the lane's
  ## register block under the blocked convention.
  ##
  ## Contract:
  ## - the block's last axis spreads across the simdgroup's 32 lanes
  ## - the other axes live in-thread
  make_tensor(T, dims)

# ═══════════════════════════════════════════════════════════════
#  Tile geometry
# ═══════════════════════════════════════════════════════════════

template tileWith*(v: TensorView, tiler: static tuple): auto =
  ## Tile geometry of `v`: its layout zipped-divided by the tiler,
  ## ((tile shape), (tile grid)) mode pairs.
  v.getLayout().zipped_divide(tiler)

template getTileGrid*(geom: Layout): auto =
  ## Tile grid of the geometry, the tile coordinates covering the tensor.
  geom.shape[1]

# ═══════════════════════════════════════════════════════════════
#  Sizes and fill values
# ═══════════════════════════════════════════════════════════════

template zero*(T: typedesc): untyped =
  when T is float16: cast[T](0'u16)
  elif T is bfloat16: cast[T](0'u16)
  else: T(0)

# ═══════════════════════════════════════════════════════════════
#  Launch mapping
# ═══════════════════════════════════════════════════════════════

template getLocalTileID*(gridShape: IntOrIntTuple): auto =
  ## Squadron position mapped to a flat index over the tile grid
  ## through the fleet layout.
  let w = threadgroups_per_grid.x
  let h = threadgroups_per_grid.y
  let d = threadgroups_per_grid.z
  let x = threadgroup_position_in_grid.x
  let y = threadgroup_position_in_grid.y
  let z = threadgroup_position_in_grid.z
  let fleet = make_layout(makeIntTuple((w, h, d)))
  crd2idx(fleet, (x, y, z))

template getLocalThreadID*(mma: static TiledMma): int =
  ## Lane index of the thread inside its squadron.
  thread_index_in_threadgroup
