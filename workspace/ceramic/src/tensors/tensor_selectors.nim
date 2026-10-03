## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensor selection: element access, subviews, slicing, partitioning.

import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import std/macros

import workspace/ceramic/src/macros/varargs_to_par
import workspace/ceramic/src/int_tuples
import ./tensor_datatypes


proc pop(tree: var NimNode): NimNode {.compileTime.} =
  result = tree[tree.len-1]
  tree.del(tree.len-1)

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  `()` — dual dispatch: all-int → element, has _ → sub-View
# ═════════════════════════════════════════════════════════════════════════

template tensorSubViewImpl(dataBase: untyped, layout: Layout, coords: varargs[untyped]): untyped =
  # Layout and coords are pasted verbatim twice, this is intentional.
  # Layout and coordinates are pure so no side-effect will be evaluated twice
  # and we argue that it's easier for compilers, especially shader compilers that might not be as tuned as LLVM,
  # to do constant folding and common subexpression elimination on an integer expression
  # than through temporaries especially if wrapped in block expressions.
  make_view(dataBase +% toIntVal(crd2idx(layout, varargs_to_par(coords))), slice(layout, varargs_to_par(coords)))

template `()`*(t: TensorOwned; args: varargs[untyped]): untyped =
  when hasUnderscore(args):
    tensorSubViewImpl(t.data[0].addr, t.layout, args)
  else:
    # We can't wrap the whole expression into a block or it isn't a lvalue
    # and so can't be assigned to.
    # At the same time, coord MUST be wrapped, or we have scoping and name collision issues.
    {.warning: "Assignment through `()` is discouraged, use `[]=` instead".}
    t.data[toIntVal crd2idx(t.layout, varargs_to_par(args))]

template `()`*(tv: TensorView; args: varargs[untyped]): untyped =
  when hasUnderscore(args):
    tensorSubViewImpl(tv.data, tv.layout, args)
  else:
    # We can't wrap the whole expression into a block or it isn't a lvalue
    # and so can't be assigned to.
    # At the same time, coord MUST be wrapped, or we have scoping and name collision issues.
    {.warning: "Assignment through `()` is discouraged, use `[]=` instead".}
    tv.data[toIntVal crd2idx(tv.layout, varargs_to_par(args))]

# ═════════════════════════════════════════════════════════════════════════
#  `[]` — element access only (underscore rejected)
# ═════════════════════════════════════════════════════════════════════════

template `[]`*(t: TensorOwned; args: varargs[untyped]): untyped =
  when hasUnderscore(varargs_to_par(args)):
    {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
  t.data[toIntVal crd2idx(t.layout, varargs_to_par(args))]

template `[]`*(tv: TensorView; args: varargs[untyped]): untyped =
  when hasUnderscore(varargs_to_par(args)):
    {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
  tv.data[toIntVal crd2idx(tv.layout, varargs_to_par(args))]

macro `[]=`*(t: TensorOwned; args: varargs[untyped]): untyped =
  var a = args
  let val = pop(a)
  let coord = getAST(varargs_to_par(a))
  result = quote do:
    when hasUnderscore(`coord`):
      {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
    else:
      `t`.data[toIntVal crd2idx(`t`.layout, `coord`)] = `val`

macro `[]=`*(tv: TensorView; args: varargs[untyped]): untyped =
  var a = args
  let val = pop(a)
  let coord = getAST(varargs_to_par(a))
  result = quote do:
    when hasUnderscore(`coord`):
      {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
    else:
      `tv`.data[toIntVal crd2idx(`tv`.layout, `coord`)] = `val`

# ═════════════════════════════════════════════════════════════════════════
#  slice — subtensor via underscore dispatch
# ═════════════════════════════════════════════════════════════════════════

template slice*(t: TensorOwned; coords: varargs[untyped]): untyped =
  tensorSubViewImpl(t.data[0].addr, t.layout, coords)

template slice*(tv: TensorView; coords: varargs[untyped]): untyped =
  tensorSubViewImpl(tv.data, tv.layout, coords)

# ═════════════════════════════════════════════════════════════════════════
#  repeatTuple
# ═════════════════════════════════════════════════════════════════════════

macro repeat(elem: typed, n: static int): untyped =
  result = nnkTupleConstr.newTree()
  for i in 0 ..< n:
    result.add elem

# ═════════════════════════════════════════════════════════════════════════
#  inner_partition / outer_partition / local_tile / local_partition
#  CuTe: tensor_impl.hpp, zipped_divide + slice_and_offset
# ═════════════════════════════════════════════════════════════════════════

#  Static tiler contract:
#  - CuTe tilers are static, the tile shape is a compile-time constant carried through composition
#  - makeIntTuple promotes compile-time-known tiler int leaves (literals, const symbols) to Int[N]()
#  - the tile coords stay runtime, the coord is a runtime value, the shape is static

macro partitionImpl(tv: typed, zipped: Layout, coord: typed, doInner: static bool): untyped =
  result = newStmtList()
  let (sh, st) = result.destructureLayout(zipped)
  let tileShape = getTupleIndex(sh, 0)
  let restShape = getTupleIndex(sh, 1)
  let tileStride = getTupleIndex(st, 0)
  let restStride = getTupleIndex(st, 1)
  let (slicedShape, slicedStride, wholeShape, wholeStride) =
    if doInner: (restShape, restStride, tileShape, tileStride)
    else: (tileShape, tileStride, restShape, restStride)
  if coord.getTypeInst().isTupleTy():
    if doInner:
      # Tile whole first, sliced rest second.
      result.add quote do:
        make_view(`tv`.data +% toIntVal(crd2idx(`coord`, `slicedShape`, `slicedStride`)),
          make_layout(concat(`wholeShape`, slice(`slicedShape`, `coord`)),
                      concat(`wholeStride`, slice(`slicedStride`, `coord`))))
    else:
      # Sliced tile first, rest whole second.
      result.add quote do:
        make_view(`tv`.data +% toIntVal(crd2idx(`coord`, `slicedShape`, `slicedStride`)),
          make_layout(concat(slice(`slicedShape`, `coord`), `wholeShape`),
                      concat(slice(`slicedStride`, `coord`), `wholeStride`)))
  else:
    result.add quote do:
      make_view(`tv`.data +% toIntVal(crd2idx(`coord`, `slicedShape`, `slicedStride`)),
        make_layout(`wholeShape`, `wholeStride`))

template inner_partition*(tv: AnyTensor; tiler: typed; coord: typed): untyped =
  ## Cut the tensor into tiles, select the one tile `coord` targets,
  ## the rest of the tiles is gone from the view.
  ##
  ## Say a threadgroup works on one tile of a bigger tensor at a time,
  ## the tile `coord` names: the tiler cuts, `coord` picks,
  ## the view holds only the picked tile, nothing of the grid around it.
  ##
  ##    tensor (6, 8) column-major, tiles of (2, 2), coord (1, 1):
  ##    ┌─────────┬─────────┬─────────┬─────────┐
  ##    │  1   7  │ 13  19  │ 25  31  │ 37  43  │
  ##    │  2   8  │ 14  20  │ 26  32  │ 38  44  │
  ##    ├─────────┼─────────┼─────────┼─────────┤
  ##    │  3   9  │ 15  21  │ 27  33  │ 39  45  │
  ##    │  4  10  │ 16  22  │ 28  34  │ 40  46  │
  ##    ├─────────┼─────────┼─────────┼─────────┤
  ##    │  5  11  │ 17  23  │ 29  35  │ 41  47  │
  ##    │  6  12  │ 18  24  │ 30  36  │ 42  48  │
  ##    └─────────┴─────────┴─────────┴─────────┘
  ##    coord (1, 1) picks the middle tile, the view holds only it:
  ##    15 21 / 16 22, the rest of the grid drops away
  ##
  ## Use it at the threadgroup level:
  ##   one call names one tile, tile
  ##   dimensions survive, grid dimensions do not.
  ##
  ## Contract:
  ## - the result shape is the tiler's shape, memory shared with the tensor
  ## - `coord` indexes the grid of tiles, one slot per tiler dimension
  ## - a scalar coord addresses the tiles by linear index, a tuple
  ##   coord per dimension
  ## - an underscore in the coord keeps that grid slot whole,
  ##   the result has a dimension indexing the leftover tiles
  partitionImpl(tv, zipped_divide(tv.layout, tiler), coord, doInner = true)

template outer_partition*(tv: AnyTensor; tiler: typed; coord: typed): untyped =
  ## Cut the tensor into tiles, select the same slice from every tile,
  ## the tiles themselves are gone from the view.
  ##
  ## Say the threads of a threadgroup share the tile, each thread works
  ## on one slice of it: the tiler cuts, `coord` names the slice,
  ## the view holds that slice of every tile, one result dimension
  ## per grid slot
  ##
  ##    tensor (6, 8) column-major, tiles of (2, 2), coord (1, 1):
  ##    ┌─────────┬─────────┬─────────┬─────────┐
  ##    │  1   7  │ 13  19  │ 25  31  │ 37  43  │
  ##    │  2   8  │ 14  20  │ 26  32  │ 38  44  │
  ##    ├─────────┼─────────┼─────────┼─────────┤
  ##    │  3   9  │ 15  21  │ 27  33  │ 39  45  │
  ##    │  4  10  │ 16  22  │ 28  34  │ 40  46  │
  ##    ├─────────┼─────────┼─────────┼─────────┤
  ##    │  5  11  │ 17  23  │ 29  35  │ 41  47  │
  ##    │  6  12  │ 18  24  │ 30  36  │ 42  48  │
  ##    └─────────┴─────────┴─────────┴─────────┘
  ##    coord (1, 1) picks the bottom-right cell of every tile,
  ##    the view holds the grid of them as a (3, 4) block:
  ##    8 20 32 44 / 10 22 34 46 / 12 24 36 48
  ##
  ## Use it at the thread level:
  ##   one call per worker, the picked slice
  ##   repeats over every tile, the worker sees the whole tensor's rhythm.
  ##
  ## Contract:
  ## - the result shape is the grid of tiles, the tile dims drop
  ## - `coord` indexes positions inside a tile, the same positions `inner_partition`'s result covers
  ## - a scalar coord addresses the tile by linear index, a tuple coord per dimension
  partitionImpl(tv, zipped_divide(tv.layout, tiler), coord, doInner = false)

template local_tile*(tv: AnyTensor; tiler: typed; coord: typed): untyped =
  ## Select the one tile of the tensor that the current threadgroup owns.
  ##
  ## Say a kernel splits a big tensor across threadgroups: the tiler
  ## gives the tile shape, `coord` gives the threadgroup's position,
  ## the result is the tile view the mainloop loads and stores.
  ##
  ##    tensor (6, 8) column-major, tiles of (2, 2), coord (1, 1):
  ##    ┌─────────┬─────────┬─────────┬─────────┐
  ##    │  1   7  │ 13  19  │ 25  31  │ 37  43  │
  ##    │  2   8  │ 14  20  │ 26  32  │ 38  44  │
  ##    ├─────────┼─────────┼─────────┼─────────┤
  ##    │  3   9  │ 15  21  │ 27  33  │ 39  45  │
  ##    │  4  10  │ 16  22  │ 28  34  │ 40  46  │
  ##    ├─────────┼─────────┼─────────┼─────────┤
  ##    │  5  11  │ 17  23  │ 29  35  │ 41  47  │
  ##    │  6  12  │ 18  24  │ 30  36  │ 42  48  │
  ##    └─────────┴─────────┴─────────┴─────────┘
  ##    coord (1, 1) picks the middle tile, the view holds only it:
  ##    15 21 / 16 22
  ##
  ##    local_tile(tensor, (64, 64), (block_i, block_j))
  ##    # → a GEMM's (64, 64) CTA tile, threadgroups walk the grid
  ##    #  of tiles by stepping `coord`
  ##
  ## Contract:
  ## - the result shape is the tiler's shape, memory shared with the tensor
  ## - `coord` indexes the grid of tiles, one slot per tiler dimension
  ## - a scalar coord addresses the tiles by linear index, a tuple coord per dimension
  inner_partition(tv, tiler, coord)

template local_tile*(tv: AnyTensor; tiler, coord, proj: typed): untyped =
  ## 4-arg local_tile with projection, strips unwanted dimensions before partitioning.
  ## Contract: the projection marks the dimensions kept, the rest drops
  ## out of both the tiler and the coord before the 3-arg dispatch.
  ##
  ## CuTe: local_tile(tensor, tiler, coord, proj) =
  ##   local_tile(tensor, dice(proj, tiler), dice(proj, coord))
  local_tile(tv, dice(tiler, proj), dice(coord, proj))

template local_partition*(tv: AnyTensor; tile: Layout; idx: int or Int): untyped =
  ## Select the one slice of the tensor that the current thread owns.
  ##
  ## Say threads share the work tile by tile: the thread layout says
  ## who sits where, `idx` is the flat thread id, the tiler becomes
  ## the arrangement's shape, the coord its coordinates, the result is
  ## the slice thread `idx` owns repeated over the grid of tiles.
  ##
  ##    tensor (6, 8) column-major, thread layout (2, 2), thread id 2:
  ##    ┌─────────┬─────────┬─────────┬─────────┐
  ##    │  1   7  │ 13  19  │ 25  31  │ 37  43  │
  ##    │  2   8  │ 14  20  │ 26  32  │ 38  44  │
  ##    ├─────────┼─────────┼─────────┼─────────┤
  ##    │  3   9  │ 15  21  │ 27  33  │ 39  45  │
  ##    │  4  10  │ 16  22  │ 28  34  │ 40  46  │
  ##    ├─────────┼─────────┼─────────┼─────────┤
  ##    │  5  11  │ 17  23  │ 29  35  │ 41  47  │
  ##    │  6  12  │ 18  24  │ 30  36  │ 42  48  │
  ##    └─────────┴─────────┴─────────┴─────────┘
  ##    thread 2 sits at (0, 1) of the (2, 2) arrangement, the view
  ##    is the grid of top-right cells, a (3, 4) block:
  ##    7 19 31 43 / 9 21 33 45 / 11 23 35 47
  ##
  ## Use it after `local_tile`:
  ##   the threadgroup first takes its tile,
  ##   then each thread inside takes its slice.
  ##
  ## Contract:
  ## - identical to `outer_partition`, the tiler from the thread
  ##   layout's shape, the coord from `idx2crd`
  ## - `idx` must be inside the thread layout, out of range ids pick
  ##   tiles past the tensor
  outer_partition(tv, product_each(tile.shape), idx2crd(tile, idx))

template local_partition*(tv: AnyTensor; tile: Layout; idx: int or Int; proj: typed): untyped =
  ## 4-arg local_partition with projection, strips unwanted dimensions before partitioning.
  ## Contract: the projection marks the dimensions kept, the rest drops
  ## out of the tile layout before the 3-arg dispatch.
  ##
  ## CuTe: local_partition(tensor, tile, index, proj) =
  ##   local_partition(tensor, dice(proj, tile), index)
  local_partition(tv, dice(tile, proj), idx)

# ═════════════════════════════════════════════════════════════════════════
#  displace
# ═════════════════════════════════════════════════════════════════════════

func displace*[T, Sh, St](t: TensorView[T, Sh, St]; coord: IntOrIntTuple): auto {.inline, noInit.} =
  ## Offset TensorView by `coord` (logical coords).
  ## Returns: a sub-view with shape `original_shape - coord` (element-wise),
  ## the data pointer advanced by `crd2idx(layout, coord)`, strides preserved.
  let off = crd2idx(t.layout, coord)
  let ns = zipLeavesWith(t.layout.shape, coord):
    it_a - it_b
  make_view(t.data +% off, make_layout(ns, t.layout.stride))

func displace*[T, Sh, St](t: TensorOwned[T, Sh, St]; coord: IntOrIntTuple): auto {.inline, noInit.} =
  ## Offset TensorOwned by `coord` (logical coords). Returns a sub-view whose shape is
  ## `original_shape - coord` (element-wise).
  displace(t.view(), coord)
