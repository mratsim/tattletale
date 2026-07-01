## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros

import ./macros/varargs_to_par
import ./ptr_arithmetic
import ./int_tuples
import ./layouts
import ./layout_indexing_gpu
import ./layout_indexing
import ./tensor_datatypes

export layout_indexing_gpu
export layout_indexing
export tensor_datatypes

proc pop(tree: var NimNode): NimNode {.compileTime.} =
  ## varargs[untyped] consumes all arguments, so []= pops the val
  ## https://github.com/nim-lang/Nim/issues/5855
  result = tree[tree.len-1]
  tree.del(tree.len-1)

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  `()` — dual dispatch: all-int → element, has _ → sub-View
# ═════════════════════════════════════════════════════════════════════════

template `()`*(t: Tensor; args: varargs[untyped]): untyped =
  when hasUnderscore(args):
    block:
      evalOnceAs(coord, varargs_to_par(args))
      evalOnceAs(sub, slice(t.layout, coord))
      evalOnceAs(offset, crd2idx(t.layout, coord))
      make_view(t.data[0].addr +% toIntVal(offset), sub)
  else:
    # We can't wrap the whole expression into a block or it isn't a lvalue
    # and so can't be assigned to.
    # At the same time, coord MUST be wrapped, or we have scoping and name collision issues.
    {.warning: "Assignment through `()` is discouraged, use `[]=` instead".}
    let pos = block:
      evalOnceAs(coord, varargs_to_par(args))
      crd2idx(t.layout, coord)
    t.data[toIntVal pos]

template `()`*(tv: TensorView; args: varargs[untyped]): untyped =
  when hasUnderscore(args):
    block:
      evalOnceAs(coord, varargs_to_par(args))
      evalOnceAs(sub, slice(tv.layout, coord))
      evalOnceAs(offset, crd2idx(tv.layout, coord))
      make_view(tv.data +% toIntVal(offset), sub)
  else:
    # We can't wrap the whole expression into a block or it isn't a lvalue
    # and so can't be assigned to.
    # At the same time, coord MUST be wrapped, or we have scoping and name collision issues.
    {.warning: "Assignment through `()` is discouraged, use `[]=` instead".}
    let pos = block:
      evalOnceAs(coord, varargs_to_par(args))
      crd2idx(tv.layout, coord)
    tv.data[toIntVal pos]

# ═════════════════════════════════════════════════════════════════════════
#  `[]` — element access only (underscore rejected)
# ═════════════════════════════════════════════════════════════════════════

template `[]`*(t: Tensor; args: varargs[untyped]): untyped =
  let pos = block:
    evalOnceAs(coord, varargs_to_par(args))
    when hasUnderscoreImpl(coord):
      {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
    toIntVal crd2idx(t.layout, coord)
  t.data[pos]

template `[]`*(tv: TensorView; args: varargs[untyped]): untyped =
  let pos = block:
    evalOnceAs(coord, varargs_to_par(args))
    when hasUnderscoreImpl(coord):
      {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
    toIntVal crd2idx(tv.layout, coord)
  tv.data[pos]

macro `[]=`*(t: Tensor; args: varargs[untyped]): untyped =
  var a = args
  let val = pop(a)
  let coord = getAST(varargs_to_par(a))
  result = quote do:
    when hasUnderscoreImpl(`coord`):
      {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
    else:
      `t`.data[toIntVal crd2idx(`t`.layout, `coord`)] = `val`

macro `[]=`*(tv: TensorView; args: varargs[untyped]): untyped =
  var a = args
  let val = pop(a)
  let coord = getAST(varargs_to_par(a))
  result = quote do:
    when hasUnderscoreImpl(`coord`):
      {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
    else:
      `tv`.data[toIntVal crd2idx(`tv`.layout, `coord`)] = `val`

# ═════════════════════════════════════════════════════════════════════════
#  slice — subtensor via underscore dispatch
# ═════════════════════════════════════════════════════════════════════════

template slice*(t: Tensor; coords: varargs[untyped]): untyped =
  block:
    evalOnceAs(crd, varargs_to_par(coords))
    evalOnceAs(sub, slice(t.layout, crd))
    let off = crd2idx(t.layout, crd)
    make_view(t.data[0].addr +% off.toIntVal(), sub)

template slice*(tv: TensorView; coords: varargs[untyped]): untyped =
  block:
    evalOnceAs(crd, varargs_to_par(coords))
    evalOnceAs(sub, slice(tv.layout, crd))
    let off = crd2idx(tv.layout, crd)
    make_view(tv.data +% off.toIntVal(), sub)

# ═════════════════════════════════════════════════════════════════════════
#  repeatTuple
# ═════════════════════════════════════════════════════════════════════════

macro repeat(elem: typed, n: static int): untyped =
  result = nnkTupleConstr.newTree()
  for i in 0 ..< n:
    result.add elem

# ═════════════════════════════════════════════════════════════════════════
#  inner_partition / outer_partition / local_tile / local_partition
#  CuTe: tensor_impl.hpp — zipped_divide + slice_and_offset
# ═════════════════════════════════════════════════════════════════════════

template inner_partition*(tv: TensorView or Tensor; tiler: typed; coord: typed): untyped =
  ## Keep tile modes, slice rest modes with coord.
  ## CuTe: zipped_divide(tensor, tiler)(repeat<R0>(_), append<R1>(coord, _))
  block:
    evalOnceAs zd, zipped_divide(tv.layout, tiler)
    when coord is tuple:
      evalOnceAs c, coord
      evalOnceAs keptRest, make_layout(slice(zd.shape[1], c), slice(zd.stride[1], c))
      evalOnceAs offset, crd2idx(c, zd.shape[1], zd.stride[1])
      evalOnceAs subLayout, make_layout(concat(zd.shape[0], keptRest.shape), concat(zd.stride[0], keptRest.stride))
      make_view(tv.data +% toIntVal(offset), subLayout)
    else:
      evalOnceAs offset, crd2idx(coord, zd.shape[1], zd.stride[1])
      evalOnceAs subLayout, make_layout(zd.shape[0], zd.stride[0])
      make_view(tv.data +% toIntVal(offset), subLayout)

template outer_partition*(tv: TensorView or Tensor; tiler: typed; coord: typed): untyped =
  ## Slice tile modes with coord, keep rest modes.
  ## CuTe: zipped_divide(tensor, tiler)(append<R0>(coord, _), repeat<R1>(_))
  block:
    evalOnceAs zd, zipped_divide(tv.layout, tiler)
    when coord is tuple:
      evalOnceAs c, coord
      evalOnceAs keptTile, make_layout(slice(zd.shape[0], c), slice(zd.stride[0], c))
      evalOnceAs offset, crd2idx(c, zd.shape[0], zd.stride[0])
      evalOnceAs subLayout, make_layout(concat(keptTile.shape, zd.shape[1]), concat(keptTile.stride, zd.stride[1]))
      make_view(tv.data +% toIntVal(offset), subLayout)
    else:
      evalOnceAs offset, crd2idx(coord, zd.shape[0], zd.stride[0])
      evalOnceAs subLayout, make_layout(zd.shape[1], zd.stride[1])
      make_view(tv.data +% toIntVal(offset), subLayout)

template local_tile*(tv: TensorView or Tensor; tiler: typed; coord: typed): untyped =
  ## Alias for inner_partition — select a single tile.
  ## CuTe: local_tile = inner_partition
  inner_partition(tv, tiler, coord)

template local_tile*(tv: TensorView or Tensor; tiler, coord, proj: typed): untyped =
  ## 4-arg local_tile with projection — strips unwanted modes before partitioning.
  ## CuTe: local_tile(tensor, tiler, coord, proj) =
  ##   local_tile(tensor, dice(proj, tiler), dice(proj, coord))
  block:
    evalOnceAs t, tiler
    evalOnceAs c, coord
    evalOnceAs pt, dice(t, proj)
    evalOnceAs pc, dice(c, proj)
    local_tile(tv, pt, pc)

template local_partition*(tv: TensorView or Tensor; tile: Layout; idx: int or Int): untyped =
  ## 3-arg local_partition — select tile by index within a thread layout.
  ## CuTe: local_partition = outer_partition with product_each(tile.shape)
  block:
    evalOnceAs thrLayout, tile
    evalOnceAs tiler, product_each(thrLayout.shape)
    evalOnceAs coord, idx2crd(thrLayout, idx)
    outer_partition(tv, tiler, coord)

template local_partition*(tv: TensorView or Tensor; tile: Layout; idx: int or Int; proj: typed): untyped =
  ## 4-arg local_partition with projection — strip unwanted modes before partitioning.
  ## CuTe: local_partition(tensor, tile, index, proj) =
  ##   local_partition(tensor, dice(proj, tile), index)
  block:
    evalOnceAs thrLayout, tile
    evalOnceAs projected, dice(thrLayout, proj)
    local_partition(tv, projected, idx)

# ═════════════════════════════════════════════════════════════════════════
#  displace
# ═════════════════════════════════════════════════════════════════════════

func displace*[T, Sh, St](t: TensorView[T, Sh, St]; coord: IntOrIntTuple): auto {.inline, noInit.} =
  ## Offset TensorView by `coord` (logical coords). Returns a sub-view whose shape is
  ## `original_shape - coord` (element-wise). Data pointer advances by
  ## `crd2idx(layout, coord)`. Strides preserved.
  let off = crd2idx(t.layout, coord)
  let ns = zipLeavesWith(t.layout.shape, coord):
    it_a - it_b
  make_view(t.data +% off, make_layout(ns, t.layout.stride))

func displace*[T, Sh, St](t: Tensor[T, Sh, St]; coord: IntOrIntTuple): auto {.inline, noInit.} =
  ## Offset Tensor by `coord` (logical coords). Returns a sub-view whose shape is
  ## `original_shape - coord` (element-wise).
  displace(t.view(), coord)
