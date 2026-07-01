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

export layout_indexing_gpu
export layout_indexing

proc pop(tree: var NimNode): NimNode {.compileTime.} =
  ## varargs[untyped] consumes all arguments, so []= pops the val
  ## https://github.com/nim-lang/Nim/issues/5855
  result = tree[tree.len-1]
  tree.del(tree.len-1)

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  Tensor / TensorView
# ═════════════════════════════════════════════════════════════════════════

type
  Tensor*[T, Sh, St] = object
    data*: seq[T]
    offset*: int
    layout*: Layout[Sh, St]

  TensorView*[T, Sh, St] = object
    data*: ptr UncheckedArray[T]
    layout*: Layout[Sh, St]

# ═════════════════════════════════════════════════════════════════════════
#  Constructors
# ═════════════════════════════════════════════════════════════════════════

func make_tensor*[Sh, St, T](L: Layout[Sh, St]; _: typedesc[T]): Tensor[T, Sh, St] =
  Tensor[T, Sh, St](data: newSeq[T](cosize(L).toIntVal()), offset: 0, layout: L)

func make_tensor*[T, Sh, St](data: openArray[T]; off: int; L: Layout[Sh, St]): Tensor[T, Sh, St] =
  Tensor[T, Sh, St](data: @data, offset: off, layout: L)

template make_tensor*(shape, stride: IntOrIntTuple; T: typedesc): untyped =
  make_tensor(make_layout(shape, stride), T)

template make_tensor*(data: openArray; off: int; shape, stride: IntOrIntTuple): untyped =
  make_tensor(data, off, make_layout(shape, stride))

template make_tensor*(shape: IntOrIntTuple; order: static StrideOrder = LayoutLeft; T: typedesc): untyped =
  make_tensor(make_layout(shape, order), T)

template make_tensor*(data: openArray; off: int; shape: IntOrIntTuple; order: static StrideOrder = LayoutLeft): untyped =
  make_tensor(data, off, make_layout(shape, order))

# ─────────────────────────────────────────────────────────────────────────
#  make_view(Layout)
# ─────────────────────────────────────────────────────────────────────────

func make_view*[T, Sh, St](ptr_data: ptr UncheckedArray[T]; L: Layout[Sh, St]): TensorView[T, Sh, St] {.inline, noInit.} =
  TensorView[T, Sh, St](data: cast[ptr UncheckedArray[T]](ptr_data), layout: L)

func make_view*[T, Sh, St](data: var seq[T]; L: Layout[Sh, St]): TensorView[T, Sh, St] {.inline, noInit.} =
  TensorView[T, Sh, St](data: cast[ptr UncheckedArray[T]](addr data[0]), layout: L)

template make_view*(data: ptr UncheckedArray; L: Layout): untyped =
  type ElemType = type(data[])
  type Sh = typeof(L.shape)
  type St = typeof(L.stride)
  TensorView[ElemType, Sh, St](
    data: cast[ptr UncheckedArray[ElemType]](data),
    layout: L)

func make_view*[T, Sh, St](data: ptr T; L: Layout[Sh, St]): TensorView[T, Sh, St] {.inline, noInit.} =
  make_view(cast[ptr UncheckedArray[T]](data), L)

func make_view*[T, ShA, StA, ShB, StB](
    tv: TensorView[T, ShA, StA];
    L: Layout[ShB, StB]): TensorView[T, ShB, StB] =
  ## Reinterpret a TensorView with a new layout (same data pointer).
  TensorView[T, ShB, StB](data: tv.data, layout: L)

# ─────────────────────────────────────────────────────────────────────────
#  make_view(shape, stride tuples) — convenience, delegates to make_view(Layout)
# ─────────────────────────────────────────────────────────────────────────

template make_view*[T; Sh, St: IntOrIntTuple](ptr_data: ptr UncheckedArray[T]; shape: Sh, stride: St): untyped =
  make_view(ptr_data, make_layout(shape, stride))

template make_view*[T; Sh, St: IntOrIntTuple](data: openArray[T]; shape: Sh, stride: St): untyped =
  make_view(addr data[0], make_layout(shape, stride))

template make_view*[T; Sh, St: IntOrIntTuple](data: ptr T; shape: Sh, stride: St): untyped =
  make_view(data, make_layout(shape, stride))

template make_view*[T; ShA, StA, Sh, St](tv: TensorView[T, ShA, StA]; shape: Sh, stride: St): untyped =
  ## Reinterpret a TensorView with new shape and stride (same data pointer).
  make_view(tv, make_layout(shape, stride))

# shape-only overloads — natural col-major strides
template make_view*[T](ptr_data: ptr UncheckedArray[T]; shape: IntOrIntTuple; order: static StrideOrder = LayoutLeft): untyped =
  make_view(ptr_data, make_layout(shape, order))

template make_view*[T](data: var seq[T]; shape: IntOrIntTuple; order: static StrideOrder = LayoutLeft): untyped =
  make_view(data, make_layout(shape, order))

template make_view*[T](data: openArray[T]; shape: IntOrIntTuple; order: static StrideOrder = LayoutLeft): untyped =
  make_view(addr data[0], make_layout(shape, order))

template make_view*[T](data: ptr T; shape: IntOrIntTuple; order: static StrideOrder = LayoutLeft): untyped =
  make_view(data, make_layout(shape, order))

# ─────────────────────────────────────────────────────────────────────────
#  view(owning Tensor → TensorView)
# ─────────────────────────────────────────────────────────────────────────

func view*[T, Sh, St](t: Tensor[T, Sh, St]): TensorView[T, Sh, St] =
  TensorView[T, Sh, St](
    data: cast[ptr UncheckedArray[T]](addr(t.data[t.offset])),
    layout: t.layout)

# ═════════════════════════════════════════════════════════════════════════
#  Accessors
# ═════════════════════════════════════════════════════════════════════════

template layout*(t: Tensor): untyped = t.layout
template layout*(tv: TensorView): untyped = tv.layout

template shape*(t: Tensor): untyped = t.layout.shape
template shape*(tv: TensorView): untyped = tv.layout.shape

template stride*(t: Tensor): untyped = t.layout.stride
template stride*(tv: TensorView): untyped = tv.layout.stride

template rank*(tv: TensorView): untyped = tv.layout.rank()
template rank*(t: Tensor): untyped = t.layout.rank()

template size*(tv: TensorView): untyped = tv.layout.size()
template size*(t: Tensor): untyped = t.layout.size()

template cosize*(tv: TensorView): untyped = tv.layout.cosize()
template cosize*(t: Tensor): untyped = t.layout.cosize()

# ═════════════════════════════════════════════════════════════════════════
#  `()` — dual dispatch: all-int → element, has _ → sub-View
# ═════════════════════════════════════════════════════════════════════════

template `()`*(t: Tensor; args: varargs[untyped]): untyped =
  when hasUnderscore(args):
    block:
      evalOnceAs(coord, varargs_to_par(args))
      evalOnceAs(sub, slice(t.layout, coord))
      evalOnceAs(offset, crd2idx(t.layout, coord))
      make_view(t.data[0].addr +% t.offset +% toIntVal(offset), sub)
  else:
    # We can't wrap the whole expression into a block or it isn't a lvalue
    # and so can't be assigned to.
    # At the same time, coord MUST be wrapped, or we have scoping and name collision issues.
    {.warning: "Assignment through `()` is discouraged, use `[]=` instead".}
    let pos = block:
      evalOnceAs(coord, varargs_to_par(args))
      t.offset + crd2idx(t.layout, coord)
    t.data[pos]

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
  t.data[t.offset + pos]

macro `[]=`*(t: Tensor; args: varargs[untyped]): untyped =
  var a = args
  let val = pop(a)
  # getAST evaluates varargs_to_par at compile-time with `a`'s actual NimNode value
  let coord = getAST(varargs_to_par(a))
  result = quote do:
    when hasUnderscoreImpl(`coord`):
      {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
    else:
      `t`.data[`t`.offset + crd2idx(`t`.layout, `coord`)] = `val`

template `[]`*(tv: TensorView; args: varargs[untyped]): untyped =
  let pos = block:
    evalOnceAs(coord, varargs_to_par(args))
    when hasUnderscoreImpl(coord):
      {.fatal: "_ not allowed in operator[] — use operator() for sub-Views".}
    toIntVal crd2idx(tv.layout, coord)
  tv.data[pos]

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
    make_view(t.data[0].addr +% t.offset +% off.toIntVal(), sub)

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
#  inner_partition / outer_partition / local_tile
#  CuTe: tensor_impl.hpp — zipped_divide + slice_and_offset
# ═════════════════════════════════════════════════════════════════════════

template inner_partition*(tv: TensorView or Tensor; tiler: typed; coord: typed): untyped =
  ## Keep tile modes, slice rest modes with coord.
  ## CuTe: zipped_divide(tensor, tiler)(repeat<R0>(_), coord)
  block:
    evalOnceAs(zd, zipped_divide(tv.layout, tiler))
    # zipped_divide returns rank-2 Layout: ((tile_modes), (rest_modes))
    # Result layout = tile group (shape[0], stride[0])
    # Offset = crd2idx(coord, rest group shape, rest group stride)
    evalOnceAs(offset, crd2idx(coord, zd.shape[1], zd.stride[1]))
    evalOnceAs(subLayout, make_layout(zd.shape[0], zd.stride[0]))
    make_view(tv.data +% toIntVal(offset), subLayout)

template outer_partition*(tv: TensorView or Tensor; tiler: typed; coord: typed): untyped =
  ## Slice tile modes with coord, keep rest modes.
  ## CuTe: zipped_divide(tensor, tiler)(coord, repeat<R1>(_))
  block:
    evalOnceAs(zd, zipped_divide(tv.layout, tiler))
    # zipped_divide returns rank-2 Layout: ((tile_modes), (rest_modes))
    # Result layout = rest group (shape[1], stride[1])
    # Offset = crd2idx(coord, tile group shape, tile group stride)
    evalOnceAs(offset, crd2idx(coord, zd.shape[0], zd.stride[0]))
    evalOnceAs(subLayout, make_layout(zd.shape[1], zd.stride[1]))
    make_view(tv.data +% toIntVal(offset), subLayout)


template local_tile*(tv: TensorView or Tensor; tiler: typed; coord: typed): untyped =
  ## Alias for inner_partition — select a single tile.
  ## CuTe: local_tile = inner_partition
  inner_partition(tv, tiler, coord)

# ═════════════════════════════════════════════════════════════════════════
#  displace — offset a Tensor/TensorView, return sub-view with auto-deduced shape
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

# ═════════════════════════════════════════════════════════════════════════
#  Display
# ═════════════════════════════════════════════════════════════════════════

proc `$`*[T, Sh, St](t: Tensor[T, Sh, St]): string =
  "Tensor[offset=" & $t.offset & "] o (" & $t.layout.shape & "):(" & $t.layout.stride & ")"

proc `$`*[T, Sh, St](tv: TensorView[T, Sh, St]): string =
  "TensorView o (" & $tv.layout.shape & "):(" & $tv.layout.stride & ")"
