## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ./int_tuples
import ./layouts

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

func make_view*[T, Sh, St](ptr_data: ptr UncheckedArray[T]; L: Layout[Sh, St]): TensorView[T, Sh, St] =
  TensorView[T, Sh, St](data: cast[ptr UncheckedArray[T]](ptr_data), layout: L)

func make_view*[T, Sh, St](data: var seq[T]; L: Layout[Sh, St]): TensorView[T, Sh, St] =
  TensorView[T, Sh, St](data: cast[ptr UncheckedArray[T]](addr data[0]), layout: L)

template make_view*(data: ptr UncheckedArray; L: Layout): untyped =
  type ElemType = type(data[])
  type Sh = typeof(L.shape)
  type St = typeof(L.stride)
  TensorView[ElemType, Sh, St](
    data: cast[ptr UncheckedArray[ElemType]](data),
    layout: L)

func make_view*[T, Sh, St](data: ptr T; L: Layout[Sh, St]): TensorView[T, Sh, St] =
  make_view(cast[ptr UncheckedArray[T]](data), L)

# ─────────────────────────────────────────────────────────────────────────
#  make_view(shape, stride tuples) — convenience, delegates to make_view(Layout)
# ─────────────────────────────────────────────────────────────────────────

template make_view*[T](ptr_data: ptr UncheckedArray[T]; shape, stride: IntOrIntTuple): untyped =
  make_view(ptr_data, make_layout(shape, stride))

template make_view*[T](data: openArray[T]; shape, stride: IntOrIntTuple): untyped =
  make_view(addr data[0], make_layout(shape, stride))

template make_view*[T](data: ptr T; shape, stride: IntOrIntTuple): untyped =
  make_view(data, make_layout(shape, stride))

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
#  Linear indexing — single int
# ═════════════════════════════════════════════════════════════════════════

template `[]`*[T, Sh, St](t: Tensor[T, Sh, St]; idx: int): untyped =
  t.data[t.offset + t.layout[idx]]

template `[]=`*[T, Sh, St](t: Tensor[T, Sh, St]; idx: int; val: T) =
  t.data[t.offset + t.layout[idx]] = val

template `[]`*[T, Sh, St](tv: TensorView[T, Sh, St]; idx: int): untyped =
  tv.data[tv.layout[idx]]

template `[]=`*[T, Sh, St](tv: TensorView[T, Sh, St]; idx: int; val: T) =
  tv.data[tv.layout[idx]] = val

# ═════════════════════════════════════════════════════════════════════════
#  Multi-index — operator()
#  Matches CuTe: Tensor::operator()(Coord const& coord)
#  Template: t((i, j)) — tuple coordinate of any arity.
#  Macro:   t(i, j, k) — individual ints packed into tuple.
# ═════════════════════════════════════════════════════════════════════════

template `()`*[T, Sh, St](t: Tensor[T, Sh, St]; coord: tuple): untyped =
  t.data[t.offset + t.layout[coord]]

template `()`*[T, Sh, St](tv: TensorView[T, Sh, St]; coord: tuple): untyped =
  tv.data[tv.layout[coord]]

macro `()`*(t: Tensor; args: varargs[int]): untyped =
  let coord = nnkPar.newTree()
  args.copyChildrenTo(coord)
  result = nnkBracketExpr.newTree(newDotExpr(t, ident"data"),
    newCall(bindSym"+", newDotExpr(t, ident"offset"),
      nnkBracketExpr.newTree(newDotExpr(t, ident"layout"), coord)))

macro `()`*(t: TensorView; args: varargs[int]): untyped =
  let coord = nnkPar.newTree()
  args.copyChildrenTo(coord)
  result = nnkBracketExpr.newTree(newDotExpr(t, ident"data"),
    nnkBracketExpr.newTree(newDotExpr(t, ident"layout"), coord))

# ═════════════════════════════════════════════════════════════════════════
#  slice — subtensor via Joker
# ═════════════════════════════════════════════════════════════════════════


template slice*[T, Sh, St](t: Tensor[T, Sh, St]; coord: untyped): untyped =
  let off = crd2idx(coord, t.layout)
  let sh = slice(coord, t.layout.shape)
  let st = slice(coord, t.layout.stride)
  make_view(cast[ptr UncheckedArray[T]](addr(t.data[t.offset + off])),
            make_layout(sh, st))

template slice*[T, Sh, St](tv: TensorView[T, Sh, St]; coord: untyped): untyped =
  let off = crd2idx(coord, tv.layout)
  let sh = slice(coord, tv.layout.shape)
  let st = slice(coord, tv.layout.stride)
  make_view(cast[ptr UncheckedArray[T]](cast[int](tv.data) +% off *% sizeof(T).int),
            make_layout(sh, st))

# ═════════════════════════════════════════════════════════════════════════
#  copyFrom / copyFromIf — flat-index element copy primitives
# ═════════════════════════════════════════════════════════════════════════

template copyFrom*[T, ShA, StA, ShB, StB](
    dst: var TensorView[T, ShB, StB];
    src: TensorView[T, ShA, StA]) =
  for i in 0 ..< size(dst):
    dst(i) = src(i)

template copyFromIf*[T, ShA, StA, ShB, StB](
    dst: var TensorView[T, ShB, StB];
    src: TensorView[T, ShA, StA];
    predicate: typed;
    defaultVal: T) =
  for i in 0 ..< size(dst):
    if predicate(i):
      dst(i) = src(i)
    else:
      dst(i) = defaultVal

# ═════════════════════════════════════════════════════════════════════════
#  local_tile — extract subtensor from tiled_divide result
# ═════════════════════════════════════════════════════════════════════════

func local_tile*[T, Sh, St, Ti, Si](
    tv: TensorView[T, Sh, St];
    tiled: Layout[Ti, Si];
    a, b: int): auto =
  ## tiled_divide produces shape ((tileM, tileK), mP, kP) — tile nested, rest flat.
  ## Use nested coordinate ((_, _), a, b) to keep tile and collapse rest.
  let (sub, off) = slice_and_offset(((_, _), a, b), tiled)
  make_view(cast[ptr UncheckedArray[T]](
    cast[int](tv.data) +% off *% sizeof(T).int), sub)

# ═════════════════════════════════════════════════════════════════════════
#  Display
# ═════════════════════════════════════════════════════════════════════════

proc `$`*[T, Sh, St](t: Tensor[T, Sh, St]): string =
  "Tensor[offset=" & $t.offset & "] o (" & $t.layout.shape & "):(" & $t.layout.stride & ")"

proc `$`*[T, Sh, St](tv: TensorView[T, Sh, St]): string =
  "TensorView o (" & $tv.layout.shape & "):(" & $tv.layout.stride & ")"
