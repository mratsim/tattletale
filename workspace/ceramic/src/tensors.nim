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

func make_view*[T, ShA, StA, ShB, StB](
    tv: TensorView[T, ShA, StA];
    L: Layout[ShB, StB]): TensorView[T, ShB, StB] =
  ## Reinterpret a TensorView with a new layout (same data pointer).
  TensorView[T, ShB, StB](data: tv.data, layout: L)

# ─────────────────────────────────────────────────────────────────────────
#  make_view(shape, stride tuples) — convenience, delegates to make_view(Layout)
# ─────────────────────────────────────────────────────────────────────────

template make_view*[T](ptr_data: ptr UncheckedArray[T]; shape, stride: IntOrIntTuple): untyped =
  make_view(ptr_data, make_layout(shape, stride))

template make_view*[T](data: openArray[T]; shape, stride: IntOrIntTuple): untyped =
  make_view(addr data[0], make_layout(shape, stride))

template make_view*[T](data: ptr T; shape, stride: IntOrIntTuple): untyped =
  make_view(data, make_layout(shape, stride))

template make_view*[T, ShA, StA](tv: TensorView[T, ShA, StA]; shape, stride: IntOrIntTuple): untyped =
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
#  Flat indexing — single int via parens (macro enables call from templates)
# ═════════════════════════════════════════════════════════════════════════

macro `()`*(t: Tensor; args: varargs[int]): untyped =
  var coord = nnkPar.newTree()
  args.copyChildrenTo(coord)
  result = nnkBracketExpr.newTree(newDotExpr(t, ident"data"),
    newCall(bindSym"+", newDotExpr(t, ident"offset"),
      newCall(bindSym"crd2idx", newDotExpr(t, ident"layout"), coord)))

macro `()`*(tv: TensorView; args: varargs[int]): untyped =
  var coord = nnkPar.newTree()
  args.copyChildrenTo(coord)
  result = nnkBracketExpr.newTree(newDotExpr(tv, ident"data"),
    newCall(bindSym"crd2idx", newDotExpr(tv, ident"layout"), coord))

# ═════════════════════════════════════════════════════════════════════════
#  Multi-index — operator[]
#  Matches numpy/PyTorch: t[i, j] — varargs in brackets.
#  Also accepts tuple coordinates: t[(i, j), (k, l)]
# ═════════════════════════════════════════════════════════════════════════

template `[]`*[T, Sh, St](t: Tensor[T, Sh, St]; coord: tuple): untyped =
  t.data[t.offset + t.layout(coord)]

template `[]`*[T, Sh, St](tv: TensorView[T, Sh, St]; coord: tuple): untyped =
  tv.data[tv.layout(coord)]

macro `[]`*(t: Tensor; args: varargs[IntOrIntTuple]): untyped =
  var coord = nnkPar.newTree()
  args.copyChildrenTo(coord)
  result = nnkBracketExpr.newTree(newDotExpr(t, ident"data"),
    newCall(bindSym"+", newDotExpr(t, ident"offset"),
      newCall(bindSym"crd2idx", newDotExpr(t, ident"layout"), coord)))

macro `[]`*(t: TensorView; args: varargs[IntOrIntTuple]): untyped =
  var coord = nnkPar.newTree()
  args.copyChildrenTo(coord)
  result = nnkBracketExpr.newTree(newDotExpr(t, ident"data"),
    newCall(bindSym"crd2idx", newDotExpr(t, ident"layout"), coord))

# ═════════════════════════════════════════════════════════════════════════
#  slice — subtensor via Joker
# ═════════════════════════════════════════════════════════════════════════


template slice*[T, Sh, St](t: Tensor[T, Sh, St]; coord: untyped): untyped =
  ## Extract a sub-tensor positioned at the given coordinate.
  ##
  ## For each mode of the tensor's layout:
  ##   - coord has `_` → keep that mode in the result
  ##   - coord has int → collapse that mode (data pointer advances by int×stride)
  ##
  ## Internally calls `slice` on shape+stride and offsets the data pointer
  ## by `crd2idx(layout, coord)`.
  ##
  ## runnableExamples:
  ##   let t = make_tensor(make_layout((3, 4), (1, 3)), float32)
  ##   let col1 = t.slice((_, 1))   # rows × column-1 → (3):(1)
  ##   doAssert $col1.layout == "(3,):(1,)"
  let off = crd2idx(t.layout, coord)
  let sh = slice(coord, t.layout.shape)
  let st = slice(coord, t.layout.stride)
  make_view(cast[ptr UncheckedArray[T]](addr(t.data[t.offset + off])),
            make_layout(sh, st))

template slice*[T, Sh, St](tv: TensorView[T, Sh, St]; coord: untyped): untyped =
  ## Extract a sub-view positioned at the given coordinate.
  ## Same semantics as Tensor.slice — offsets the data pointer by the
  ## int-paired dimensions' contribution.
  ##
  ## runnableExamples:
  ##   var buf: array[12, float32]
  ##   let v = make_view(addr(buf[0]), make_layout((3, 4), (1, 3)))
  ##   let col1 = v.slice((_, 1))   # rows × column-1 → (3):(1)
  ##   doAssert $col1.layout == "(3,):(1,)"
  let off = crd2idx(tv.layout, coord)
  let sh = slice(coord, tv.layout.shape)
  let st = slice(coord, tv.layout.stride)
  make_view(cast[ptr UncheckedArray[T]](cast[int](tv.data) +% off *% sizeof(T).int),
            make_layout(sh, st))

# ═════════════════════════════════════════════════════════════════════════
#  fill — set every logical element to a constant
# ═════════════════════════════════════════════════════════════════════════
##
##  CuTe: fill(tensor, value) — flat loop over tensor's logical domain.
##  The layout remaps the flat index to memory, handling strides,
##  broadcasts (stride-0), and arbitrary nesting.
##
template fillWith*[T, Sh, St](tv: var TensorView[T, Sh, St]; val: T) =
  ## Set every logical element of `tv` to `val`.
  let n = size(tv.layout)
  for i in 0 ..< n:
    tv(i) = val

template fillWith*[T, Sh, St](t: var Tensor[T, Sh, St]; val: T) =
  ## Set every logical element of `t` to `val`.
  let n = size(t.layout)
  for i in 0 ..< n:
    t(i) = val

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
