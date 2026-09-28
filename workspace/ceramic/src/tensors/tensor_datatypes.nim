## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra/layouts
import workspace/ceramic/src/layout_algebra/ptr_arithmetic

# ═════════════════════════════════════════════════════════════════════════
#  TensorOwned / TensorView
# ═════════════════════════════════════════════════════════════════════════

type
  TensorOwned*[T, Sh, St] = object
    ## Owning tensor — stack-allocated array. No heap, no seq.
    ## Requires static shape/stride (compile-time cosize).
    data*: array[cosize(Layout[Sh, St]), T]
    layout*: Layout[Sh, St]

  TensorView*[T, Sh, St] = object
    ## Non-owning tensor — points to external memory.
    data*: ptr UncheckedArray[T]
    layout*: Layout[Sh, St]

type AnyTensor*[T, Sh, St] = TensorView[T, Sh, St] or TensorOwned[T, Sh, St]
  ## Any tensor, either the owning `TensorOwned` or the non-owning
  ## `TensorView`, with matching element type and matching shape/stride
  ## tuple types. Shorthand for the union used in kernel signatures:
  ##
  ## - `TensorView[...]`, non-owning
  ## - `TensorOwned[...]`, owning
  ##
  ## Fragments and gmem operands arrive as either.

# ═════════════════════════════════════════════════════════════════════════
#  Constructors
# ═════════════════════════════════════════════════════════════════════════

# ── Owning: make_tensor(T, Layout) ─────────────────────────────────────

func make_tensor*[Sh, St, T](_: typedesc[T]; L: Layout[Sh, St]): TensorOwned[T, Sh, St] {.inline.} =
  ## Owning tensor — stack array, no heap. Requires static cosize.
  TensorOwned[T, Sh, St](layout: L)

template make_tensor*[T](_: typedesc[T]; shape: IntOrIntTuple;
                         order: static StrideOrder = LayoutLeft): untyped =
  make_tensor(T, make_layout(shape, order))

template make_tensor*[T](_: typedesc[T]; shape, stride: IntOrIntTuple): untyped =
  make_tensor(T, make_layout(shape, stride))

# ── make_tensor_like — create owning tensor with compact strides ──────

func make_tensor_like*[T, Sh, St](t: TensorView[T, Sh, St]): auto {.inline.} =
  make_tensor(T, make_layout_like(t.layout))

func make_tensor_like*[T, Sh, St](t: TensorOwned[T, Sh, St]): auto {.inline.} =
  ## Creates a compact-stride owning tensor matching the input's shape
  ## and element type.
  make_tensor(T, make_layout_like(t.layout))

func make_tensor_like*[T, Sh, St, NewT](t: TensorView[T, Sh, St]; _: typedesc[NewT]): auto {.inline.} =
  make_tensor(NewT, make_layout_like(t.layout))

func make_tensor_like*[T, Sh, St, NewT](t: TensorOwned[T, Sh, St]; _: typedesc[NewT]): auto {.inline.} =
  ## Creates a compact-stride owning tensor matching the input's shape,
  ## with element type NewT.
  make_tensor(NewT, make_layout_like(t.layout))


# ── Non-owning: make_view(ptr, Layout) ─────────────────────────────────

func make_view*[T, Sh, St](data: ptr UncheckedArray[T] or ptr T;
                           L: Layout[Sh, St]): TensorView[T, Sh, St] {.inline.} =
  TensorView[T, Sh, St](data: cast[ptr UncheckedArray[T]](data), layout: L)

template make_view*[T](data: ptr UncheckedArray[T] or ptr T;
                       shape: IntOrIntTuple;
                       order: static StrideOrder = LayoutLeft): untyped =
  make_view(data, make_layout(shape, order))

template make_view*[P: ptr | ptr UncheckedArray, ShT, StT: IntOrIntTuple](
    data: P; shape: ShT; stride: StT): untyped =
  make_view(data, make_layout(shape, stride))


# ── Non-owning: make_view(openArray, Layout) — zero-copy ───────────────

func make_view*[T, Sh, St](data: openArray[T];
                           L: Layout[Sh, St]): TensorView[T, Sh, St] {.inline.} =
  make_view(cast[ptr UncheckedArray[T]](addr data[0]), L)

template make_view*[T](data: openArray[T];
                       shape: IntOrIntTuple;
                       order: static StrideOrder = LayoutLeft): untyped =
  make_view(data, make_layout(shape, order))

template make_view*[T](data: openArray[T];
                       shape, stride: IntOrIntTuple): untyped =
  make_view(data, make_layout(shape, stride))

# ── Non-owning: make_view(TensorView, Layout) — reinterpret ────────────

func make_view*[T, ShA, StA, ShB, StB](
    tv: TensorView[T, ShA, StA];
    L: Layout[ShB, StB]): TensorView[T, ShB, StB] {.inline.} =
  ## Reinterpret a view with a new layout (same data pointer).
  TensorView[T, ShB, StB](data: tv.data, layout: L)

template make_view*(tv: TensorView;
                       shape: IntOrIntTuple;
                       order: static StrideOrder = LayoutLeft): untyped =
  make_view(tv, make_layout(shape, order))

template make_view*(tv: TensorView;
                       shape, stride: IntOrIntTuple): untyped =
  make_view(tv, make_layout(shape, stride))

# ═════════════════════════════════════════════════════════════════════════
#  view() — TensorOwned → TensorView
# ═════════════════════════════════════════════════════════════════════════

func view*[T, Sh, St](t: TensorOwned[T, Sh, St]): TensorView[T, Sh, St] {.inline.} =
  ## Non-owning view sharing the owning tensor's memory and layout.
  TensorView[T, Sh, St](
    data: cast[ptr UncheckedArray[T]](addr t.data[0]),
    layout: t.layout)

# ═════════════════════════════════════════════════════════════════════════
#  Layout accessors
# ═════════════════════════════════════════════════════════════════════════

template layout*(t: TensorOwned): untyped =
  ## Reads the layout of the owning tensor's layout.
  t.layout
template layout*(tv: TensorView): untyped = tv.layout

template shape*(t: TensorOwned): untyped =
  ## Reads the shape of the owning tensor's layout.
  t.layout.shape
template shape*(tv: TensorView): untyped = tv.layout.shape

template stride*(t: TensorOwned): untyped =
  ## Reads the stride of the owning tensor's layout.
  t.layout.stride
template stride*(tv: TensorView): untyped = tv.layout.stride

template rank*(tv: TensorView): untyped = tv.layout.rank()
template rank*(t: TensorOwned): untyped =
  ## Reads the rank of the owning tensor's layout.
  t.layout.rank()

template size*(tv: TensorView): untyped = tv.layout.size()
template size*(t: TensorOwned): untyped =
  ## Reads the size of the owning tensor's layout.
  t.layout.size()

template cosize*(tv: TensorView): untyped = tv.layout.cosize()
template cosize*(t: TensorOwned): untyped =
  ## Reads the cosize of the owning tensor's layout.
  t.layout.cosize()

# ═════════════════════════════════════════════════════════════════════════
#  Display
# ═════════════════════════════════════════════════════════════════════════

proc `$`*[T, Sh, St](t: TensorOwned[T, Sh, St]): string =
  "TensorOwned o (" & $t.layout.shape & "):(" & $t.layout.stride & ")"

proc `$`*[T, Sh, St](tv: TensorView[T, Sh, St]): string =
  "TensorView o (" & $tv.layout.shape & "):(" & $tv.layout.stride & ")"
