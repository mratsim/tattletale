## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./int_tuples
import ./layouts
import ./ptr_arithmetic

# ═════════════════════════════════════════════════════════════════════════
#  Tensor / TensorView
# ═════════════════════════════════════════════════════════════════════════

type
  Tensor*[T, Sh, St] = object
    ## Owning tensor — stack-allocated array. No heap, no seq.
    ## Requires static shape/stride (compile-time cosize).
    data*: array[cosize(Layout[Sh, St]), T]
    layout*: Layout[Sh, St]

  TensorView*[T, Sh, St] = object
    ## Non-owning tensor — points to external memory.
    data*: ptr UncheckedArray[T]
    layout*: Layout[Sh, St]

# ═════════════════════════════════════════════════════════════════════════
#  Constructors
# ═════════════════════════════════════════════════════════════════════════

# ── Owning: make_tensor(T, Layout) ─────────────────────────────────────

func make_tensor*[Sh, St, T](_: typedesc[T]; L: Layout[Sh, St]): Tensor[T, Sh, St] =
  ## Owning tensor — stack array, no heap. Requires static cosize.
  Tensor[T, Sh, St](layout: L)

template make_tensor*[T](_: typedesc[T]; shape: IntOrIntTuple;
                         order: static StrideOrder = LayoutLeft): untyped =
  make_tensor(T, make_layout(shape, order))

template make_tensor*[T](_: typedesc[T]; shape, stride: IntOrIntTuple): untyped =
  make_tensor(T, make_layout(shape, stride))

# ── make_tensor_like — create owning tensor with compact strides ──────

func make_tensor_like*[T, Sh, St](t: TensorView[T, Sh, St]): auto =
  make_tensor(T, make_layout_like(t.layout))

func make_tensor_like*[T, Sh, St](t: Tensor[T, Sh, St]): auto =
  make_tensor(T, make_layout_like(t.layout))

func make_tensor_like*[T, Sh, St, NewT](t: TensorView[T, Sh, St]; _: typedesc[NewT]): auto =
  make_tensor(NewT, make_layout_like(t.layout))

func make_tensor_like*[T, Sh, St, NewT](t: Tensor[T, Sh, St]; _: typedesc[NewT]): auto =
  make_tensor(NewT, make_layout_like(t.layout))


# ── Non-owning: make_view(ptr, Layout) ─────────────────────────────────

func make_view*[T, Sh, St](data: ptr UncheckedArray[T];
                           L: Layout[Sh, St]): TensorView[T, Sh, St] =
  TensorView[T, Sh, St](data: data, layout: L)

template make_view*[T](data: ptr UncheckedArray[T];
                       shape: IntOrIntTuple;
                       order: static StrideOrder = LayoutLeft): untyped =
  make_view(data, make_layout(shape, order))

template make_view*[T](data: ptr UncheckedArray[T];
                       shape, stride: IntOrIntTuple): untyped =
  make_view(data, make_layout(shape, stride))

# ── Non-owning: make_view(ptr T, Layout) ──────────────────────────────

func make_view*[T, Sh, St](data: ptr T;
                           L: Layout[Sh, St]): TensorView[T, Sh, St] =
  make_view(cast[ptr UncheckedArray[T]](data), L)


# ── Non-owning: make_view(openArray, Layout) — zero-copy ───────────────

func make_view*[T, Sh, St](data: openArray[T];
                           L: Layout[Sh, St]): TensorView[T, Sh, St] =
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
    L: Layout[ShB, StB]): TensorView[T, ShB, StB] =
  ## Reinterpret a view with a new layout (same data pointer).
  TensorView[T, ShB, StB](data: tv.data, layout: L)

# ═════════════════════════════════════════════════════════════════════════
#  view() — Tensor → TensorView
# ═════════════════════════════════════════════════════════════════════════

func view*[T, Sh, St](t: Tensor[T, Sh, St]): TensorView[T, Sh, St] =
  TensorView[T, Sh, St](
    data: cast[ptr UncheckedArray[T]](addr t.data[0]),
    layout: t.layout)

# ═════════════════════════════════════════════════════════════════════════
#  Layout accessors
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
#  Display
# ═════════════════════════════════════════════════════════════════════════

proc `$`*[T, Sh, St](t: Tensor[T, Sh, St]): string =
  "Tensor o (" & $t.layout.shape & "):(" & $t.layout.stride & ")"

proc `$`*[T, Sh, St](tv: TensorView[T, Sh, St]): string =
  "TensorView o (" & $tv.layout.shape & "):(" & $tv.layout.stride & ")"
