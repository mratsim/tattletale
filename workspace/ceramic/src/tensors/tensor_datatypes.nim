## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layouts
import workspace/ceramic/src/int_tuples

# ═════════════════════════════════════════════════════════════════════════
#  TensorOwned / TensorView
# ═════════════════════════════════════════════════════════════════════════

type
  TensorOwned*[T; L: Layout] = object
    ## Owning tensor — stack-allocated array.
    ## Requires static shape/stride (compile-time cosize).
    data*: array[cosize(L), T]

  TensorView*[T; L: Layout] = object
    data*: ptr UncheckedArray[T]
    offset*: int
    layout*: L

type AnyTensor*[T; L: Layout] = TensorView[T, L] or TensorOwned[T, L]

# ═════════════════════════════════════════════════════════════════════════
#  Constructors
# ═════════════════════════════════════════════════════════════════════════

# ── Owning: make_tensor(T, Layout) ─────────────────────────────────────

func make_tensor*[T, L](_: typedesc[T], l: L): TensorOwned[T, L] {.inline.} =
  TensorOwned[T, L]()

template make_tensor*[T](_: typedesc[T]; shape: IntOrIntTuple;
                         order: static StrideOrder = LayoutLeft): untyped =
  make_tensor(T, make_layout(shape, order))

template make_tensor*[T](_: typedesc[T]; shape, stride: IntOrIntTuple): untyped =
  make_tensor(T, make_layout(shape, stride))

# ── make_tensor_like — create owning tensor with compact strides ──────

func make_tensor_like*[T, L](t: TensorView[T, L]): auto {.inline.} =
  make_tensor(T, make_layout_like(t.getLayout()))

func make_tensor_like*[T, L](t: TensorOwned[T, L]): auto {.inline.} =
  make_tensor(T, make_layout_like(t.getLayout()))

func make_tensor_like*[NewT](t: TensorView, _: typedesc[NewT]): auto {.inline.} =
  make_tensor(NewT, make_layout_like(t.getLayout()))

func make_tensor_like*[NewT](t: TensorOwned, _: typedesc[NewT]): auto {.inline.} =
  make_tensor(NewT, make_layout_like(t.getLayout()))


# ── Non-owning: make_view(ptr, Layout) ─────────────────────────────────

template make_view*[T, L](pD: ptr UncheckedArray[T] or ptr T, l: L, elementOffset: int = 0): TensorView[T, L] =
  TensorView[T, L](data: cast[ptr UncheckedArray[T]](pD), offset: elementOffset, layout: l)

template make_view*[T](data: ptr UncheckedArray[T] or ptr T;
                       shape: IntOrIntTuple;
                       order: static StrideOrder = LayoutLeft): untyped =
  make_view(data, make_layout(shape, order))

template make_view*[P: ptr | ptr UncheckedArray, ShT, StT: IntOrIntTuple](
    data: P; shape: ShT; stride: StT): untyped =
  make_view(data, make_layout(shape, stride))

template slice*(t: AnyTensor, R: Layout, sel: untyped): auto =
  make_view(addr t.data[0], slice(R, sel), getOffset(t) + crd2idx(R, sel))

# ── Non-owning: make_view(openArray, Layout) — zero-copy ───────────────

func make_view*[T, L](data: openArray[T], l: L, elementOffset: int = 0): TensorView[T, L] {.inline.} =
  make_view(cast[ptr UncheckedArray[T]](addr data[0]), l, elementOffset)

template make_view*[T](data: openArray[T];
                       shape: IntOrIntTuple;
                       order: static StrideOrder = LayoutLeft): untyped =
  make_view(data, make_layout(shape, order))

template make_view*[T](data: openArray[T];
                       shape, stride: IntOrIntTuple): untyped =
  make_view(data, make_layout(shape, stride))

# ── Non-owning: make_view(TensorView, Layout) — reinterpret ────────────

func make_view*[T, La, Lb](tv: TensorView[T, La], l: Lb): TensorView[T, Lb] {.inline.} =
  TensorView[T, Lb](data: tv.data, offset: tv.offset, layout: l)

template make_view*(tv: TensorView;
                       shape: IntOrIntTuple;
                       order: static StrideOrder = LayoutLeft): untyped =
  make_view(tv, make_layout(shape, order))

template make_view*(tv: TensorView;
                       shape, stride: IntOrIntTuple): untyped =
  make_view(tv, make_layout(shape, stride))

# ═════════════════════════════════════════════════════════════════════════
#  Layout accessors
# ═════════════════════════════════════════════════════════════════════════

func getOffset*(tv: TensorView): int {.inline.} = tv.offset
func getOffset*(t: TensorOwned): int {.inline.} = 0

func getOriginPtr*[T, L](tv: TensorView[T, L]): ptr UncheckedArray[T] {.inline.} =
  cast[ptr UncheckedArray[T]](addr tv.data[tv.offset])

func getOriginPtr*[T, L](t: TensorOwned[T, L]): ptr UncheckedArray[T] {.inline.} =
  cast[ptr UncheckedArray[T]](addr t.data[0])

template getLayout*[T, Sh, St](t: TensorOwned[T, Layout[Sh, St]]): Layout[Sh, St] =
  make_layout(default(Sh), default(St))
template getLayout*(tv: TensorView): untyped = tv.layout

# ═════════════════════════════════════════════════════════════════════════
#  view() — TensorOwned → TensorView
# ═════════════════════════════════════════════════════════════════════════

func view*[T, L](t: TensorOwned[T, L]): TensorView[T, L] {.inline.} =
  TensorView[T, L](data: cast[ptr UncheckedArray[T]](addr t.data[0]), layout: t.getLayout())

template shape*[T, Sh, St](t: TensorOwned[T, Layout[Sh, St]]): untyped =
  ## All-static Sh is the shape value, no Layout build needed.
  default(Sh)
template shape*(tv: TensorView): untyped = tv.layout.shape

template stride*[T, Sh, St](t: TensorOwned[T, Layout[Sh, St]]): untyped =
  ## All-static St is the stride value, no Layout build needed.
  default(St)
template stride*(tv: TensorView): untyped = tv.layout.stride

template rank*(t: TensorOwned): untyped =
  t.getLayout().rank()
template rank*(tv: TensorView): untyped = tv.layout.rank()

template size*(t: TensorOwned): untyped =
  ## size folds the shape leaves only.
  t.getLayout().size()
template size*(tv: TensorView): untyped = tv.layout.size()

template cosize*(tv: TensorView): untyped = tv.layout.cosize()
template cosize*(t: TensorOwned): untyped =
  t.getLayout().cosize()

# ═════════════════════════════════════════════════════════════════════════
#  Display
# ═════════════════════════════════════════════════════════════════════════

proc `$`*(t: TensorOwned): string =
  "TensorOwned o (" & $t.shape & "):(" & $t.stride & ")"

proc `$`*(tv: TensorView): string =
  "TensorView o (" & $tv.layout.shape & "):(" & $tv.layout.stride & ")"
