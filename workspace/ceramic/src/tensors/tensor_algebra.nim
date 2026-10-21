# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_indexing
import ./tensor_datatypes
import std/macros

export layout_indexing.slice, layout_indexing.dice
export layout_indexing.sliceAxes, layout_indexing.diceAxes

macro sliceAxes*[T, Sh, St](t: TensorOwned[T, Sh, St], axes: varargs[int]{lit|`const`}): untyped =
  ## Sliced layout over the tensor's axes, the others keep their place.
  result = bindSym"sliceAxes".newCall(newDotExpr(t, bindSym"getLayout"))
  for a in axes:
    result.add(a)

macro sliceAxes*(tv: TensorView, axes: varargs[int]{lit|`const`}): untyped =
  ## Sliced layout over the tensor's axes, the others keep their place.
  result = bindSym"sliceAxes".newCall(newDotExpr(tv, newIdentNode("layout")))
  for a in axes:
    result.add(a)

macro diceAxes*[T, Sh, St](t: TensorOwned[T, Sh, St], axes: varargs[int]{lit|`const`}): untyped =
  ## Layout over the picked axes alone, in the given order.
  result = bindSym"diceAxes".newCall(newDotExpr(t, bindSym"getLayout"))
  for a in axes:
    result.add(a)

macro diceAxes*(tv: TensorView, axes: varargs[int]{lit|`const`}): untyped =
  ## Layout over the picked axes alone, in the given order.
  result = bindSym"diceAxes".newCall(newDotExpr(tv, newIdentNode("layout")))
  for a in axes:
    result.add(a)

template logical_divide*[T, Sh, St](t: TensorOwned[T, Sh, St], tiler: Layout): auto =
  ## A view aliasing the caller's storage, a by-value param would dangle
  ## for small inline arrays.
  make_view(addr t.data[0], t.getLayout().logical_divide(tiler))

func logical_divide*[T, Sh, St](t: TensorView[T, Sh, St], tiler: Layout): auto {.inline.} =
  make_view(addr t.data[0], t.getLayout().logical_divide(tiler))

func getIdentityLayout*[T, Sh, St](t: TensorView[T, Sh, St]): auto {.inline.} =
  ## Identity layout over the tensor's shape, its coordinate space.
  make_identity_layout(t.shape)

func getIdentityLayout*[T, Sh, St](t: TensorOwned[T, Sh, St]): auto {.inline.} =
  ## Identity layout over the tensor's shape, its coordinate space.
  make_identity_layout(t.shape)
