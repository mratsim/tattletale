# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_indexing
import ./tensor_datatypes
import workspace/crucible
import std/macros

export layout_indexing.slice, layout_indexing.dice
export layout_indexing.sliceAxes, layout_indexing.diceAxes

macro sliceAxes*(t: TensorOwned, axes: varargs[int]{lit|`const`}): untyped =
  ## Sliced layout over the tensor's axes, the others keep their place.
  result = bindSym"sliceAxes".newCall(newDotExpr(t, bindSym"getLayout"))
  for a in axes:
    result.add(a)

macro sliceAxes*(tv: TensorView, axes: varargs[int]{lit|`const`}): untyped =
  ## Sliced layout over the tensor's axes, the others keep their place.
  result = bindSym"sliceAxes".newCall(newDotExpr(tv, newIdentNode("layout")))
  for a in axes:
    result.add(a)

macro diceAxes*(t: TensorOwned, axes: varargs[int]{lit|`const`}): untyped =
  ## Layout over the picked axes alone, in the given order.
  result = bindSym"diceAxes".newCall(newDotExpr(t, bindSym"getLayout"))
  for a in axes:
    result.add(a)

macro diceAxes*(tv: TensorView, axes: varargs[int]{lit|`const`}): untyped =
  ## Layout over the picked axes alone, in the given order.
  result = bindSym"diceAxes".newCall(newDotExpr(tv, newIdentNode("layout")))
  for a in axes:
    result.add(a)

template logical_divide*(t: TensorOwned, tiler: Layout): auto =
  ## A view aliasing the caller's storage, a by-value param would dangle
  ## for small inline arrays.
  make_view(addr t.data[0], t.getLayout().logical_divide(tiler), getOffset(t))

func isNil*[T, L](v: TensorView[T, L]): bool {.inline.} =
  ## True when the view has no backing buffer.
  v.data.isNil()

func logical_divide*(t: TensorView, tiler: Layout): auto {.inline.} =
  make_view(addr t.data[0], t.getLayout().logical_divide(tiler), getOffset(t))

func getIdentityLayout*(t: TensorView): auto {.inline.} =
  ## Identity layout over the tensor's shape, its coordinate space.
  make_identity_layout(t.shape)

func getIdentityLayout*(t: TensorOwned): auto {.inline.} =
  ## Identity layout over the tensor's shape, its coordinate space.
  make_identity_layout(t.shape)

func atAxisIndex*(src: TensorView, axis: static int, idx: int): auto =
  ## View of `src` with `axis` fixed at `idx` and removed from the shape.
  ##
  ##   atAxisIndex(src, axis = 1, idx = 2)  # src (2, 3, 4) -> view (2, 4), view[a, b] aliases src[a, 2, b]
  when src.rank() == 1:
    # a rank-1 `src` keeps no other coordinate, the view is the full tensor
    make_view(src.data, src.getLayout(), src.offset)
  else:
    let others = src.sliceAxes(axis)
    let axisAlone = src.diceAxes(axis)
    make_view(src.data, others, src.offset + crd2idx(axisAlone, idx))

func alongAxisAt*(src: TensorView, axis: static int, idx: int): auto =
  ## Returns a rank-1 view of the `axis` strip of `src`.
  ## `idx` is the flat position over the other axes, dimension 0 fastest.
  ##
  ##   alongAxisAt(src, axis = 1, idx = 5)  # src (2, 3, 4) -> view (3,), view[r] aliases src[1, r, 2]
  when src.rank() == 1:
    # a rank-1 `src` is one strip, the only flat position is the full tensor
    make_view(src.data, src.getLayout(), src.offset)
  else:
    let others = src.sliceAxes(axis)
    let axisAlone = src.diceAxes(axis)
    make_view(src.data, axisAlone, src.offset + crd2idx(others, idx))
