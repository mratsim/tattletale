# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_indexing
import ./tensor_datatypes

export layout_indexing.slice, layout_indexing.dice
export layout_indexing.sliceAxes, layout_indexing.diceAxes

func logical_divide*[T, Sh, St](t: TensorOwned[T, Sh, St], tiler: Layout): auto {.inline.} =
  make_view(addr t.data[0], t.getLayout().logical_divide(tiler))

func logical_divide*[T, Sh, St](t: TensorView[T, Sh, St], tiler: Layout): auto {.inline.} =
  make_view(addr t.data[0], t.getLayout().logical_divide(tiler))

func getIdentityLayout*[T, Sh, St](t: TensorView[T, Sh, St]): auto {.inline.} =
  ## Identity layout over the tensor's shape, its coordinate space.
  make_identity_layout(t.shape)

func getIdentityLayout*[T, Sh, St](t: TensorOwned[T, Sh, St]): auto {.inline.} =
  ## Identity layout over the tensor's shape, its coordinate space.
  make_identity_layout(t.shape)
