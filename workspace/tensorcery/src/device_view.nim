# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Kernel-face view of a device tensor.
##
## Kernels declare their tensor parameters `distinct DeviceView`.
## The launch binds the generics from the host-side TensorBridge argument.
## The bridge's DeviceMem leaf contributes the element type.
##
## Bridge and view carry the same fields in the same order:
## - data: DeviceMem handle on the host side, base data pointer on the device
## - offset: element offset of the view origin from the buffer base
## - layout: the view's Layout
##
## Kernels build a ceramic TensorView from a DeviceView to reuse the tile
## load/store machinery:
##
##   let zv = make_view(z.data +% z.offset, z.layout)

import workspace/ceramic/src/layout_algebra

type DeviceView*[T, Sh, St] = object
  ## Device-side tensor view: `offset` is the view origin's element offset
  ## from the buffer base.
  data*: ptr UncheckedArray[T]
  offset*: int
  layout*: Layout[Sh, St]
