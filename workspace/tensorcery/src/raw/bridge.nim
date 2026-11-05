# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## BridgeTensor staging: the handoff between the tensor layer and the kernel layer.
##
## A BridgeTensor contains one device buffer address plus the runtime shape and stride
## leaves of a view. `toView` translates it into the kernel-side TensorView
## (src/kernels/k_view.nim), which never leaves the kernel layer.

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery/src/kernels/k_view

type BridgeTensor*[T] = object
  ## Staging view over one device buffer: the tensor layer's view parameters
  ## before translation into the kernel image.
  base*: uint64                  ## device address of the buffer's element 0
  offset*: int                   ## element offset of the view origin
  shape*, strides*: SmallSeq[4, int32]

proc toView*[T](b: BridgeTensor[T]): TensorView[T] =
  ## Translates the staging view into the kernel-side TensorView through makeView,
  ## applying the padded image contract: shape pads with 1s, strides pad with 0s,
  ## step keeps the last real axis stride.
  makeView[T](b.base, b.offset, b.shape, b.strides)
