# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Kernel-side tensor view: the staging boundary between tensor layer and kernels.
##
## | stage        | role                                                                     |
## | ------------ | ------------------------------------------------------------------------ |
## | BridgeTensor | tensor-layer view parameters over one device buffer (src/raw/bridge.nim) |
## | makeView     | pads the leaves into the 4-axis image, producing the TensorView          |
##
## The TensorView type never leaves the kernel layer.
##
## Padded image contract feeding the divmod chains:
##
## - `shape` pads with 1s, keeping the divmod chain safe
## - `strides` pads with 0s, so padded axes contribute nothing
## - `step` is the last real axis stride, never a padded 0

import workspace/data_structures/src/small_seqs

type TensorView*[T] = object
  ## Kernel-side view over one device buffer of element type T.
  ##
  ## Padded image contract feeding the kernels' divmod chains:
  ##
  ## - `shape` pads with 1s, keeping the divmod chain safe
  ## - `strides` pads with 0s, so padded axes contribute nothing
  ## - `step` is the last real axis stride, never a padded 0
  base*: uint64                  ## device address of the buffer's element 0
  off*: int32                    ## element offset of the view origin
  shape*, strides*: array[4, int32]
  step*: int32

proc makeView*[T](base: uint64, offset: int, shape, strides: SmallSeq[4, int32]): TensorView[T] =
  ## Builds the kernel's TensorView from a device address and runtime-int leaves
  ## through the padded image contract above.
  result.base = base
  result.off = int32(offset)
  for i in 0 ..< 4:
    if i < int(shape.len):
      result.shape[i] = shape[i]
      result.strides[i] = strides[i]
    else:
      result.shape[i] = 1
      result.strides[i] = 0
  result.step = int32(strides[int(shape.len) - 1])
