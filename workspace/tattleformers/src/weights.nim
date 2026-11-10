# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Checkpoint weight loader for the tattleformers model stack.
##
## safetensors header -> erased bf16 Tensor:
##
## - one DMA per tensor, over the Tensor face's upload boundary
## - zero-copy stream from the checkpoint's memory mapping
## - no owned host copy

import
  workspace/tensorcery/src/tensorcery,
  workspace/safetensors/src/collections,
  workspace/safetensors/src/safetensors

proc loadTensor*(weights: SafetensorsCollection, name: string): Tensor =
  ## Loads one checkpoint tensor as a dense erased bf16 Tensor.
  ##
  ## Args:
  ## - weights: the opened checkpoint collection
  ## - name: the tensor's safetensors key
  ##
  ## The erased upload is a flat write over a dense contiguous layout,
  ## the safetensors layout contract guarantees it. The upload streams
  ## the checkpoint's memory mapping directly: no owned host copy.
  ##
  ## Raises ValueError naming the tensor for a dtype other than BF16,
  ## the tattleformers weights' only element type.
  let (stDtype, shape) = weights.tensorInfo(name)
  if stDtype != BF16:
    raise ValueError.newException(
      "loadTensor: '" & name & "' dtype " & $stDtype &
      " unsupported, the tattleformers weights are bfloat16")
  result = empty(shape)
  let view = weights.getMmapView(name)
  result.upload(toOpenArray(
    cast[ptr UncheckedArray[bfloat16]](view.data), 0, view.size div 2 - 1))
