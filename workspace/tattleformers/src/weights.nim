# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Checkpoint weight loader for the tattleformers engine.
##
## safetensors header -> erased bf16 Tensor:
##
## - one DMA per tensor, over the openArray byte copy boundary
## - zero-copy stream from the checkpoint's memory mapping
## - no owned host copy, no libtorch import

import
  workspace/crucible,
  workspace/tensorcery/src/tensor_datatypes,
  workspace/safetensors/src/collections,
  workspace/safetensors/src/safetensors

proc loadTensor*(engine: AnyEngine, weights: SafetensorsCollection, name: string): Tensor =
  ## Loads one checkpoint tensor as a dense erased bf16 Tensor.
  ##
  ## Args:
  ## - engine: the engine that allocates the device buffer
  ## - weights: the opened checkpoint collection
  ## - name: the tensor's safetensors key
  ##
  ## The erased upload is a flat write over a dense contiguous layout,
  ## the safetensors layout contract guarantees it.
  ## Raises ValueError naming the tensor for a dtype other than BF16,
  ## the engine's only weight element type.
  let (stDtype, shape) = weights.tensorInfo(name)
  if stDtype != BF16:
    raise ValueError.newException(
      "loadTensor: '" & name & "' dtype " & $stDtype &
      " unsupported, the tattleformers weights are bfloat16")
  result = newTensor(engine, shape, kBf16)
  let view = weights.getMmapView(name)
  result.engine.copyToDeviceFromHost(result.node,
    toOpenArray(cast[ptr UncheckedArray[byte]](view.data), 0, view.size - 1))
