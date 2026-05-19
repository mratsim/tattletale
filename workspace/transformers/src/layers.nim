# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/options
import workspace/libtorch
import ./layers/all_reexports {.all.}

export all_reexports

type
  Layer =
    RopeGQAttention or
    Embedding or
    Linear or
    LMHead or
    GatedMLP or
    RMSNorm or
    # Rope has no trainable weight, only model-load-time cos/sin cache
    TransformerBlock

proc to*[T: Layer](layer: T, target: Device|DeviceKind|ScalarKind): T =
  ## Move or convert all tensors of a "layer" object
  result = T()
  for name, dst, src in fieldPairs(result[], layer[]):
    # TODO: Handle tuples of Tensors?
    when src is Tensor|Layer:
      when src is Embedding:
        {.error: "NotImplemented: Embedding conversion is not supported at the moment due to tied embeddings edge case".}
      dst = src.to(target)
    elif src is Option[Tensor]:
      if src.isSome():
        dst = some(src.unsageGet().to(target))
    elif src is seq[Tensor]|seq[Layer]:
      dst = newSeq[typeof(src)](src.len)
      for i in 0 ..< src.len:
        dst[i] = src[i].to(target)
    elif src is tuple:
      {.error: "NotImplemented: tuple generic conversion is not implemented yet".}
    else:
      dst = src
