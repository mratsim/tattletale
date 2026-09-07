# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Quantization format interfaces — types + registry + registration.
##
## Pure interface: no codec implementation is imported here.
## Codecs import this file to register themselves.

import
  pkg/packedjson,
  std/tables,
  workspace/safetensors,
  workspace/libtorch,
  workspace/safetensors/src/collections,
  ../layers/linear,
  ../layers/lmhead,
  ../layers/norm,
  ../layers/embedding,
  ../layers/ffn,
  ./datatypes

export datatypes

type
  QuantLoaders* = object
    ## Load quantized layers.
    ## In practice, only linear and LMHead layers are quantized.
    linear*: proc(view: SafetensorsCollection, prefix: string, cfg: JsonNode, device: DeviceKind): Linear {.nimcall.}
    lmHead*: proc(view: SafetensorsCollection, device: DeviceKind): LMHead {.nimcall.}
    deployDtype*: ScalarKind
    ## Deployment dtype for all non-quantized tensors, storage dtype aside.

var QuantLoaderRegistry* {.compileTime.}: Table[QuantFormatKind, QuantLoaders]
  ## Compile-time registry populated by codec static blocks.
