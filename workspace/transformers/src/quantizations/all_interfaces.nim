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
  pkg/packedjson,
  std/tables,
  workspace/safetensors,
  workspace/libtorch,
  ../layers/linear,
  ../layers/norm,
  ../layers/embedding,
  ./datatypes

export datatypes

type
  QuantLoaders* = object
    linear*: proc(st: Safetensor, prefix: string, cfg: JsonNode, device: DeviceKind): Linear {.nimcall.}
    rmsNorm*: proc(st: Safetensor, prefix: string, cfg: JsonNode, device: DeviceKind): Tensor {.nimcall.}
    embedding*: proc(st: Safetensor, prefix: string, cfg: JsonNode, device: DeviceKind): Tensor {.nimcall.}
    activationDtype*: ScalarKind  ## Activation dtype for this quant format

var QuantLoaderRegistry* {.compileTime.}: Table[QuantFormatKind, QuantLoaders]
  ## Compile-time registry populated by codec static blocks.
