# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  std/options,
  workspace/libtorch as F,
  workspace/positron,
  ../quantizations/datatypes
when defined(cuda):
  import workspace/libpositron_cuda
{.experimental: "callOperator".}

type
  Linear* = ref object
    ## Linear layer
    ##
    ## Input:
    ##   - An externally provided `x` of shape [batch_size, in_features]
    ##   - A local weight of shape [out_features, in_features]
    ##   - Optionally a local bias of shape [1, out_features]
    ##
    ## Return:
    ##   - Weight * x + bias
    weight: Tensor
    bias: Option[Tensor]
    in_features*: int
    out_features*: int
    case quant_format*: QuantFormatKind
    of qBF16:
      discard
    of qExl3:
      suh: Tensor    # [in_features] float16 — Hadamard input scale (EXL3 only)
      svh: Tensor    # [out_features] float16 — Hadamard output scale (EXL3 only)

func init*(_: type Linear, weight: Tensor, bias = none(Tensor)): Linear =
  ## Creates a linear layer from existing weights.
  ##
  ## Args:
  ##   weight: Pre-initialized weight tensor of shape (out_features, in_features)
  ##   bias: Optional bias tensor of shape (out_features,)
  ##
  ## Computes:
  ##   y = x @ weight^T + bias
  Linear(
    quant_format: qBF16,
    weight: weight,
    bias: bias,
    in_features: weight.size(1),
    out_features: weight.size(0)
  )

proc init*(_: type Linear, weight: Tensor, bias = none(Tensor), suh, svh: Tensor): Linear =
  ## EXL3 weight is stored [in_features, out_features] (for F.mm layout, matching ext.hgemm)
  Linear(
    quant_format: qEXL3,
    weight: weight,
    bias: bias,
    in_features: weight.size(0),
    out_features: weight.size(1),
    suh: suh,
    svh: svh,
  )


proc forward*(self: Linear, x: Tensor): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   x: Input tensor of shape (..., in_features)
  ##
  ## Returns:
  ##   Output tensor of shape (..., out_features)

  case self.quant_format
  of qBF16:
    result =
      if self.bias.isSome:
        F.linear(x, self.weight, self.bias.unsafeGet())
      else:
        F.linear(x, self.weight)
  of qEXL3:
    # EXL3 operates in float16
    # Input Hadamard: suh before FWHT; Output Hadamard: svh after FWHT
    when defined(cuda):
      if x.deviceType() == kCuda:
        let xf16 = x.to(kFloat16)
        let xh = hadamard_rotate_128_cuda(xf16, pre_scale = some(self.suh), post_scale = none(Tensor))
        result = F.matmul(xh, self.weight)
        result = hadamard_rotate_128_cuda(result, pre_scale = none(Tensor), post_scale = some(self.svh))
        if self.bias.isSome:
          result += self.bias.unsafeGet()
        return
    # CPU fallback: portable tensor-op FWHT
    let xf16 = x.to(kFloat16)
    let xh = hadamard_rotate_128(xf16, pre_scale = some(self.suh), post_scale = none(Tensor))
    result = F.matmul(xh, self.weight)
    result = hadamard_rotate_128(result, pre_scale = none(Tensor), post_scale = some(self.svh))
    if self.bias.isSome:
      result += self.bias.unsafeGet()

template `()`*(layer: Linear, x: Tensor): untyped =
  forward(layer, x)
