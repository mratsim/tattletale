# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  std/options,
  workspace/libtorch as F

type
  Linear* = object
    ## Linear layer
    ##
    ## Input:
    ##   - An externally provided `x` of shape [batch_size, in_features]
    ##   - A local weight of shape [out_features, in_features]
    ##   - Optionally a local bias of shape [1, out_features]
    ##
    ## Return:
    ##   - Weight * x + bias
    weight*: TorchTensor
    bias*: Option[TorchTensor]
    in_features*: int
    out_features*: int

func init*(_: type Linear, weight: TorchTensor, bias = none(TorchTensor)): Linear =
  ## Creates a linear layer from existing weights.
  ##
  ## Args:
  ##   weight: Pre-initialized weight tensor of shape (out_features, in_features)
  ##   bias: Optional bias tensor of shape (out_features,)
  ##
  ## Computes:
  ##   y = x @ weight^T + bias
  Linear(
    weight: weight,
    bias: bias,
    in_features: weight.size(1),
    out_features: weight.size(0)
  )

proc forward*(self: Linear, x: TorchTensor): TorchTensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   x: Input tensor of shape (..., in_features)
  ##
  ## Returns:
  ##   Output tensor of shape (..., out_features)

  if self.bias.isSome:
    F.linear(x, self.weight, self.bias.get())
  else:
    F.linear(x, self.weight)
