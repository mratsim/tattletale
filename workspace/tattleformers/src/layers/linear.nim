# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  workspace/tensorcery/src/tensorcery

{.experimental: "callOperator".}

type
  Linear* = ref object
    ## Linear layer
    ##
    ## Input:
    ##   - An externally provided `x` of shape (..., in_features)
    ##   - A local weight of shape [out_features, in_features]
    ##   - Optionally a local bias of shape [1, out_features]
    ##
    ## Return:
    ##   - Weight * x + bias
    weight: Tensor
    bias: Option[Tensor]
    in_features*: int
    out_features*: int
    # quant_format is qBF16 only: the qExl3 branch is unported.
    # EXL3 weight is stored [in_features, out_features] (for F.mm layout, matching ext.hgemm)

proc init*(_: type Linear, weight: Tensor, bias = none(Tensor)): Linear =
  ## Creates a linear layer from existing weights.
  ##
  ## Args:
  ## - weight: pre-initialized weight tensor of shape (out_features, in_features)
  ## - bias: optional bias tensor of shape (out_features,)
  ##
  ## Computes y = x @ weight^T + bias
  Linear(
    weight: weight,
    bias: bias,
    in_features: weight.shape()[1],
    out_features: weight.shape()[0]
  )

proc forward*(self: Linear, x: Tensor): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   x: Input tensor of shape (..., in_features)
  ##
  ## Returns:
  ##   Output tensor of shape (..., out_features)

  # Qwen3 runs bias-free (attention_bias = false), the no-bias path is primary.
  # Reference bias semantics: F.linear(x, w, b) = x @ w^T + b, b of shape
  # (out_features,) broadcast over the token dimension.
  if self.bias.isSome:
    raise ValueError.newException(
      "Linear bias is unported: the Qwen3 path runs bias-free " &
      "(reference semantics: F.linear(x, w, b) = x @ weight^T + bias)")

  # GEMM: the stored (out_features, in_features) weight feeds the matmul
  # op's BLAS-Bᵀ form directly, the transposition lives in the op
  # (zero copy). The Tensor-face linear takes that stored form.
  # Rank 2: flatten (..., in_features) to (tokens, in_features),
  # then restore the leading dimensions over the output
  let s = x.shape()
  var tokens = 1
  for i in 0 ..< s.len - 1:
    tokens *= s[i]
  let flat = x.reshape([tokens, self.in_features])
  let proj = linear(flat, self.weight)

  var outShape: seq[int]
  for i in 0 ..< s.len - 1:
    outShape.add(s[i])
  outShape.add(self.out_features)
  proj.reshape(outShape)

template `()`*(layer: Linear, x: Tensor): untyped =
  forward(layer, x)
