# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  workspace/libtorch

type VarBuilder* = ref object
  tensors*: seq[(string, TorchTensor)]

proc pp*(vb: VarBuilder, name: string): VarBuilder =
  new result
  result.tensors = vb.tensors

proc clone*(vb: VarBuilder): VarBuilder =
  new result
  result.tensors = vb.tensors

proc get_tensor*(vb: VarBuilder, name: string): TorchTensor =
  for (n, t) in vb.tensors:
    if n == name:
      return t
  raise newException(KeyError, "Tensor not found: " & name)

type RmsNorm* = ref object
  weight*: TorchTensor
  normalized_shape*: int
  eps*: float64

proc newRmsNorm*(
  normalized_shape: int,
  eps: float64,
  vb: VarBuilder
): RmsNorm =
  new result
  result.normalized_shape = normalized_shape
  result.eps = eps

  let shape = @[normalized_shape.int64]
  result.weight = zeros(shape, kFloat32)

func rms_norm_libtorch*(
  input: TorchTensor,
  normalized_shape: IntArrayRef,
  weight: TorchTensor,
  eps: float64
): TorchTensor {.importcpp: "torch::rms_norm(@)".}

proc forward*(self: RmsNorm, x: TorchTensor): TorchTensor =
  let shape = @[self.normalized_shape.int64]
  rms_norm_libtorch(x, shape.asTorchView(), self.weight, self.eps)
