# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ─────────────────────  math_consts (shared numeric constants)  ─────────────────────

## Numeric constants shared across the ceramic tile kernels.

const Log2e* = 1.4426950408889634'f32
  ## f32 log base 2 of e, the scale factor of the exp2-based decay and silu
  ## spellings. Metal has no exp device builtin, so e^x is spelled exp2(x·log2e).

const InvSqrt2Pi* = 0.7978845608028654'f32
  ## 1/sqrt(2·pi), the factor of the gelu_pytorch_tanh spelling,
  ## 0.5·g·(1 + tanh(InvSqrt2Pi·(g + GeluCoef·g³))).

const GeluCoef* = 0.044715'f32
  ## Cubic coefficient of the gelu_pytorch_tanh spelling, see InvSqrt2Pi.

const InvSqrt128* = 0.088388347648'f32
  ## 1/sqrt(128), the Hadamard-128 normalization factor of the EXL3
  ## incoherence processing and the fp32 attention q scale.
  ## hadamard_transforms' INV_SQRT_128 is an alias of this constant.
