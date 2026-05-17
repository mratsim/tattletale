# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  workspace/positron,
  ./linear

{.experimental: "callOperator".}

type
  GatedMLP* = ref object
    ## Gated MLP layer with separate gate and up projections and SiLU activation.
    ##
    ## This follows the Qwen3 MLP architecture:
    ##   gate = gate_proj(x)        # (..., intermediate_size)
    ##   up   = up_proj(x)          # (..., intermediate_size)
    ##   activation = silu(gate) * up
    ##   output = down_proj(activation)
    ##
    ## Input:
    ##   - An externally provided `x` of shape (..., hidden_size)
    ##
    ## Return:
    ##   - Output tensor of shape (..., hidden_size)
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    activation: ActivationKind

func init*(
    _: type GatedMLP,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: ActivationKind = kSilu
  ): GatedMLP =
  ## Creates a GatedMLP layer from pre-constructed Linear projections.
  GatedMLP(
    gate_proj: gate_proj,
    up_proj: up_proj,
    down_proj: down_proj,
    activation: activation
  )

func init*(
    _: type GatedMLP,
    gate_weight, up_weight, down_weight: Tensor,
    activation: ActivationKind = kSilu
  ): GatedMLP =
  ## Creates a GatedMLP layer from separate gate and up weight tensors.
  ## The weights are still stored as separate projections (no fusion).
  GatedMLP(
    gate_proj: Linear.init(gate_weight),
    up_proj: Linear.init(up_weight),
    down_proj: Linear.init(down_weight),
    activation: activation
  )

proc forward*(self: GatedMLP, x: Tensor): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   x: Input tensor of shape (..., hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (..., hidden_size)
  ##
  ## Computes:
  ##   gate_out = self.gate_proj(x)     # (..., intermediate_size)
  ##   up_out   = self.up_proj(x)       # (..., intermediate_size)
  ##   act      = silu(gate_out) * up_out
  ##   return self.down_proj(act)
  let gate_out = self.gate_proj(x)
  let up_out = self.up_proj(x)
  let act_out =
    case self.activation
    of kSilu: F.silu(gate_out) * up_out # TODO silu_and_mul fusion
  result = self.down_proj(act_out)

template `()`*(layer: GatedMLP, x: Tensor): untyped =
  forward(layer, x)
