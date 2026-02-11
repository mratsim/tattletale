# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  ./linear,
  ../kernels/activations

type
  GatedMLP* = object
    ## Gated MLP layer with fused gate-up projection and SiLU activation.
    ##
    ## This follows the Qwen3 MLP architecture:
    ##   gate_up = Linear(hidden_size, 2 * intermediate_size)
    ##   act = silu_and_mul(gate_up)
    ##   output = Linear(intermediate_size, hidden_size)(act)
    ##
    ## Input:
    ##   - An externally provided `x` of shape (..., hidden_size)
    ##
    ## Return:
    ##   - Output tensor of shape (..., hidden_size)
    gate_up*: Linear
    down*: Linear
    act*: ActivationKind

func init*(_: type GatedMLP, gate_weight, up_weight, down_weight: TorchTensor, activation: ActivationKind): GatedMLP =
  ## Creates a GatedMLP layer from separate gate and up weights.
  ##
  ## Args:
  ##   gate_weight: Weight tensor of shape (intermediate_size, hidden_size)
  ##   up_weight: Weight tensor of shape (intermediate_size, hidden_size)
  ##   down_weight: Weight tensor of shape (hidden_size, intermediate_size)
  ##   act: Activation function to use

  let gate_up_fused = F.cat([gate_weight, up_weight], 0)
  let gate_up = Linear.init(gate_up_fused)
  let down = Linear.init(down_weight)
  GatedMLP(gate_up: gate_up, down: down, act: activation)

proc forward*(self: GatedMLP, x: TorchTensor): TorchTensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   x: Input tensor of shape (..., hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (..., hidden_size)
  ##
  ## Computes:
  ##   gate_up = self.gate_up.forward(x)  # (..., 2 * intermediate_size)
  ##   act = silu_and_mul(gate_up)        # (..., intermediate_size)
  ##   return self.down.forward(act)      # (..., hidden_size)

  let gate_up_out = self.gate_up.forward(x)
  let act_out =
    case self.act
    of kSilu: silu_and_mul(gate_up_out)
  result = self.down.forward(act_out)
