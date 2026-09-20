# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch

type
  ActivationKind* {.size: sizeof(int8).} = enum
    kSilu = 0
    kGeluTanh
      ## Tanh-approximate GELU, the `gelu_pytorch_tanh` activation of the gemma lineage checkpoints

func gelu_tanh*(x: Tensor): Tensor =
  ## Tanh-approximate GELU, the `gelu_pytorch_tanh` activation of the gemma lineage checkpoints.
  ##
  ## Formula:
  ##   result = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  ##
  ## Returns:
  ##   - A tensor with the shape and dtype of the input
  ##   - Evaluated by the backend's fused tanh-approximate gelu kernel,
  ##     the op the reference runtimes call for this activation, so bf16
  ##     activations keep the fused kernel's rounding
  gelu(x, "tanh")

func silu_and_mul*(x: Tensor): Tensor =
  ## Fused SiLU and Mul activation.
  ##
  ## Input:
  ##   - A tensor `x` of shape (..., 2 * hidden_size)
  ##
  ## Return:
  ##   - Output tensor of shape (..., hidden_size)
  ##
  ## Formula:
  ##   output = silu(x[..., :hidden_size]) * x[..., hidden_size:]
  ##
  ## C++ signature (FlashInfer):
  ##   @flashinfer_api
  ##   def silu_and_mul(
  ##       input: torch.Tensor, out: torch.Tensor = None, enable_pdl: Optional[bool] = None
  ##   ) -> torch.Tensor
  ##
  ## Note:
  ##   This is typically used after a fused gate_up projection:
  ##   gate_up = nn.Linear(hidden_size, 2 * hidden_size)
  ##   act = silu_and_mul(gate_up(x))  # Internally splits: silu(gate) * up
  ##   output = down_proj.forward(act)
  let chunks = x.chunk(2, -1)
  return silu(chunks[0]) * chunks[1]
