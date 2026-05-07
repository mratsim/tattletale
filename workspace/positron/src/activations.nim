# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/libtorch

type
  ActivationKind* {.size: sizeof(int8).} = enum
    kSilu = 0

func silu_and_mul*(x: TorchTensor): TorchTensor =
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
