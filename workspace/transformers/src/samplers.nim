# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F

proc sample*(logits: Tensor, temp = 1.0f): int =
  ## Gumbel-max sampling.
  ##
  ## Replaces softmax + multinomial with a single argmax pass:
  ##
  ## ```
  ## sampled = argmax( logits / temperature  +  Gumbel(0,1) )
  ## ```
  ##
  ## Gumbel(0,1) = -log(-log(U(0,1)))
  ##
  ## **Complexity**: O(V) — no sort, no softmax, no CDF.
  ## **GPU-friendly**: all ops stay on GPU. No CPU-GPU sync.

  # Clamp uniform random away from exact 0.0 — log(0) is -inf, breaks gumbel
  let u = F.rand_like(logits, F.kFloat32)
            .clamp(1e-6, 1.0)

  # Gumbel(0,1) = -log(-log(U))
  let gumbel = -log(-log(u))

  # logits / temp + gumbel → argmax
  let noisy = logits / temp + gumbel
  let idx = noisy.argmax().item(int)
  return idx
