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
