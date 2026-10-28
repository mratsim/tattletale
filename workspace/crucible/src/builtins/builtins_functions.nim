# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Built-in device functions (exp2/rsqrt) and the Nim-operator/function
## name tables the codegen consults.

import std/math
import ./builtins_pragmas

# ═══════════════════════════════════════════════════════════════════════
#  The scalar math builtins
# ═══════════════════════════════════════════════════════════════════════

proc exp2*(x: float32): float32 {.builtin.} = pow(2'f32, x)
  ## Returns 2ˣ
  ##
  ## Declared {.builtin.} so lowers to the backen native function
  ## or the body on host

proc rsqrt*(x: float32): float32 {.builtin.} = 1 / sqrt(x)
  ## Returns 1/√x
  ##
  ## Declared {.builtin.} so lowers the backen native function
  ## or the body on host
