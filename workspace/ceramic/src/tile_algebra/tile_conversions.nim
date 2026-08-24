## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#          Device scalar math builtins
#
# ############################################################

import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  The scalar math builtins: the element ops of the tile/col-vec maps
# ═════════════════════════════════════════════════════════════════════════
#
#  Declared here so they resolve before the tile_ops map templates (which
#  pass their own names as map ops): a same-named tile template declared
#  earlier would capture the reference.

proc exp2*(x: float32): float32 {.builtin.} = discard
  ## Device-side `2^x`, declared `{.builtin.}` so the DSL forwards
  ## the backend's native `exp2` (used by the online-softmax maps,
  ## exact for integer exponents).

proc rsqrt*(x: float32): float32 {.builtin.} = discard
  ## Device-side `1/sqrt(x)`: the DSL forwards the backend's native `rsqrt`.

