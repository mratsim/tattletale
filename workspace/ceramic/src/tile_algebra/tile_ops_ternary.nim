# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./tiles
import ./tile_foreach

# ############################################################
#
#             Element-wise ternary functions
#
# ############################################################

# In-place accumulation
# ---------------------

func mulacc*(dst: var Tile, a, b: Tile) {.inline.} =
  forEach var d in dst, x in a, y in b:
    d += x * y

# Destination buffer
# ------------------

func muladd*(dst: var Tile, a, b, c: Tile) {.inline.} =
  forEach var d in dst, x in a, y in b, z in c:
    d = x * y + z

# Out-of-place
# ------------

func muladd*(a, b, c: Tile): Tile {.inline.} =
  result.muladd(a, b, c)
