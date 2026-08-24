## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#          Tile ops: the op umbrella
#
# ############################################################
#
# Aggregates the typed tile-op modules: the unary and binary maps,
# the row maps, the reductions, and the col-vec ops. The ops are
# plain typed procs: their identifiers resolve in their own modules,
# so the consumers import the surface by name and nothing expands
# at the call site.

import ./tiles
import ./tile_conversions
import ./tile_ops_unary
import ./tile_ops_binary
import ./tile_ops_reductions

export tiles, tile_conversions, tile_ops_unary, tile_ops_binary,
       tile_ops_reductions
