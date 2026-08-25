## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## The tile algebra: the whole tile_algebra directory, re-exported.
## Import this one module to get the tile types, ops, io, mma and
## epilogues.

import ./layout_algebra
import ./tile_algebra/tiles
import ./tile_algebra/tile_config
import ./tile_algebra/tile_io
import ./tile_algebra/tile_mma
import ./tile_algebra/tile_ops_unary
import ./tile_algebra/tile_ops_binary
import ./tile_algebra/tile_ops_reductions
import ./tile_algebra/tile_epilogues
import ./tile_algebra/tile_epilogues_backend

export layout_algebra, tiles, tile_config, tile_io,
       tile_mma, tile_ops_unary, tile_ops_binary,
       tile_ops_reductions, tile_epilogues, tile_epilogues_backend
