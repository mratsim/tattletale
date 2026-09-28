## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## The tile algebra: the whole tile_algebra directory, re-exported.
## Import this one module to get the tile types, ops, io, mma and
## epilogues.

import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tile_algebra/tiles
import workspace/ceramic/src/tile_algebra/tile_config
import workspace/ceramic/src/tile_algebra/tile_io
import workspace/ceramic/src/tile_algebra/tile_mma
import workspace/ceramic/src/tile_algebra/tile_ops_unary
import workspace/ceramic/src/tile_algebra/tile_ops_binary
import workspace/ceramic/src/tile_algebra/tile_ops_reductions
import workspace/ceramic/src/tile_algebra/tile_epilogues
import workspace/ceramic/src/tile_algebra/tile_epilogues_backend

export tiles, tile_config, tile_io,
       tile_mma, tile_ops_unary, tile_ops_binary,
       tile_ops_reductions, tile_epilogues, tile_epilogues_backend
