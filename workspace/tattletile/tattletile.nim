# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## tattletile entry point: the Tile IR object model, the subset parser, the printer, the tile passes and the layout algebra.
##
## Pipeline entries:
## - `parseTileIr`, runtime parsing for dev and inspection
## - `parseTileIrFile`, Nim VM ingestion through staticRead

import
  src/tattletile/layout_algebra,
  src/tattletile/mma_atom,
  src/tattletile/tile_ir,
  src/tattletile/tile_ir_parser,
  src/tattletile/tile_ir_printer,
  src/tattletile/tile_passes

export layout_algebra, mma_atom, tile_ir, tile_ir_parser, tile_ir_printer, tile_passes
