# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## tattletile entry point: the Tile IR object model, the subset parser, the printer, the tile passes, the layout algebra and the CIRv2 bridge.
##
## Pipeline entries:
## - `parseTileIr`, runtime parsing for dev and inspection
## - `parseTileIrFile`, Nim VM ingestion through staticRead
## - `compileTileIrToMsl` and `compileTileIrToSpirv`, compile-time compilation,
##   one shared CIRv2 bridge, the backend is crucible's pass composition
## - `embedKernel`, the product entry, Tile IR file to a const kernel object
##   in the Nim VM, the backend parameter picks the artifact

import
  src/tattletile/layout_algebra,
  src/tattletile/mma_atom,
  src/tattletile/tile_ir,
  src/tattletile/tile_ir_parser,
  src/tattletile/tile_ir_printer,
  src/tattletile/tile_passes,
  src/tattletile/low_tile,
  src/tattletile/embed

export layout_algebra, mma_atom, tile_ir, tile_ir_parser, tile_ir_printer, tile_passes, low_tile, embed
