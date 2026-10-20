# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile algebra: register tiles, tile load/store and the matmul
## thread->value maps, all on the layout algebra.

import workspace/crucible/src/codegen/builtins/builtins_gpu_types
export builtins_gpu_types.float16, builtins_gpu_types.bfloat16

import workspace/ceramic/src/tensors
import workspace/ceramic/src/int_tuples/int_tuples_datatypes
import workspace/ceramic/src/int_tuples/int_tuples_unsanctioned_helpers
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/int_tuples/int_tuples_transforms
import workspace/ceramic/src/layout_algebra/layout_constructors
import workspace/ceramic/src/layout_algebra/layouts_datatypes
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/tile_algebra/tiles
import workspace/ceramic/src/tile_algebra/tile_loadstore
import workspace/ceramic/src/tile_algebra/tile_reductions
import workspace/ceramic/src/tile_algebra/tile_matmul

export tensors.TensorOwned, tensors.TensorView
export layouts_datatypes.size
export tiles, tile_loadstore, tile_reductions, tile_matmul

# Whitelist: the symbols the tile templates' macro emissions resolve at the caller's scope
export int_tuples_datatypes.Int
export int_tuples_streams.tupleStream, int_tuples_streams.items
export int_tuples_transforms.flatten
export int_tuples_datatypes.`+`, int_tuples_datatypes.`-`, int_tuples_datatypes.`*`
export int_tuples_datatypes.`div`, int_tuples_datatypes.`mod`
export int_tuples_datatypes.`==`
export int_tuples_unsanctioned_helpers.toInt
export ism_coord_strides.CoordStride
export ism_coord_strides.`*`, ism_coord_strides.`+`
export layout_constructors.make_layout
export layouts_datatypes.cosize
export h_mma_configgen.MmaOperand
export h_mma_properties.getLayoutA, h_mma_properties.getLayoutB, h_mma_properties.getLayoutC
export h_mma_properties.getThreadCount, h_mma_properties.getM, h_mma_properties.getN
export h_mma_properties.getK, h_mma_properties.valuesPerThread, h_mma_properties.getLayout
