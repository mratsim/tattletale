# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile algebra: register tiles, tile load/store and the matmul
## thread->value maps, all on the layout algebra.

import
  workspace/crucible/src/crucible_gpu_builtins,
  workspace/ceramic/src/tensors,
  workspace/ceramic/src/int_tuples,
  workspace/ceramic/src/layout_algebra,
  workspace/ceramic/src/hardware/h_mma_configgen,
  workspace/ceramic/src/hardware/h_mma_properties,
  workspace/ceramic/src/tile_algebra/tiles,
  workspace/ceramic/src/tile_algebra/tile_foreach,
  workspace/ceramic/src/tile_algebra/tile_elementary,
  workspace/ceramic/src/tile_algebra/tile_ops_unary,
  workspace/ceramic/src/tile_algebra/tile_ops_binary,
  workspace/ceramic/src/tile_algebra/tile_ops_ternary,
  workspace/ceramic/src/tile_algebra/tile_loadstore,
  workspace/ceramic/src/tile_algebra/tile_reductions,
  workspace/ceramic/src/tile_algebra/tile_reduce,
  workspace/ceramic/src/tile_algebra/tile_matmul

export tiles, tile_foreach, tile_elementary, tile_ops_unary, tile_ops_binary, tile_ops_ternary,
       tile_loadstore, tile_reductions, tile_reduce, tile_matmul

# Symbols outside tile_algebra's jurisdiction, re-exported per symbol
# for the single-import surface of the kernels and the macro emissions:
export tensors.TensorOwned, tensors.TensorView, tensors.size,
       tensors.`[]`, tensors.`[]=`
export int_tuples.Int, int_tuples.`+`, int_tuples.`-`, int_tuples.`*`,
       int_tuples.`div`, int_tuples.`mod`, int_tuples.`==`,
       int_tuples.round_multiple_down, int_tuples.round_multiple_up,
       int_tuples.toInt, int_tuples.flatten, int_tuples.tupleStream,
       int_tuples.items
export layout_algebra.Layout, layout_algebra.make_layout, layout_algebra.make_ordered_layout,
       layout_algebra.cosize,
       layout_algebra.crd2idx, layout_algebra.idx2crd,
       layout_algebra.logical_divide, layout_algebra.diceAxes,
       layout_algebra.sliceAxes
export tensors.diceAxes, tensors.sliceAxes,
       tensors.getLayout
export h_mma_configgen, h_mma_properties
export crucible_gpu_builtins.atomicFetchAdd
export crucible_gpu_builtins.reduceAdd
