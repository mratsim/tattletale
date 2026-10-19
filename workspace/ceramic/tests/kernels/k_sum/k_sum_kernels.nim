# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Device kernel procs for the k_sum test suite, imported by
## t_kernel_sum_template.nim. Device procs must live in module scope,
## never inside a template body, for the launch bridge to resolve them.
##
## Each case fixes its block, the lane's register block with the last axis
## spread across the 32 lanes, and runs k_tile_sum's device functions.

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_sum

# 2D: the (8, 4) block covers (8, 128) of the source
proc sumCaseFull*(o: distinct TensorView, src: distinct TensorView) =
  sum_full((8, 4), o, src)

proc sumCaseAxis*(o: distinct TensorView, src: distinct TensorView) =
  sum_axis((8, 4), o, src)

# 3D: the (2, 2, 2) block covers (2, 2, 64)
proc sumCase3DFull*(o: distinct TensorView, src: distinct TensorView) =
  sum_full((2, 2, 2), o, src)

proc sumCase3DAxis*(o: distinct TensorView, src: distinct TensorView) =
  sum_axis((2, 2, 2), o, src)

# 4D: the (2, 2, 2, 2) block covers (2, 2, 2, 64)
proc sumCase4DFull*(o: distinct TensorView, src: distinct TensorView) =
  sum_full((2, 2, 2, 2), o, src)

proc sumCase4DAxis*(o: distinct TensorView, src: distinct TensorView) =
  sum_axis((2, 2, 2, 2), o, src)
