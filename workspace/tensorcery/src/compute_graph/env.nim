# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/ceramic/src/tile_algebra,
  workspace/ceramic/src/hardware/h_mma_properties

const LaneWidth* = 32
  ## Threads per lane group the ops launch and partition over.
  ## Supported: Nvidia warp = 32, Apple simdgroup = 32.
  ## TODO: AMD wave = 64.

template GemmMma*(T: typedesc): untyped =
  ## TODO: the (32, 32, 16) tile schedule is hardcoded
  tiledMma(float32, T, T, 32, 32, 16)
