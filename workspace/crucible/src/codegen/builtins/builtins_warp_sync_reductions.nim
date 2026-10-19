# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

# ═══════════════════════════════════════════════════════════
#   Warp-synchronous reductions
# ═══════════════════════════════════════════════════════════

proc reduceAdd*[T](v: T): T {.builtin.} = discard
  ## Reduces `v` across the subgroup with addition and broadcasts
  ## the total to every active lane. The value of `v` is only read
  ## on the active lanes.
  ##
  ## Backends emit: MSL `simd_sum`, GLSL `subgroupAdd`,
  ## WGSL `subgroupAdd`, OpenCL `sub_group_reduce_add`.
  ## CUDA lowers to a `__shfl_xor_sync` butterfly over deltas 16/8/4/2/1.

proc reduceMax*[T](v: T): T {.builtin.} = discard
  ## Reduces `v` across the subgroup with max and broadcasts
  ## the result to every active lane. The value of `v` is only read
  ## on the active lanes.
  ##
  ## Backends emit: MSL `simd_max`, GLSL `subgroupMax`,
  ## WGSL `subgroupMax`, OpenCL `sub_group_reduce_max`.
  ## CUDA lowers to a `__shfl_xor_sync` butterfly over deltas 16/8/4/2/1.
