# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan GEMM kernel, 4x8-cell schedule: the launch boundary.
##
## `gemmVulkan(d, a, b, alpha, beta)` computes d = alpha·A·Bᵀ + beta·d
## on a Vulkan engine, one launch, no shared memory, no subgroup use.
## The device body lives in `k_gemm_vulkan_impl.nim`, and this module is the import surface for callers.

import workspace/ceramic/src/tile_algebra
import ./k_gemm_vulkan_impl

export k_gemm_vulkan_impl
