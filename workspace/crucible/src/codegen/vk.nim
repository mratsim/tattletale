## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan (SPIR-V) runtime compilation and execution.
##
## This module ties together the `vulkan:` macro (GLSL codegen) with
## Vulkan compute runtime execution.
##
## Usage:
##
##   import workspace/crucible/src/codegen/vk
##
##   # 1. Generate GLSL code at compile time with `vulkan:` macro
##   const vkCode = vulkan:
##     proc add(a: ptr UncheckedArray[uint32];
##              b: ptr UncheckedArray[uint32];
##              output: ptr UncheckedArray[uint32]) {.global.} =
##       output[0] = a[0] + b[0]
##
##   # 2. Execute via Vulkan
##   var ctx = initVulkan()
##   let result = execVulkan(ctx, vkCode, "main", 8,
##     inputs = [([10'u32, 20'u32], 8u)])
##   ctx.shutdown()

import std/strformat

import ./gpu_compiler
import ./exec/vulkan_runtime

export gpu_compiler
export vulkan_runtime
