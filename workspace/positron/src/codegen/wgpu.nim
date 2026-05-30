## Constantine
## Copyright (c) 2018-2019    Status Research & Development GmbH
## Copyright (c) 2020-Present Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## WebGPU runtime compilation and execution.
##
## This module ties together the `webgpu:` macro (WGSL codegen) with
## wgpu-native execution, analogous to how `nvrtc.nim` ties together
## the `cuda:` macro with NVRTC + CUDA driver execution.
##
## Usage:
##
##   import workspace/positron/src/codegen/wgpu
##
##   # 1. Generate WGSL code at compile time with `webgpu:` macro
##   const wgslCode = webgpu:
##     proc add(a: ptr UncheckedArray[uint32];
##              b: ptr UncheckedArray[uint32];
##              output: ptr UncheckedArray[uint32]) {.global.} =
##       output[0] = a[0] + b[0]
##
##   # 2. Execute on CPU via wgpu-native
##   var ctx = initWgpu()
##   let result = execWgpu(ctx, wgslCode, "add", 4, inputs = ...)
##   ctx.shutdown()

import std/strformat

import ./gpu_compiler
import ./exec/wgpu_runtime

export gpu_compiler
export wgpu_runtime
