# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU kernel code generator, emits native GPU code: CUDA, OpenCL, Vulkan, WebGPU, Metal.
##
## Public API, one import: `import workspace/crucible`
##
## Backends: cuda: / opencl: / vulkan: / webgpu: / metal:
## - var engine = bkCuda.init()
## - engine.ingest(kernelCode)
## - engine.launchKernel<<grid, blk>>("addKernel", out, (a, b))
##
## Re-exports:
## - gpu_compiler, the DSL macros + builtins
## - runtime/engines, the HwEngine concept and the five backends
## - runtime/launch_device_fn and launchDeviceFn

import ./src/codegen/gpu_compiler
import ./src/runtime/engines
import ./src/runtime/launch_device_fn
export gpu_compiler, engines, launchDeviceFn
