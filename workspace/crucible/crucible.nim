# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU kernel code generator — takes Ceramic layout algebra and Positron kernel
## specifications and emits native GPU code (CUDA, OpenCL, Vulkan, WebGPU, Metal).
##
## Public API — a single import gives everything:
##   import workspace/crucible
##
##   const kernelCode = cuda:                 # opencl: / vulkan: / webgpu: / metal:
##     proc addKernel(a: ptr UncheckedArray[uint32]; ...) {.global.} = ...
##
##   var engine = bkCuda.init()               # the HwEngine concept + backends
##   engine.ingest(kernelCode)
##   engine.run<<(1, 128)>>("addKernel", out, (a, b))
##
## Re-exports: gpu_compiler
## (the `cuda:`/`opencl:`/`vulkan:`/`webgpu:`/`metal:` DSL macros + builtins)
## and runtime/engines (HwEngine, init, ingest, getArtifact, run, check, deviceName, BackendKind/bk*).
## Internal layers (passes, exec/, engines/nvrtc legacy driver) stay deep by design.

import ./src/codegen/gpu_compiler
import ./src/runtime/engines
export gpu_compiler, engines
