# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU kernel code generator, emits native GPU code: CUDA, OpenCL, Vulkan, WebGPU, Metal.
##
## Public API, one import: `import workspace/crucible`.
## The curated surface lives in `src/crucible.nim`.
## Everything not exported there is internal, only IR tests reach into the internals.
##
## Backends: cuda: / opencl: / vulkan: / webgpu: / metal:
## - var engine = bkCuda.init()
## - engine.ingest(kernelCode)
## - engine.launchKernel<<grid, blk>>("addKernel", out, (a, b))

import ./src/crucible
export crucible
