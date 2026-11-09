# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Backend engine handles, constructed on first use
##
## - each instance sits in a raw pointer: no destructor, no eager driver
##   library load in binaries that import tensorcery without the backend
## - the test entry points and user mains grab their backend here

import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api

var
  gMetalEngineInstance: pointer
  gOpenCLEngineInstance: pointer
  gVulkanEngineInstance: pointer
  gWebgpuEngineInstance: pointer
  gCudaEngineInstance: pointer

proc gMetalEngine*(): MetalEngine =
  if gMetalEngineInstance.isNil:
    gMetalEngineInstance = cast[pointer](newEngine(runMetal))
  cast[MetalEngine](gMetalEngineInstance)

proc gOpenCLEngine*(): OpenCLEngine =
  if gOpenCLEngineInstance.isNil:
    gOpenCLEngineInstance = cast[pointer](newEngine(runOpenCL))
  cast[OpenCLEngine](gOpenCLEngineInstance)

proc gVulkanEngine*(): VulkanEngine =
  if gVulkanEngineInstance.isNil:
    gVulkanEngineInstance = cast[pointer](newEngine(runVulkan))
  cast[VulkanEngine](gVulkanEngineInstance)

proc gWebgpuEngine*(): WebgpuEngine =
  if gWebgpuEngineInstance.isNil:
    gWebgpuEngineInstance = cast[pointer](newEngine(runWebGPU))
  cast[WebgpuEngine](gWebgpuEngineInstance)

proc gCudaEngine*(): CudaEngine =
  if gCudaEngineInstance.isNil:
    gCudaEngineInstance = cast[pointer](newEngine(runCuda))
  cast[CudaEngine](gCudaEngineInstance)
