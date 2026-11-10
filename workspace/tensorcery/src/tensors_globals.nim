# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/ceramic
import workspace/iface
import tensors_internals
import tensors_internal_types
import compute_graph/eval
import compute_graph/ops_2i1o/ops_add
import compute_graph/ops_1iNo/ops_split

var
  gMetalEngineInstance: pointer
  gOpenCLEngineInstance: pointer
  gVulkanEngineInstance: pointer
  gWebgpuEngineInstance: pointer
  gCudaEngineInstance: pointer

proc gMetalEngineErased(_: static Runtime = runMetal): AnyEngine =
  if gMetalEngineInstance.isNil:
    gMetalEngineInstance = cast[pointer](newEngine(runMetal))
  cast[MetalEngine](gMetalEngineInstance).toAnyEngine()

proc gOpenCLEngineErased(_: static Runtime = runOpenCL): AnyEngine =
  if gOpenCLEngineInstance.isNil:
    gOpenCLEngineInstance = cast[pointer](newEngine(runOpenCL))
  cast[OpenCLEngine](gOpenCLEngineInstance).toAnyEngine()

proc gVulkanEngineErased(_: static Runtime = runVulkan): AnyEngine =
  if gVulkanEngineInstance.isNil:
    gVulkanEngineInstance = cast[pointer](newEngine(runVulkan))
  cast[VulkanEngine](gVulkanEngineInstance).toAnyEngine()

proc gWebgpuEngineErased(_: static Runtime = runWebGPU): AnyEngine =
  if gWebgpuEngineInstance.isNil:
    gWebgpuEngineInstance = cast[pointer](newEngine(runWebGPU))
  cast[WebgpuEngine](gWebgpuEngineInstance).toAnyEngine()

proc gCudaEngineErased(_: static Runtime = runCuda): AnyEngine =
  if gCudaEngineInstance.isNil:
    gCudaEngineInstance = cast[pointer](newEngine(runCuda))
  cast[CudaEngine](gCudaEngineInstance).toAnyEngine()

proc gEngine*(run: static Runtime): AnyEngine =
  when run == runMetal:
    gMetalEngineErased()
  elif run == runOpenCL:
    gOpenCLEngineErased()
  elif run == runVulkan:
    gVulkanEngineErased()
  elif run == runWebGPU:
    gWebgpuEngineErased()
  elif run == runCuda:
    gCudaEngineErased()
  else:
    {.error: "gEngine: unsupported run kind".}
