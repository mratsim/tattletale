# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Engine API surface: the HwEngine concept, the backend dispatch, the engines' API re-exports.

import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/engines/cuda_api
import workspace/crucible/src/runtime/engines/opencl_api
import workspace/crucible/src/runtime/engines/vulkan_api
import workspace/crucible/src/runtime/engines/webgpu_api
import workspace/crucible/src/runtime/engines/metal_api

type
  HwEngine* = concept engine, type E
    proc ingest(engine: E, source: string)
    proc getArtifact(engine: E)
    proc deviceName(engine: E): string
    proc allocDevice(engine: E, T: typedesc, len: int)
    proc copyToDeviceFromHost(engine: E, dst: auto, src: auto, n: int)
    proc copyToHostFromDevice(engine: E, dst: auto, src: auto, n: int)

## Runtime names a GPU runtime backend: `newEngine(runX)` constructs the engine that executes kernels on it.
## The compile-target enums BackendKind and CompileTarget name emitted shader languages and stay separate from it.
type
  Runtime* = enum
    runCuda     ## CUDA runtime (NVRTC)
    runWebGPU   ## WebGPU runtime (wgpu)
    runOpenCL   ## OpenCL runtime (OpenCL C)
    runVulkan   ## Vulkan runtime (MoltenVK-compatible loader)
    runMetal    ## Metal runtime (macOS)

# when/elif, not case: the branches return distinct engine types.
# Nim type-checks every branch of a case statement even on a static value, so a case would force one common type for the `auto` result.
proc newEngine*(runtime: static Runtime): auto {.inline.} =
  when runtime == runCuda:
    newCudaEngine()
  elif runtime == runOpenCL:
    newOpenCLEngine()
  elif runtime == runVulkan:
    newVulkanEngine()
  elif runtime == runWebGPU:
    newWebgpuEngine()
  elif runtime == runMetal:
    newMetalEngine()

export engines_common.BackendKind, engines_common.Dim3, engines_common.LaunchConfig,
       engines_common.dim3, engines_common.DeviceMem
export cuda_api.CudaEngine, opencl_api.OpenCLEngine, vulkan_api.VulkanEngine,
       webgpu_api.WebgpuEngine, metal_api.MetalEngine
export cuda_api.newCudaEngine,   cuda_api.ingest, cuda_api.getArtifact, cuda_api.deviceName,
       cuda_api.allocDevice, cuda_api.copyToDeviceFromHost, cuda_api.copyToHostFromDevice
export opencl_api.newOpenCLEngine,   opencl_api.ingest,     opencl_api.getArtifact,    opencl_api.deviceName,
       opencl_api.allocDevice, opencl_api.copyToDeviceFromHost, opencl_api.copyToHostFromDevice
export vulkan_api.newVulkanEngine,   vulkan_api.ingest,     vulkan_api.getArtifact,    vulkan_api.deviceName,
       vulkan_api.allocDevice, vulkan_api.copyToDeviceFromHost, vulkan_api.copyToHostFromDevice
export webgpu_api.newWebgpuEngine,   webgpu_api.ingest,   webgpu_api.getArtifact,  webgpu_api.deviceName,
       webgpu_api.allocDevice, webgpu_api.copyToDeviceFromHost, webgpu_api.copyToHostFromDevice
export metal_api.newMetalEngine, metal_api.ingest,  metal_api.getArtifact, metal_api.deviceName,
       metal_api.allocDevice, metal_api.copyToDeviceFromHost, metal_api.copyToHostFromDevice
