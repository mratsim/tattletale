# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Engine API surface: the HwEngine concept, the backend dispatch, the engines' API re-exports.

import
  ./engines_common,
  ./nvrtc, ./cl, ./vk, ./wgpu, ./metal

type
  HwEngine* = concept engine, type E
    proc ingest(engine: E, source: string)
    proc getArtifact(engine: E)
    proc deviceName(engine: E): string
    proc allocDevice(engine: E, T: typedesc, len: int)
    proc copyToDeviceFromHost(engine: E, dst: auto, src: auto, n: int)
    proc copyToHostFromDevice(engine: E, dst: auto, src: auto, n: int)

proc newEngine*(backend: static BackendKind): auto =
  when backend == bkCuda:
    newCudaEngine()
  elif backend == bkOpenCL:
    newOpenCLEngine()
  elif backend == bkVulkan:
    newVulkanEngine()
  elif backend == bkWGSL:
    newWgpuEngine()
  elif backend == bkMetal:
    newMetalEngine()
  else:
    {.error: "newEngine: unknown BackendKind".}

export engines_common.BackendKind, engines_common.Dim3, engines_common.LaunchConfig,
       engines_common.dim3, engines_common.DeviceMem
export nvrtc.newCudaEngine,   nvrtc.ingest, nvrtc.getArtifact, nvrtc.deviceName,
       nvrtc.allocDevice, nvrtc.copyToDeviceFromHost, nvrtc.copyToHostFromDevice
export cl.newOpenCLEngine,   cl.ingest,     cl.getArtifact,    cl.deviceName,
       cl.allocDevice, cl.copyToDeviceFromHost, cl.copyToHostFromDevice
export vk.newVulkanEngine,   vk.ingest,     vk.getArtifact,    vk.deviceName,
       vk.allocDevice, vk.copyToDeviceFromHost, vk.copyToHostFromDevice
export wgpu.newWgpuEngine,   wgpu.ingest,   wgpu.getArtifact,  wgpu.deviceName,
       wgpu.allocDevice, wgpu.copyToDeviceFromHost, wgpu.copyToHostFromDevice
export metal.newMetalEngine, metal.ingest,  metal.getArtifact, metal.deviceName,
       metal.allocDevice, metal.copyToDeviceFromHost, metal.copyToHostFromDevice
