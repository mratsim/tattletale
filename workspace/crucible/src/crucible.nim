# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Curated public surface of crucible.
##
## | Section       | Symbols                                                                                                               |
## | ------------- | --------------------------------------------------------------------------------------------------------------------- |
## | Engines       | `init`, `BackendKind`, `HwEngine`, `ingest`, `getArtifact`, `deviceName`                                              |
## | Device memory | `DeviceMem`, `allocDevice`, `freeDevice`, `copyToDeviceFromHost`, `copyToHostFromDevice`                              |
## | Launch sugar  | `launchKernel`, `launchDeviceFn`, chevrons `<<`/`>>`, `LaunchSugar`, `DeviceFnSugar`, `DeviceFnProxy`, `LaunchConfig` |
## | Geometry      | `Dim3`, `dim3`                                                                                                        |
## | Codegen       | `cuda`, `opencl`, `vulkan`, `webgpu`, `metal`                                                                         |
## | Builtins      | everything in `builtins` and `builtins_compilermagic`, callable from kernel bodies                                    |

import
  ./codegen/gpu_compiler,
  ./runtime/engines/engines_common,
  ./runtime/engines/nvrtc,
  ./runtime/engines/cl,
  ./runtime/engines/vk,
  ./runtime/engines/wgpu,
  ./runtime/engines/metal,
  ./runtime/launch/launch_kernel,
  ./runtime/launch/launch_device_fn

# ═══════════════════════════════════════════════════════════════
#  Engines
# ═══════════════════════════════════════════════════════════════

export engines_common.BackendKind, engines_common.HwEngine,
       engines_common.init

export nvrtc.newCudaEngine,   nvrtc.ingest, nvrtc.getArtifact, nvrtc.deviceName
export cl.newOpenCLEngine,   cl.ingest,     cl.getArtifact,    cl.deviceName
export vk.newVulkanEngine,   vk.ingest,     vk.getArtifact,    vk.deviceName
export wgpu.newWgpuEngine,   wgpu.ingest,   wgpu.getArtifact,  wgpu.deviceName
export metal.newMetalEngine, metal.ingest,  metal.getArtifact, metal.deviceName

export nvrtc.freeDeviceImpl, cl.freeDeviceImpl, vk.freeDeviceImpl,
       wgpu.freeDeviceImpl, metal.freeDeviceImpl

export launch_kernel.launchKernel
export launch_device_fn.launchDeviceFn

proc allocDevice*(engine: HwEngine, T: typedesc, len: int): auto {.inline.} =
  ## Allocates len elements of T on the engine, returns a typed DeviceMem handle
  let h = engine.allocDeviceImpl(len * sizeof(T))
  DeviceMem[typeof(h), T](mem: h)

proc copyToDeviceFromHost*[H, T](engine: HwEngine, dst: DeviceMem[H, T], src: ptr T or ptr UncheckedArray[T], n: int) {.inline.} =
  ## Copies n elements from host to device
  engine.copyToDeviceFromHostImpl(dst.mem, src, n * sizeof(T))

proc copyToHostFromDevice*[H, T](engine: HwEngine, dst: ptr T or ptr UncheckedArray[T], src: DeviceMem[H, T], n: int) {.inline.} =
  ## Copies n elements from device to host
  engine.copyToHostFromDeviceImpl(dst, src.mem, n * sizeof(T))

# ═══════════════════════════════════════════════════════════════
#  Launch sugar
# ═══════════════════════════════════════════════════════════════

export launch_kernel.LaunchSugar,
       launch_kernel.`<<`, launch_kernel.`>>`
export engines_common.LaunchConfig, engines_common.Dim3,
       engines_common.dim3, engines_common.DeviceMem
export launch_device_fn.DeviceFnSugar,
       launch_device_fn.DeviceFnProxy, launch_device_fn.`<<`, launch_device_fn.`>>`


# ═══════════════════════════════════════════════════════════════
#  Codegen
# ═══════════════════════════════════════════════════════════════

export gpu_compiler.cuda, gpu_compiler.opencl, gpu_compiler.vulkan,
       gpu_compiler.webgpu, gpu_compiler.metal
export gpu_compiler.builtins, gpu_compiler.builtins_compilermagic
