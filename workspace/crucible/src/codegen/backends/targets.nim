# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Compilation targets: the backend identifier shared by the printers,
## the engines and the launch argument serialization.

type
  HlslFlavor* = enum
    ## Resource-binding flavor of the emitted HLSL.
    ## The flavor changes the resource-binding section only, the language body is identical.
    fvD3D12   ## D3D12-native StructuredBuffer / RWStructuredBuffer / ConstantBuffer
              ## on `register(tN/uN/bN, space0)`, consumable by DXC for DXIL and SPIR-V
    fvVulkan  ## zero-descriptor Vulkan flavor: every parameter binds into one `[[vk::push_constant]]` block,
              ## buffers as device addresses,
              ## accesses construct `vk::BufferPointer` from the address,
              ## dxc -spirv emits physical-storage-buffer words for the address route

  BackendKind* = enum
    bkCuda      ## CUDA backend
    bkWebGPU    ## WebGPU WGSL backend
    bkOpenCL    ## OpenCL C backend
    bkGlsl      ## GLSL backend executed over Vulkan (MoltenVK)
    bkMetal     ## Metal (MSL) backend
    bkHlsl      ## HLSL backend, compiled through glslang's HLSL frontend

export BackendKind
