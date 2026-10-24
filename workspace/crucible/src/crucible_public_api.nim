# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Curated public surface of crucible.
##
## | Section       | Symbols                                                                                                                             |
## | ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
## | Engines       | `newEngine`, `BackendKind`, `HwEngine`, `ingest`, `getArtifact`, `deviceName`                                                       |
## | Device memory | `DeviceMem`, `allocDevice`, `copyToDeviceFromHost`, `copyToHostFromDevice`                                                          |
## | Launch sugar  | `launchKernel`, `launchDeviceFn`, chevrons `<<`/`>>`, `LaunchSugar`, `DeviceFnSugar`, `DeviceFnProxy`, `LaunchConfig` |
## | Geometry      | `Dim3`, `dim3`                                                                                                                      |
## | Codegen       | `cuda`, `opencl`, `glsl`, `webgpu`, `metal`, `hlsl` (+ `HlslFlavor`), `spirv`                                                       |

import
  ./codegen/gpu_compiler,
  ./runtime/engines/engines_api,
  ./runtime/launch/launch_kernel,
  ./runtime/launch/launch_device_fn

# ═══════════════════════════════════════════════════════════════════════
#  Engines
# ═══════════════════════════════════════════════════════════════════════

export engines_api

export launch_kernel.launchKernel
export launch_device_fn.launchDeviceFn

# ═══════════════════════════════════════════════════════════════════════
#  Launch sugar
# ═══════════════════════════════════════════════════════════════════════

export launch_kernel.LaunchSugar,
       launch_kernel.`<<`, launch_kernel.`>>`
export launch_device_fn.DeviceFnSugar,
       launch_device_fn.DeviceFnProxy,
       launch_device_fn.`<<`, launch_device_fn.`>>`


# ═══════════════════════════════════════════════════════════════════════
#  Codegen
# ═══════════════════════════════════════════════════════════════════════

export gpu_compiler.cuda, gpu_compiler.opencl, gpu_compiler.glsl,
       gpu_compiler.webgpu, gpu_compiler.metal, gpu_compiler.hlsl,
       gpu_compiler.spirv
# exporting the type also exports the non-pure enum's members by bare name,
# individual enum fields cannot be exported ("enum field cannot be exported individually")
from ./codegen/backends/hlsl_lang import HlslFlavor
export HlslFlavor
