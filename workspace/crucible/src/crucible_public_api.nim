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
## | Engines       | `newEngine`, `BackendKind`, `HwEngine`, `ingest`, `getArtifact`, `deviceName`                                         |
## | Device memory | `DeviceMem`, `allocDevice`, `copyToDeviceFromHost`, `copyToHostFromDevice`                                            |
## | Launch sugar  | `launchKernel`, `launchDeviceFn`, chevrons `<<`/`>>`, `LaunchSugar`, `DeviceFnSugar`, `DeviceFnProxy`, `LaunchConfig` |
## | Autodiff      | `bwd_diff`, `KernelPair`, `BwdParamMeta`, `BwdParamKind`, `BwdGradKind`, `BwdExtentSource`, `setupAD`, `AdHandle`     |
## | Geometry      | `Dim3`, `dim3`                                                                                                        |
## | Codegen       | `cuda`, `opencl`, `glsl`, `webgpu`, `metal`, `hlsl` (+ `HlslFlavor`)                                                  |

import
  ./codegen/gpu_compiler,
  ./codegen/kernel_pair,
  ./runtime/engines/engines_api,
  ./runtime/launch/launch_kernel,
  ./runtime/launch/launch_device_fn,
  ./runtime/ad_handle

# ═══════════════════════════════════════════════════════════════════════
#  Engines
# ═══════════════════════════════════════════════════════════════════════

export engines_api

export launch_kernel.launchKernel
export launch_device_fn.launchDeviceFn
export launch_device_fn.bwd_diff

export kernel_pair.KernelPair, kernel_pair.BwdParamMeta, kernel_pair.BwdParamKind,
       kernel_pair.BwdGradKind, kernel_pair.BwdExtentSource,
       kernel_pair.ReduceSource, kernel_pair.ScalarWorkspace

export ad_handle.AdHandle, ad_handle.setupAD, ad_handle.forward, ad_handle.backward

# ═══════════════════════════════════════════════════════════════════════
#  Launch sugar
# ═══════════════════════════════════════════════════════════════════════

export launch_kernel.LaunchSugar,
       launch_kernel.`<<`, launch_kernel.`>>`
export launch_device_fn.DeviceFnSugar,
       launch_device_fn.DeviceFnProxy, launch_device_fn.`<<`, launch_device_fn.`>>`


# ═══════════════════════════════════════════════════════════════════════
#  Codegen
# ═══════════════════════════════════════════════════════════════════════

export gpu_compiler.cuda, gpu_compiler.opencl, gpu_compiler.glsl,
       gpu_compiler.webgpu, gpu_compiler.metal, gpu_compiler.hlsl
# the type export carries the non-pure enum's members to importers by bare name,
# individual enum fields cannot be exported ("enum field cannot be exported individually")
from ./codegen/backends/hlsl_lang import HlslFlavor
export HlslFlavor
