# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Curated public surface of crucible.
##
## | Section       | Symbols                                                                                                                                           |
## | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
## | Engines       | `newEngine`, `CodegenKind`, `HwEngine`, `ingest`, `getArtifact`, `deviceName`                                                                     |
## | Device memory | `DeviceMem`, `allocDevice`, `copyToDeviceFromHost`, `copyToHostFromDevice`                                                                        |
## | Launch sugar  | `launchKernel`, `launchDeviceFn`, `launchCpuFn`, chevrons `<<`/`>>`, `LaunchSugar`, `DeviceFnSugar`, `DeviceFnProxy`, `LaunchConfig`              |
## | BatchCommands | `recordBatch`, `enqueue`, `submit`, `replay`, chevrons `<<`/`>>`, `BatchCommands`, `BatchEntry`, `EnqueueSugar`, `EnqueueProxy`, `EnqueueProxyCt` |
## | Geometry      | `Dim3`, `dim3`                                                                                                                                    |
## | Codegen       | `cuda`, `opencl`, `glsl`, `webgpu`, `metal`, `hlsl` (+ `HlslFlavor`), `spirv`                                                                     |

import
  ./builtins/builtins_gpu_types,
  ./builtins/builtins_compilermagic,
  ./codegen/gpu_compiler,
  ./runtime/engines/engines_api,
  ./runtime/launch/launch_kernel,
  ./runtime/launch/launch_device_fn,
  ./runtime/launch/launch_cpu_fn,
  ./runtime/batching/batch

# ═══════════════════════════════════════════════════════════════════════
#  Engines
# ═══════════════════════════════════════════════════════════════════════

export engines_api

# GPU builtin vocabulary: scalar types and compile-time backend queries
export builtins_gpu_types.float16, builtins_gpu_types.bfloat16,
       builtins_gpu_types.`==`, builtins_gpu_types.to,
       builtins_compilermagic.CompileTarget, builtins_compilermagic.ccGetBackend,
       builtins_compilermagic.crucibleCompileTarget

export launch_kernel.launchKernel
export launch_device_fn.launchDeviceFn
export launch_cpu_fn.launchCpuFn

# ═══════════════════════════════════════════════════════════════════════
#  Launch sugar
# ═══════════════════════════════════════════════════════════════════════

export launch_kernel.LaunchSugar,
       launch_kernel.`<<`, launch_kernel.`>>`
export launch_device_fn.DeviceFnSugar,
       launch_device_fn.DeviceFnProxy,
       launch_device_fn.`<<`, launch_device_fn.`>>`

export batch.recordBatch, batch.enqueue, batch.`<<`, batch.`>>`, batch.submit,
       batch.replay, batch.ctgHasStagedKernels
export batch.BatchCommands, batch.BatchEntry, batch.EnqueueSugar, batch.EnqueueProxy, batch.EnqueueProxyCt


# ═══════════════════════════════════════════════════════════════════════
#  Codegen
# ═══════════════════════════════════════════════════════════════════════

export gpu_compiler.cuda, gpu_compiler.opencl, gpu_compiler.glsl,
       gpu_compiler.webgpu, gpu_compiler.metal, gpu_compiler.hlsl,
       gpu_compiler.spirv
# exporting the type also exports the non-pure enum's members by name,
# individual enum fields cannot be exported ("enum field cannot be exported individually")
from ./codegen/backends/lang_hlsl import HlslFlavor
export HlslFlavor
