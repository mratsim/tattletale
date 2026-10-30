# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU kernel code generator, emits native GPU code: CUDA, OpenCL, GLSL, WebGPU, Metal.
##
## Public API, one import: `import workspace/crucible`.
## The curated surface lives in `src/crucible_public_api.nim`, the GPU builtin vocabulary
## in `src/crucible_gpu_builtins.nim`, both re-exported here in full.
## Everything not exported there is internal, only compiler tests reach into the internals.
##
## | Section      | Symbols                                                                                   |
## | ------------ | ----------------------------------------------------------------------------------------- |
## | Launch       | `launchKernel`, `launchDeviceFn`, chevrons `<<`/`>>`                                      |
## | Scalar types | `float16`, `bfloat16`, `fp8e4m3`, `fp8e5m2`, `toFp16`, `toFp32`, `asFp16`, `asU16`        |
## | Coordinates  | `ivec3`                                                                                   |
## | Math         | `exp2`, `rsqrt`                                                                           |
## | Atomics      | `atomicFetchAdd`                                                                          |
## | Pragmas      | `global`, `workgroup`, `device`, `forceinline`, `smem`, `const_mem`, `builtin`, `exportc` |
## | Comp. magic  | `CompileTarget`, `ccGetBackend`                                                           |
##
## | Metal Shading language (MSL)     | CUDA                                                                         | OpenCL                                                                                                        | Vulkan (GLSL)             | WGSL                                                   |
## | -------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------ |
## | `thread_position_in_grid`        | `(blockIdx.d*blockDim.d+threadIdx.d)`                                        | `get_global_id(d)`                                                                                            | `gl_GlobalInvocationID`   | `global_id`                                            |
## | `threadgroup_position_in_grid`   | `blockIdx`                                                                   | `get_group_id(d)`                                                                                             | `gl_WorkGroupID`          | `workgroup_id`                                         |
## | `thread_position_in_threadgroup` | `threadIdx`                                                                  | `get_local_id(d)`                                                                                             | `gl_LocalInvocationID`    | `local_invocation_id`                                  |
## | `threads_per_threadgroup`        | `blockDim`                                                                   | `get_local_size(d)`                                                                                           | `gl_WorkGroupSize`        | deferred (no WGSL builtin)                             |
## | `threadgroups_per_grid`          | `gridDim`                                                                    | `get_num_groups(d)`                                                                                           | `gl_NumWorkGroups`        | `num_workgroups`                                       |
## | `thread_index_in_threadgroup`    | `(threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x)` | `(get_local_id(2)*get_local_size(0)*get_local_size(1) + get_local_id(1)*get_local_size(0) + get_local_id(0))` | `gl_LocalInvocationIndex` | `local_invocation_index`                               |
## | `threadgroup_barrier`            | `__syncthreads()`                                                            | `barrier(CLK_LOCAL_MEM_FENCE)`                                                                                | `barrier()`               | `workgroupBarrier()`                                   |
## | `reduceAdd`                      | `warpReduceAdd` (emitted `__forceinline__` template)                         | `sub_group_reduce_add`                                                                                        | `subgroupAdd`             | `subgroupAdd` (no directive; subgroups device feature) |
## | `reduceMax`                      | `warpReduceMax` (emitted `__forceinline__` template)                         | `sub_group_reduce_max`                                                                                        | `subgroupMax`             | `subgroupMax` (no directive; subgroups device feature) |

##
## Backends: cuda: / opencl: / glsl: / webgpu: / metal:
## - var engine = newEngine(runCuda)
## - engine.ingest(kernelCode)
## - engine.launchKernel<<grid, blk>>("addKernel", out, (a, b))
## - engine.launchDeviceFn myDeviceFunction(out, args) do:
##     grid = (gx, gy)
##     blk = (bx, by)

import ./src/crucible_public_api
import ./src/crucible_gpu_builtins
export crucible_public_api
export crucible_gpu_builtins
