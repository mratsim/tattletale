# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU builtin vocabulary available inside kernel bodies, the stable
## surface downstream users bind against.
##
## | Section      | Symbols                                                                                    |
## | ------------ | ------------------------------------------------------------------------------------------ |
## | Launch       | `launchKernel`, `launchDeviceFn`, chevrons `<<`/`>>`                                       |
## | Scalar types | `float16`, `bfloat16`, `fp8e4m3`, `fp8e5m2`, `toFp16`, `toFp32`, `asFp16`, `asU16`         |
## | Coordinates  | `ivec3`                                                                                    |
## | Math         | `exp2`, `rsqrt`                                                                            |
## | Atomics      | `atomicFetchAdd`                                                                           |
## | Pragmas      | `global`, `workgroup`, `device`, `forceinline`, `smem`, `const_mem`, `builtin`, `cudaName` |
## | Comp. magic  | `CompileTarget`, `ccGetBackend`                                                            |
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


import
  ./builtins/builtins_catalog,
  ./builtins/builtins_functions,
  ./builtins/builtins_gpu_types,
  ./builtins/builtins_warp_sync_reductions,
  ./builtins/builtins_atomics,
  ./builtins/builtins_pragmas,
  ./builtins/builtins_compilermagic

export builtins_catalog

export builtins_functions.exp2, builtins_functions.rsqrt

export builtins_gpu_types.float16, builtins_gpu_types.bfloat16,
       builtins_gpu_types.fp8e4m3, builtins_gpu_types.fp8e5m2,
       builtins_gpu_types.toFp16, builtins_gpu_types.toFp32,
       builtins_gpu_types.toBf16, builtins_gpu_types.toBf32,
       builtins_gpu_types.toFp8e4m3, builtins_gpu_types.toFp8e5m2,
       builtins_gpu_types.asFp16, builtins_gpu_types.asU16,
       builtins_gpu_types.to

export builtins_warp_sync_reductions.reduceAdd, builtins_warp_sync_reductions.reduceMax,
       builtins_warp_sync_reductions.simd_shuffle, builtins_warp_sync_reductions.simd_shuffle_xor,
       builtins_warp_sync_reductions.subgroupShuffle, builtins_warp_sync_reductions.subgroupShuffleXor,
       builtins_warp_sync_reductions.shfl_sync, builtins_warp_sync_reductions.shfl_xor_sync,
       builtins_warp_sync_reductions.WaveReadLaneAt

export builtins_atomics

export builtins_pragmas

export builtins_compilermagic.CompileTarget, builtins_compilermagic.ccGetBackend
