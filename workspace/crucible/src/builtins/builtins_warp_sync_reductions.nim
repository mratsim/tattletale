# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas
import ./builtins_catalog

# ═══════════════════════════════════════════════════════════════════════
#   Warp-synchronous reductions
# ═══════════════════════════════════════════════════════════════════════

proc reduceAdd*[T](v: T): T {.builtin: cbkReduceAdd.} = discard
proc reduceMax*[T](v: T): T {.builtin: cbkReduceMax.} = discard

# ═══════════════════════════════════════════════════════════════════════
#   Cross-lane shuffles
# ═══════════════════════════════════════════════════════════════════════

# simd_shuffle backend catalogue:
#
# | Backend       | Backend syntax                                                                    |
# | ------------- | --------------------------------------------------------------------------------- |
# | MSL           | `simd_shuffle(value, simd_lane_id)`                                               |
# | GLSL (Vulkan) | `subgroupShuffle(value, id)` under `GL_KHR_shader_subgroup_shuffle`               |
# | WGSL          | `subgroupShuffle(value, sourceIndex)`, subgroups feature                          |
# | OpenCL        | `sub_group_shuffle(value, index)`                                                 |
# | CUDA          | `__shfl_sync(0xffffffff, value, lane)`                                            |
# | SPIR-V        | `OpGroupNonUniformShuffle` (opcode 345), capability `GroupNonUniformShuffle` (65) |
# | HLSL          | `WaveReadLaneAt(value, lane)`                                                     |

# simd_shuffle_xor backend catalogue:
#
# | Backend       | Backend syntax                                                                       |
# | ------------- | ------------------------------------------------------------------------------------ |
# | MSL           | `simd_shuffle_xor` `(value, mask)`                                                   |
# | GLSL (Vulkan) | `subgroupShuffleXor(value, mask)` under `GL_KHR_shader_subgroup_shuffle`             |
# | WGSL          | `subgroupShuffleXor(value, mask)`, subgroups feature                                 |
# | OpenCL        | `sub_group_shuffle_xor(value, mask)`                                                 |
# | CUDA          | `__shfl_xor_sync(0xffffffff, value, mask)`                                           |
# | SPIR-V        | `OpGroupNonUniformShuffleXor` (opcode 346), capability `GroupNonUniformShuffle` (65) |
# | HLSL          | refused loudly: no wave intrinsic exchanges with a lane-index xor                    |

proc simd_shuffle*[T](value: T, lane: int): T {.builtin: cbkSimdShuffle.} = discard
proc simd_shuffle_xor*[T](value: T, mask: int): T {.builtin: cbkSimdShuffleXor.} = discard
template subgroupShuffle*[T](value: T, lane: int): untyped = simd_shuffle(value, lane)
template subgroupShuffleXor*[T](value: T, mask: int): untyped = simd_shuffle_xor(value, mask)
template shfl_sync*[T](value: T, lane: int): untyped = simd_shuffle(value, lane)
template shfl_xor_sync*[T](value: T, mask: int): untyped = simd_shuffle_xor(value, mask)
template WaveReadLaneAt*[T](value: T, lane: int): untyped = simd_shuffle(value, lane)
