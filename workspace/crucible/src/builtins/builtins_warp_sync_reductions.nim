# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

# ═══════════════════════════════════════════════════════════════════════
#   Warp-synchronous reductions
# ═══════════════════════════════════════════════════════════════════════

proc reduceAdd*[T](v: T): T {.builtin.} = discard
  ## Reduces `v` across the subgroup with addition and broadcasts
  ## the total to every active lane. The value of `v` is only read
  ## on the active lanes.
  ##
  ## Backends emit: MSL `simd_sum`, GLSL `subgroupAdd`,
  ## WGSL `subgroupAdd`, OpenCL `sub_group_reduce_add`.
  ## CUDA lowers to a `__shfl_xor_sync` butterfly over deltas 16/8/4/2/1.

proc reduceMax*[T](v: T): T {.builtin.} = discard
  ## Reduces `v` across the subgroup with max and broadcasts
  ## the result to every active lane. The value of `v` is only read
  ## on the active lanes.
  ##
  ## Backends emit: MSL `simd_max`, GLSL `subgroupMax`,
  ## WGSL `subgroupMax`, OpenCL `sub_group_reduce_max`.
  ## CUDA lowers to a `__shfl_xor_sync` butterfly over deltas 16/8/4/2/1.

# ═══════════════════════════════════════════════════════════════════════
#   Cross-lane shuffles
# ═══════════════════════════════════════════════════════════════════════

proc simd_shuffle*[T](value: T, lane: int): T {.builtin.} = discard
  ## Gathers `value` from the lane at absolute subgroup-relative
  ## index `lane` and returns it on every lane of the subgroup.
  ##
  ## Contract:
  ## - `lane` need not be uniform across the subgroup on any backend.
  ## - `lane` must name an existing lane: an index outside the subgroup
  ##   is undefined (MSL, OpenCL, WGSL) or poison (SPIR-V).
  ## - Keep `lane` in `[0, subgroupSize)` so every lane reads a defined value.
  ##
  ## Backend catalogue:
  ##
  ## | Backend       | Spelling                                                                          | Uniformity                                                                                               |
  ## | ------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
  ## | MSL           | `simd_shuffle(value, simd_lane_id)`                                               | none: a valid lane id may differ per thread (MSL 4.1 stdlib, `simd_shuffle`)                             |
  ## | GLSL (Vulkan) | `subgroupShuffle(value, id)` under `GL_KHR_shader_subgroup_shuffle`               | none in the SPIR-V 1.6 lowering target, the KHR extension text is not vendored in this repository        |
  ## | WGSL          | `subgroupShuffle(value, sourceIndex)`, subgroups feature                          | WGSL spec not vendored, native wgpu accepts the call without a directive                                 |
  ## | OpenCL        | `sub_group_shuffle(value, index)`                                                 | OpenCL 2.1 core, no pragma                                                                               |
  ## | CUDA          | `__shfl_sync(0xffffffff, value, lane)`                                            | full mask: every warp lane participates and receives the result                                          |
  ## | SPIR-V        | `OpGroupNonUniformShuffle` (opcode 345), capability `GroupNonUniformShuffle` (65) | none in core SPIR-V 1.6: the result is poison only when the index leaves the subgroup                    |
  ## | HLSL          | `WaveReadLaneAt(value, lane)`                                                     | lane dynamically uniform: the same value on every warp lane (HLSL docs, see Slang wave-intrinsics notes) |

proc simd_shuffle_xor*[T](value: T, mask: int): T {.builtin.} = discard
  ## Pairs each lane with the lane whose subgroup-relative index is
  ## `callerLane xor mask` and exchanges `value`, returning the partner
  ## value on every lane.
  ##
  ## Contract:
  ## - `mask` must be uniform across the subgroup: a non-uniform mask
  ##   pairs lanes inconsistently. Metal requires uniformity outright
  ##   (MSL 4.1 stdlib, `simd_shuffle_xor`), the other backends stay
  ##   well-defined only under it.
  ## - `mask` must stay below the subgroup size so the paired lane exists.
  ## - Butterfly passes over masks 16/8/4/2/1 are the reduction primitive:
  ##   CUDA lowers `reduceAdd` and `reduceMax` to `__shfl_xor_sync`
  ##   butterflies over exactly those masks.
  ##
  ## Backend catalogue:
  ##
  ## | Backend       | Spelling                                                                             | Uniformity                                                                                                 |
  ## | ------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
  ## | MSL           | `simd_shuffle_xor(value, mask)`                                                      | required: the mask must be the same on every thread of the SIMD-group (MSL 4.1 stdlib, `simd_shuffle_xor`) |
  ## | GLSL (Vulkan) | `subgroupShuffleXor(value, mask)` under `GL_KHR_shader_subgroup_shuffle`             | no clause in the SPIR-V 1.6 lowering target, the KHR extension text is not vendored in this repository     |
  ## | WGSL          | `subgroupShuffleXor(value, mask)`, subgroups feature                                 | WGSL spec not vendored, native wgpu accepts the call without a directive                                   |
  ## | OpenCL        | `sub_group_shuffle_xor(value, mask)`                                                 | OpenCL 2.1 core, no pragma                                                                                 |
  ## | CUDA          | `__shfl_xor_sync(0xffffffff, value, mask)`                                           | full mask: every warp lane participates, the exchange pairs lanes bitwise under a uniform mask             |
  ## | SPIR-V        | `OpGroupNonUniformShuffleXor` (opcode 346), capability `GroupNonUniformShuffle` (65) | no uniformity clause in core SPIR-V 1.6, the caller keeps the mask uniform                                 |
  ## | HLSL          | refused loudly: no wave intrinsic exchanges with a lane-index xor                    | not applicable, HLSL has no faithful xor spelling                                                          |

template subgroupShuffle*[T](value: T, lane: int): untyped = simd_shuffle(value, lane)
template subgroupShuffleXor*[T](value: T, mask: int): untyped = simd_shuffle_xor(value, mask)
template shfl_sync*[T](value: T, lane: int): untyped = simd_shuffle(value, lane)
template shfl_xor_sync*[T](value: T, mask: int): untyped = simd_shuffle_xor(value, mask)
template WaveReadLaneAt*[T](value: T, lane: int): untyped = simd_shuffle(value, lane)
