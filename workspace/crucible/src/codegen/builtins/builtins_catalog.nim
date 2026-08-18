# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

## Unified GPU coordinate and synchronization vocabulary.
##
## Canonical names are the MSL vocabulary:
## `thread_position_in_grid`, `threadgroup_position_in_grid`,
## `thread_position_in_threadgroup`, `threads_per_threadgroup`,
## `threadgroups_per_grid`, `thread_index_in_threadgroup`,
## and `threadgroup_barrier`.
## These are the MSL attribute names, so the Metal printer emits them verbatim.
## Every other backend spelling, such as CUDA `blockIdx`, OpenCL `get_global_id`,
## GLSL `gl_GlobalInvocationID`, or WGSL `global_id`, is a template alias.
## Nim's sem expands each alias to the canonical name before the typed GPU macro sees the body.
## The IR therefore only ever contains canonical names, with no alias tables or name→kind maps in the compiler.
##
## The canonical coordinate type is the `uvec3` tuple, which supports `.x` and `[idx]` access and free destructuring.
## The `let {.builtin, compileTime.}` dummies exist so typed macro bodies typecheck.
## Their values are never evaluated or emitted.
## Calls to `{.builtin.}` procs are registered name-only by the compiler
## and forwarded to the backend spelling by each printer.

type
  uvec3* = tuple[x, y, z: uint32]
    ## Canonical GPU coordinate vector. Tuple so `.x` and `[idx]` access both work.
    ## `nim_to_gpu` normalizes tuple `[idx]` to a field access.

# ═══════════════════════════════════════════════════════════
# Canonical dummies: the MSL vocabulary
# ═══════════════════════════════════════════════════════════
# `let {.builtin, compileTime.}` not `const`. The typed macro would fold a const value
# before codegen can work with the symbol.

let thread_position_in_grid* {.builtin, compileTime.}: uvec3 = default(uvec3)
  ## Global thread index: `threadgroup_position_in_grid × threads_per_threadgroup + thread_position_in_threadgroup` (per axis).
let threadgroup_position_in_grid* {.builtin, compileTime.}: uvec3 = default(uvec3)
  ## Threadgroup index within the grid.
let thread_position_in_threadgroup* {.builtin, compileTime.}: uvec3 = default(uvec3)
  ## Thread index within its threadgroup.
let threads_per_threadgroup* {.builtin, compileTime.}: uvec3 = default(uvec3)
  ## Threadgroup size, per axis.
let threadgroups_per_grid* {.builtin, compileTime.}: uvec3 = default(uvec3)
  ## Grid size, in threadgroups, per axis.

let thread_index_in_threadgroup* {.builtin, compileTime.}: uint32 = 0'u32
  ## Flat thread index within the threadgroup.
  ##
  ## - `thread_position_in_threadgroup` linearized in x-major order
  ## - a canonical scalar builtin, not an expression template
  ## - the formula needs `threads_per_threadgroup` (no WGSL spelling)
  ##   so each printer emits its native flat index
  ##   (WGSL `local_invocation_index`)

proc threadgroup_barrier*() {.builtin.} = discard
  ## Canonical workgroup barrier. Each backend printer emits its native spelling
  ## (e.g. MSL `threadgroup_barrier(mem_flags::mem_threadgroup)`).

# ═══════════════════════════════════════════════════════════
# 1. Coordinates
# ═══════════════════════════════════════════════════════════

# CUDA idiom
template blockIdx*(): untyped = threadgroup_position_in_grid
template blockDim*(): untyped = threads_per_threadgroup
template gridDim*(): untyped = threadgroups_per_grid
template threadIdx*(): untyped = thread_position_in_threadgroup

# OpenCL idiom
template get_global_id*(d: static uint32): uint32 =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `thread_position_in_grid`.
  when d == 0: thread_position_in_grid.x
  elif d == 1: thread_position_in_grid.y
  else:        thread_position_in_grid.z
template get_group_id*(d: static uint32): uint32 =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `threadgroup_position_in_grid`.
  when d == 0: threadgroup_position_in_grid.x
  elif d == 1: threadgroup_position_in_grid.y
  else:        threadgroup_position_in_grid.z
template get_local_id*(d: static uint32): uint32 =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `thread_position_in_threadgroup`.
  when d == 0: thread_position_in_threadgroup.x
  elif d == 1: thread_position_in_threadgroup.y
  else:        thread_position_in_threadgroup.z
template get_local_size*(d: static uint32): uint32 =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `threads_per_threadgroup`.
  when d == 0: threads_per_threadgroup.x
  elif d == 1: threads_per_threadgroup.y
  else:        threads_per_threadgroup.z
template get_num_groups*(d: static uint32): uint32 =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `threadgroups_per_grid`.
  when d == 0: threadgroups_per_grid.x
  elif d == 1: threadgroups_per_grid.y
  else:        threadgroups_per_grid.z

# GLSL idiom
template gl_GlobalInvocationID*(): untyped = thread_position_in_grid
template gl_WorkGroupID*(): untyped = threadgroup_position_in_grid
template gl_LocalInvocationID*(): untyped = thread_position_in_threadgroup
template gl_WorkGroupSize*(): untyped = threads_per_threadgroup
template gl_NumWorkGroups*(): untyped = threadgroups_per_grid
template gl_LocalInvocationIndex*(): untyped = thread_index_in_threadgroup

# WGSL idiom
template global_id*(): untyped = thread_position_in_grid
template workgroup_id*(): untyped = threadgroup_position_in_grid
template local_invocation_id*(): untyped = thread_position_in_threadgroup
template num_workgroups*(): untyped = threadgroups_per_grid
template local_invocation_index*(): untyped = thread_index_in_threadgroup
# No `workgroup_size` alias: WGSL has no `workgroup_size` builtin:
# only the `@workgroup_size` attribute exists, so `threads_per_threadgroup` is deferred on WGSL.

# ═══════════════════════════════════════════════════════════
# 2. Synchronization
# ═══════════════════════════════════════════════════════════

# CUDA idiom
template syncthreads*(): untyped = threadgroup_barrier()

# OpenCL idiom
template barrier*(flags: uint32): untyped = threadgroup_barrier()
  ## OpenCL work-group barrier alias: the flag argument folds away, so `barrier(flags)` matches the OpenCL call shape.

# Vulkan idiom
template barrier*(): untyped = threadgroup_barrier()

# OpenCL 2.0 / WGSL idiom
template workgroup_barrier*(flags: uint32 = 0): untyped = threadgroup_barrier()
  ## One declaration serves both arities. The WGSL `workgroupBarrier`
  ## and the OpenCL `workgroup_barrier(flags)` are identifier-equal
  ## under Nim's style-insensitive matching.

# ═══════════════════════════════════════════════════════════
# 3. Other (CUDA-only)
# ═══════════════════════════════════════════════════════════
# `malloc`/`free`/`memcpy` dummies were dropped: memcpy is emitted only by the `decomposeMemcpy` pass
# as `__builtin_memcpy`, never via the dummy.

proc printf*(fmt: string) {.varargs, builtin.} = discard
  ## CUDA device-side `printf`. Forwards the format string and arguments verbatim.
proc cvtaGenericToShared*(p: pointer): uint32 {.cudaName: "__cvta_generic_to_shared", builtin.} = discard
  ## CUDA's `__cvta_generic_to_shared`. Converts a generic pointer to a shared-memory address
  ## and returns the 32-bit shared offset.
