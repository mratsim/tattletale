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
## and the `threadgroup_barrier` / `threadgroup_barrier_device` pair.
## These are the MSL attribute names, so the Metal printer emits them verbatim.
## Every other backend spelling, such as CUDA `blockIdx`, OpenCL `get_global_id`,
## GLSL `gl_GlobalInvocationID`, or WGSL `global_id`, is a template alias.
## Nim's semantic analysis expands each alias to the canonical name before the typed GPU macro sees the body.
## The IR therefore only ever contains canonical names.
##
## The canonical coordinate type is the `ivec3` tuple, which supports `.x` and `[idx]` access and free destructuring.
## The `let {.builtin, compileTime.}` dummies exist so typed macro bodies typecheck.
## Their values are never evaluated or emitted.
## `thread_index_in_threadgroup` is the one plain builtin among them,
## not compileTime: host-side tile-op proc bodies call it at runtime
## (the tile ops run on the host in the ceramic tests). The printers
## forward it to the backend spelling like any other builtin.
## Calls to `{.builtin.}` procs are registered name-only by the compiler
## and forwarded to the backend spelling by each printer.

type
  ivec3* = tuple[x, y, z: int]
    ## Canonical GPU coordinate vector, int domain:
    ## - crucible lowers int to the target's int width, int32 in CUDA C
    ## - positions and dimensions are structural grid coordinates,
    ##   never wraparound arithmetic inputs
    ## - tuple so `.x` and `[idx]` access both work, `nim_to_gpu` normalizes
    ##   tuple `[idx]` to a field access

# ═══════════════════════════════════════════════════════════
# Canonical dummies: the MSL vocabulary
# ═══════════════════════════════════════════════════════════
# `let {.builtin, compileTime.}` not `const`. The typed macro would fold a const value
# before codegen can work with the symbol.

let thread_position_in_grid* {.builtin, compileTime.}: ivec3 = default(ivec3)
  ## Global thread index: `threadgroup_position_in_grid × threads_per_threadgroup + thread_position_in_threadgroup` (per axis).
let threadgroup_position_in_grid* {.builtin, compileTime.}: ivec3 = default(ivec3)
  ## Threadgroup index within the grid.
let thread_position_in_threadgroup* {.builtin, compileTime.}: ivec3 = default(ivec3)
  ## Thread index within its threadgroup.
let threads_per_threadgroup* {.builtin, compileTime.}: ivec3 = default(ivec3)
  ## Threadgroup size, per axis.
let threadgroups_per_grid* {.builtin, compileTime.}: ivec3 = default(ivec3)
  ## Grid size, in threadgroups, per axis.

# Not compileTime: host-side tile-op proc bodies call it at runtime,
# and a compileTime dummy would fold before the host link. The
# builtin marking and the device path are unchanged.
let thread_index_in_threadgroup* {.builtin.}: int = 0
  ## Flat thread index within the threadgroup.
  ##
  ## - `thread_position_in_threadgroup` linearized in x-major order
  ## - a canonical scalar builtin, not an expression template
  ## - the formula needs `threads_per_threadgroup` (no WGSL spelling)
  ##   so each printer emits its native flat index
  ##   (WGSL `local_invocation_index`)

proc threadgroup_barrier*() {.builtin.} = discard
  ## Canonical workgroup barrier ordering threadgroup-address-space accesses.
  ##
  ## Each backend printer emits its native spelling
  ## (e.g. MSL `threadgroup_barrier(mem_flags::mem_threadgroup)`).
  ##
  ## Device-address-space exchanges need `threadgroup_barrier_device`, not
  ## this barrier, since `mem_threadgroup` does not order device stores.

proc threadgroup_barrier_device*()
    {.builtin.} =
  ## Workgroup barrier that orders device-address-space accesses.
  ##
  ## Precondition:
  ## the buffers this fence orders live in the device address space,
  ## the reverse fence is the plain `threadgroup_barrier` barrier.
  discard

# ═══════════════════════════════════════════════════════════
# 1. Coordinates
# ═══════════════════════════════════════════════════════════

# CUDA idiom
template blockIdx*(): untyped = threadgroup_position_in_grid
template blockDim*(): untyped = threads_per_threadgroup
template gridDim*(): untyped = threadgroups_per_grid
template threadIdx*(): untyped = thread_position_in_threadgroup

# OpenCL idiom
template get_global_id*(d: static int): int =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `thread_position_in_grid`.
  when d == 0: thread_position_in_grid.x
  elif d == 1: thread_position_in_grid.y
  elif d == 2: thread_position_in_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_global_id".}
template get_group_id*(d: static int): int =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `threadgroup_position_in_grid`.
  when d == 0: threadgroup_position_in_grid.x
  elif d == 1: threadgroup_position_in_grid.y
  elif d == 2: threadgroup_position_in_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_group_id".}
template get_local_id*(d: static int): int =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `thread_position_in_threadgroup`.
  when d == 0: thread_position_in_threadgroup.x
  elif d == 1: thread_position_in_threadgroup.y
  elif d == 2: thread_position_in_threadgroup.z
  else:
    {.error: "Invalid dimension " & $d & " in get_local_id".}
template get_local_size*(d: static int): int =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `threads_per_threadgroup`.
  when d == 0: threads_per_threadgroup.x
  elif d == 1: threads_per_threadgroup.y
  elif d == 2: threads_per_threadgroup.z
  else:
    {.error: "Invalid dimension " & $d & " in get_local_size".}
template get_num_groups*(d: static int): int =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `threadgroups_per_grid`.
  when d == 0: threadgroups_per_grid.x
  elif d == 1: threadgroups_per_grid.y
  elif d == 2: threadgroups_per_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_num_groups".}

# GLSL idiom
template gl_GlobalInvocationID*(): untyped = thread_position_in_grid
template gl_LocalInvocationIndex*(): untyped = thread_index_in_threadgroup

# WGSL idiom
template global_id*(): untyped = thread_position_in_grid
# No `workgroup_size` alias: WGSL has no `workgroup_size` builtin:
# only the `@workgroup_size` attribute exists, so `threads_per_threadgroup` is deferred on WGSL.

# ═══════════════════════════════════════════════════════════
# 2. Synchronization
# ═══════════════════════════════════════════════════════════

# CUDA idiom
template syncthreads*(): untyped = threadgroup_barrier()

# OpenCL idiom
const CLK_LOCAL_MEM_FENCE* = 1'u32

template barrier*(flags: static uint32 = CLK_LOCAL_MEM_FENCE): untyped =
  ## OpenCL work-group barrier alias. the zero-arg form also serves the
  ## Vulkan idiom.
  ## Note: Only CLK_LOCAL_MEM_FENCE is supported
  ## as CLK_GLOBAL_MEM_FENCE has no equivalent
  static: doAssert flags == CLK_LOCAL_MEM_FENCE
  threadgroup_barrier()


# as `__builtin_memcpy`, never via the dummy.

