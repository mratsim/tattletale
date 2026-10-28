# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas


type
  ivec3* = tuple[x, y, z: int]
    ## Canonical GPU coordinate vector, int domain:
    ## - crucible lowers int to the target's int width, int32 in CUDA C
    ## - tuple so `.x` and `[idx]` access both work, `nim_to_crucible` normalizes
    ##   tuple `[idx]` to a field access

# ═══════════════════════════════════════════════════════════════════════
# 0. Canonical thread coordinates (Metal Shading Language)
# ═══════════════════════════════════════════════════════════════════════
#
# {.builtin.} are repl

let thread_position_in_grid* {.builtin, compileTime.}: ivec3 = default(ivec3)
  ## Global thread index: `threadgroup_position_in_grid × threads_per_threadgroup + thread_position_in_threadgroup` (per axis).
let threadgroup_position_in_grid* {.builtin, compileTime.}: ivec3 = default(ivec3)
let thread_position_in_threadgroup* {.builtin, compileTime.}: ivec3 = default(ivec3)
let threads_per_threadgroup* {.builtin, compileTime.}: ivec3 = default(ivec3)
let threadgroups_per_grid* {.builtin, compileTime.}: ivec3 = default(ivec3)

# Not compileTime: host-side tile-op proc bodies call it at runtime,
# a compileTime dummy would fold before the host link.
let thread_index_in_threadgroup* {.builtin.}: int = 0
  ## Flat thread index within the threadgroup:
  ## `thread_position_in_threadgroup` linearized in x-major order.
  ## Each printer emits its native flat index (WGSL `local_invocation_index`).

proc threadgroup_barrier*() {.builtin.} = discard
  ## Orders threadgroup-address-space accesses.
  ## Device-address-space exchanges need `threadgroup_barrier_device`:
  ## The backend's threadgroup flag does not order device stores.

proc threadgroup_barrier_device*()
    {.builtin.} =
  ## Orders device-address-space accesses,
  ## the reverse fence of `threadgroup_barrier`.
  discard

# ═══════════════════════════════════════════════════════════════════════
# 1. Coordinates
# ═══════════════════════════════════════════════════════════════════════

template get_global_id*(d: static int): int =
  when d == 0:
    thread_position_in_grid.x
  elif d == 1:
    thread_position_in_grid.y
  elif d == 2:
    thread_position_in_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_global_id".}
template get_group_id*(d: static int): int =
  when d == 0:
    threadgroup_position_in_grid.x
  elif d == 1:
    threadgroup_position_in_grid.y
  elif d == 2:
    threadgroup_position_in_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_group_id".}
template get_local_id*(d: static int): int =
  when d == 0:
    thread_position_in_threadgroup.x
  elif d == 1:
    thread_position_in_threadgroup.y
  elif d == 2:
    thread_position_in_threadgroup.z
  else:
    {.error: "Invalid dimension " & $d & " in get_local_id".}
template get_local_size*(d: static int): int =
  when d == 0:
    threads_per_threadgroup.x
  elif d == 1:
    threads_per_threadgroup.y
  elif d == 2:
    threads_per_threadgroup.z
  else:
    {.error: "Invalid dimension " & $d & " in get_local_size".}
template get_num_groups*(d: static int): int =
  when d == 0:
    threadgroups_per_grid.x
  elif d == 1:
    threadgroups_per_grid.y
  elif d == 2:
    threadgroups_per_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_num_groups".}

# `{.builtin.}` alias declarations in `builtins_aliases` cover the CUDA/GLSL/WGSL
# coordinate names. The alias relation derives from those declarations alone.
# No parallel name table exists.
# No `workgroup_size` alias: WGSL has no `workgroup_size` builtin,
# only the `@workgroup_size` attribute, so `threads_per_threadgroup` is deferred on WGSL.

# ═══════════════════════════════════════════════════════════════════════
# 2. Synchronization
# ═══════════════════════════════════════════════════════════════════════

template syncthreads*(): untyped = threadgroup_barrier()

const CLK_LOCAL_MEM_FENCE* = 1'u32

template barrier*(flags: static uint32 = CLK_LOCAL_MEM_FENCE): untyped =
  ## OpenCL work-group barrier, the zero-arg form also serves the Vulkan idiom.
  ## Only CLK_LOCAL_MEM_FENCE is supported: CLK_GLOBAL_MEM_FENCE has no equivalent.
  static: doAssert flags == CLK_LOCAL_MEM_FENCE
  threadgroup_barrier()
