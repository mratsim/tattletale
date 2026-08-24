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
## The IR therefore only ever contains canonical names.
##
## The canonical coordinate type is the `uvec3` tuple, which supports `.x` and `[idx]` access and free destructuring.
## The `let {.builtin, compileTime.}` dummies exist so typed macro bodies typecheck.
## Their values are never evaluated or emitted.
## `thread_index_in_threadgroup` is the one plain builtin among them,
## not compileTime: host-side tile-op proc bodies call it at runtime
## (the tile ops run on the host in the ceramic tests). The printers
## forward it to the backend spelling like any other builtin.
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

# Not compileTime: host-side tile-op proc bodies call it at runtime,
# and a compileTime dummy would fold before the host link. The
# builtin marking and the device path are unchanged.
let thread_index_in_threadgroup* {.builtin.}: uint32 = 0'u32
  ## Flat thread index within the threadgroup.
  ##
  ## - `thread_position_in_threadgroup` linearized in x-major order
  ## - a canonical scalar builtin, not an expression template
  ## - the formula needs `threads_per_threadgroup` (no WGSL spelling)
  ##   so each printer emits its native flat index
  ##   (WGSL `local_invocation_index`)

let thread_index_in_simdgroup* {.builtin, compileTime.}: uint32 = 0'u32
  ## Flat lane index within the SIMD group (0..31 on Apple GPUs).
  ## Metal-only: the coordinate-builtin mechanism binds it as the kernel
  ## attribute `[[thread_index_in_simdgroup]]`, or a plain param on device
  ## functions. The other backends raise loudly (no equivalent).

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
  elif d == 2: thread_position_in_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_global_id".}
template get_group_id*(d: static uint32): uint32 =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `threadgroup_position_in_grid`.
  when d == 0: threadgroup_position_in_grid.x
  elif d == 1: threadgroup_position_in_grid.y
  elif d == 2: threadgroup_position_in_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_group_id".}
template get_local_id*(d: static uint32): uint32 =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `thread_position_in_threadgroup`.
  when d == 0: thread_position_in_threadgroup.x
  elif d == 1: thread_position_in_threadgroup.y
  elif d == 2: thread_position_in_threadgroup.z
  else:
    {.error: "Invalid dimension " & $d & " in get_local_id".}
template get_local_size*(d: static uint32): uint32 =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `threads_per_threadgroup`.
  when d == 0: threads_per_threadgroup.x
  elif d == 1: threads_per_threadgroup.y
  elif d == 2: threads_per_threadgroup.z
  else:
    {.error: "Invalid dimension " & $d & " in get_local_size".}
template get_num_groups*(d: static uint32): uint32 =
  ## OpenCL work-item dimension helper: static `d` folds to the matching component of `threadgroups_per_grid`.
  when d == 0: threadgroups_per_grid.x
  elif d == 1: threadgroups_per_grid.y
  elif d == 2: threadgroups_per_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_num_groups".}

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
# 1b. Simdgroup fragments (Apple GPU simdgroup MMAs)
# ═══════════════════════════════════════════════════════════
# Per-lane slice of a simdgroup matrix, V = 2 for the 8x8x8 atoms.
# isLayoutLeft: fragment memory in LEFT (row-major) order in fragment (row, col)
# coordinates, false for the col-major A/C operands, true for B. MSL transpose
# arg = not isLayoutLeft. Not a gather for make_filled / multiply_accumulate.
# Registered name-only: the MSL printer rewrites each call to the native intrinsic.

type SimdgroupMatrix*[T; isLayoutLeft: static bool] = object
  ## Per-lane register slice of a simdgroup matrix: V = 2 elements
  ## for the 8x8x8 atoms. Emitted as `simdgroup_float8x8` (f32) or
  ## `simdgroup_half8x8` (f16) by the MSL printer; a plain struct elsewhere.
  data*: array[2, T]

proc simdgroupLoad*[T; isLayoutLeft: static bool](frag: var SimdgroupMatrix[T, isLayoutLeft];
    src: ptr UncheckedArray[T];
    stride, offset: uint32;
    transpose: bool) {.builtin.} = discard
  ## Hardware fragment gather: `simdgroup_load(frag, src, stride, offset, transpose)`.
  ## The stride is the source row length, the offset an element offset from
  ## `src`, the transpose flag swaps the fragment's row and column axes
  ## against the source's (its value is `not isLayoutLeft`).

proc simdgroupStore*[T; isLayoutLeft: static bool](frag: SimdgroupMatrix[T, isLayoutLeft];
    dst: ptr UncheckedArray[T];
    stride, offset: uint32;
    transpose: bool) {.builtin.} = discard
  ## Hardware fragment scatter: `simdgroup_store(frag, dst, stride, offset, transpose)`.
  ## The stride is the destination row length, the offset an element
  ## offset from `dst`, the transpose flag swaps the fragment's row
  ## and column axes against the destination's.

proc simdgroupMultiplyAccumulate*[TD; TA; TB; isLayoutLeftD: static bool; isLayoutLeftA: static bool; isLayoutLeftB: static bool](
    d: var SimdgroupMatrix[TD, isLayoutLeftD];
    a: SimdgroupMatrix[TA, isLayoutLeftA];
    b: SimdgroupMatrix[TB, isLayoutLeftB]) {.builtin.} = discard
  ## One 8x8x8 MMA, in-place accumulate:
  ## `simdgroup_multiply_accumulate(d, a, b, d)`. The accumulator's
  ## element type may differ from the operands' (f32 accumulator over
  ## f16 operands).

proc makeFilledSimdgroupMatrix*[T; isLayoutLeft: static bool](val: T): SimdgroupMatrix[T, isLayoutLeft] {.builtin.} = discard
  ## Matrix filled with `val` on every lane:
  ## `make_filled_simdgroup_matrix<T, 8>(val)`. The isLayoutLeft param is
  ## carried through for type uniformity; it is not a gather.

# ═══════════════════════════════════════════════════════════
# 1c. Reduction builtins (cross-backend subgroup shuffles)
# ═══════════════════════════════════════════════════════════
# Per-lane gathers every SIMD backend spells natively: MSL
# simd_shuffle_down/simd_shuffle, CUDA __shfl_down_sync/__shfl_sync,
# OpenCL sub_group_shuffle_down/sub_group_shuffle, GLSL and WGSL
# subgroupShuffleDown/subgroupShuffle. The IR kinds live in
# GpuReductionBuiltinKind and the printers case on the kind alone.
# The Nim proc names are the canonical names (mission 13-01): only the
# IR kinds were re-homed, the consumers call by name, unchanged.

proc simdShuffleDown*[T](v: T; delta: uint32): T {.builtin.} = discard
  ## Returns, on each lane, the value of `v` held by the lane at
  ## `lane + delta` (`simd_shuffle_down`). Out-of-range sources
  ## return the calling lane's own value.

proc simdShuffle*[T](v: T; lane: uint32): T {.builtin.} = discard
  ## Returns, on each lane, the value of `v` held by the lane at
  ## absolute index `lane` (`simd_shuffle`). Only the active lanes
  ## must read `v`.

proc threadElements*[T; isLayoutLeft: static bool](
    frag: var SimdgroupMatrix[T, isLayoutLeft]; vpt: uint32): var T {.builtin.} =
  ## Returns the fragment's per-lane element at `vpt` as an lvalue:
  ## `threadElements(frag, vpt) = x` writes into the fragment, and
  ## later reads through the accessor observe that value. On the host
  ## the matrix is a plain struct and the call indexes its storage
  ## field. The MSL printer emits `frag.thread_elements()[vpt]`,
  ## the simdgroup matrix's per-lane element accessor.
  frag.data[vpt]

proc threadElements*[N: static int; T](
    frag: var array[N, T]; vpt: uint32): var T {.builtin.} =
  ## Returns the per-lane element at `vpt` of a plain per-lane value
  ## array (the FMA atom's fragment storage), as an lvalue. The lvalue
  ## aliases the array's `vpt`-th element, so writes through the accessor
  ## are visible to later reads of that element. The MSL printer emits
  ## `frag[vpt]`.
  frag[vpt]

proc threadElements*[T; isLayoutLeft: static bool](
    frag: SimdgroupMatrix[T, isLayoutLeft]; vpt: uint32): T {.builtin.} =
  ## Returns the fragment's per-lane element at `vpt`
  ## (read-only form, for non-var fragments). The MSL printer emits
  ## the same `frag.thread_elements()[vpt]` accessor as the lvalue
  ## form.
  frag.data[vpt]

proc threadElements*[N: static int; T](
    frag: array[N, T]; vpt: uint32): T {.builtin.} =
  ## Returns the per-lane element at `vpt` of a plain per-lane value
  ## array (read-only form, for non-var fragments). The MSL printer
  ## emits the same `frag[vpt]` accessor as the lvalue form.
  frag[vpt]

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
