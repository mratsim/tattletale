# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas
import std/macros


type
  ivec3* = tuple[x, y, z: int]
    ## Canonical GPU coordinate vector, int domain:
    ## - crucible lowers int to the target's int width, int32 in CUDA C
    ## - tuple so `.x` and `[idx]` access both work, `nim_to_crucible` normalizes
    ##   tuple `[idx]` to a field access

# ═══════════════════════════════════════════════════════════════════════
# Builtin kinds: the family enums that {.builtin.} payloads name
# ═══════════════════════════════════════════════════════════════════════

type
  CirCoordBuiltinKind* = enum
    cbkNone
    cbkThreadPositionInGrid
    cbkThreadgroupPositionInGrid
    cbkThreadPositionInThreadgroup
    cbkThreadsPerThreadgroup
    cbkThreadgroupsPerGrid
    cbkThreadIndexInThreadgroup

  CirSynchroBuiltinKind* = enum
    cbkNone
    cbkThreadgroupBarrier
    cbkThreadgroupBarrierDevice

  CirReductionBuiltinKind* = enum
    cbkNone
    cbkReduceAdd
    cbkReduceMax

  CirShuffleBuiltinKind* = enum
    cbkNone
    cbkSimdShuffle
    cbkSimdShuffleXor

  CirAtomicsBuiltinKind* = enum
    cbkNone
    cbkAtomicFetchAdd

macro builtinKindLadder*(E: typedesc, fname: static string): untyped =
  ## Generates `proc fname*(payload: NimNode): E {.compileTime.}`.
  ##
  ## Args:
  ## - `E`: the family enum, its members enumerate the ladder
  ## - `fname`: the exported name of the generated ladder proc
  ##
  ## Returns:
  ## - an `eqIdent` ladder over the members of `E`
  ## - the first member of `E` (the none value) when `payload` matches
  ##   none of them
  ##
  ## `payload`:
  ## - the argument of a `{.builtin: <member>.}` pragma, read back
  ##   through `getImpl`
  ## - the nil literal for a `{.builtin.}` without a payload
  ##
  ## The ladder enumerates the enum members, so the payload identifiers
  ## and the enum members cannot drift apart.
  let enumTy = E.getImpl[2]
  if enumTy.kind != nnkEnumTy:
    error("builtinKindLadder expects an enum type, found " & $enumTy.kind)
  var members: seq[string]
  for i in 1 ..< enumTy.len:
    let m = enumTy[i]
    case m.kind
    of nnkIdent, nnkSym:
      members.add m.strVal
    of nnkEnumFieldDef:
      members.add m[0].strVal
    else:
      error("unexpected enum member shape " & $m.kind)
  var body = newTree(nnkIfStmt)
  for name in members:
    body.add newTree(nnkElifBranch,
      newCall(bindSym"eqIdent", ident"payload", newLit(name)),
      newTree(nnkStmtList, newAssignment(ident"result", ident(name))))
  body.add newTree(nnkElse,
    newTree(nnkStmtList, newAssignment(ident"result", ident(members[0]))))
  result = newTree(nnkProcDef,
    postfix(ident(fname), "*"),
    newEmptyNode(),
    newEmptyNode(),
    newTree(nnkFormalParams,
      ident(E.strVal),
      newTree(nnkIdentDefs, ident"payload", ident"NimNode", newEmptyNode())),
    newTree(nnkPragma, ident"compileTime"),
    newEmptyNode(),
    newTree(nnkStmtList, body))

builtinKindLadder(CirCoordBuiltinKind, "coordBuiltinKindOf")
builtinKindLadder(CirSynchroBuiltinKind, "synchroBuiltinKindOf")
builtinKindLadder(CirReductionBuiltinKind, "reductionBuiltinKindOf")
builtinKindLadder(CirShuffleBuiltinKind, "shuffleBuiltinKindOf")
builtinKindLadder(CirAtomicsBuiltinKind, "atomicsBuiltinKindOf")

# ═══════════════════════════════════════════════════════════
# 0. Canonical thread coordinates (Metal Shading Language)
# ═══════════════════════════════════════════════════════════

let thread_position_in_grid* {.builtin: cbkThreadPositionInGrid, compileTime.}: ivec3 = default(ivec3)
  ## Global thread index: `threadgroup_position_in_grid × threads_per_threadgroup + thread_position_in_threadgroup` (per axis).
let threadgroup_position_in_grid* {.builtin: cbkThreadgroupPositionInGrid, compileTime.}: ivec3 = default(ivec3)
let thread_position_in_threadgroup* {.builtin: cbkThreadPositionInThreadgroup, compileTime.}: ivec3 = default(ivec3)
let threads_per_threadgroup* {.builtin: cbkThreadsPerThreadgroup, compileTime.}: ivec3 = default(ivec3)
let threadgroups_per_grid* {.builtin: cbkThreadgroupsPerGrid, compileTime.}: ivec3 = default(ivec3)

# Not compileTime: host-side tile-op proc bodies call it at runtime,
# a compileTime dummy would fold before the host link.
let thread_index_in_threadgroup* {.builtin: cbkThreadIndexInThreadgroup.}: int = 0
  ## Flat thread index within the threadgroup:
  ## `thread_position_in_threadgroup` linearized in x-major order.
  ## Each printer emits its native flat index (WGSL `local_invocation_index`).

proc threadgroup_barrier*() {.builtin: cbkThreadgroupBarrier.} = discard
  ## Orders threadgroup-address-space accesses.
  ## Device-address-space exchanges need `threadgroup_barrier_device`:
  ## The backend's threadgroup flag does not order device stores.

proc threadgroup_barrier_device*()
    {.builtin: cbkThreadgroupBarrierDevice.} =
  ## Orders device-address-space accesses,
  ## the reverse fence of `threadgroup_barrier`.
  discard

# ═══════════════════════════════════════════════════════════
# 1. Coordinates
# ═══════════════════════════════════════════════════════════

template blockIdx*(): untyped = threadgroup_position_in_grid
template blockDim*(): untyped = threads_per_threadgroup
template gridDim*(): untyped = threadgroups_per_grid
template threadIdx*(): untyped = thread_position_in_threadgroup

template get_global_id*(d: static int): int =
  when d == 0: thread_position_in_grid.x
  elif d == 1: thread_position_in_grid.y
  elif d == 2: thread_position_in_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_global_id".}
template get_group_id*(d: static int): int =
  when d == 0: threadgroup_position_in_grid.x
  elif d == 1: threadgroup_position_in_grid.y
  elif d == 2: threadgroup_position_in_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_group_id".}
template get_local_id*(d: static int): int =
  when d == 0: thread_position_in_threadgroup.x
  elif d == 1: thread_position_in_threadgroup.y
  elif d == 2: thread_position_in_threadgroup.z
  else:
    {.error: "Invalid dimension " & $d & " in get_local_id".}
template get_local_size*(d: static int): int =
  when d == 0: threads_per_threadgroup.x
  elif d == 1: threads_per_threadgroup.y
  elif d == 2: threads_per_threadgroup.z
  else:
    {.error: "Invalid dimension " & $d & " in get_local_size".}
template get_num_groups*(d: static int): int =
  when d == 0: threadgroups_per_grid.x
  elif d == 1: threadgroups_per_grid.y
  elif d == 2: threadgroups_per_grid.z
  else:
    {.error: "Invalid dimension " & $d & " in get_num_groups".}

template gl_GlobalInvocationID*(): untyped = thread_position_in_grid
template gl_LocalInvocationIndex*(): untyped = thread_index_in_threadgroup

template global_id*(): untyped = thread_position_in_grid
# No `workgroup_size` alias: WGSL has no `workgroup_size` builtin,
# only the `@workgroup_size` attribute, so `threads_per_threadgroup` is deferred on WGSL.

# ═══════════════════════════════════════════════════════════
# 2. Synchronization
# ═══════════════════════════════════════════════════════════

template syncthreads*(): untyped = threadgroup_barrier()

const CLK_LOCAL_MEM_FENCE* = 1'u32

template barrier*(flags: static uint32 = CLK_LOCAL_MEM_FENCE): untyped =
  ## OpenCL work-group barrier, the zero-arg form also serves the Vulkan idiom.
  ## Only CLK_LOCAL_MEM_FENCE is supported: CLK_GLOBAL_MEM_FENCE has no equivalent.
  static: doAssert flags == CLK_LOCAL_MEM_FENCE
  threadgroup_barrier()
