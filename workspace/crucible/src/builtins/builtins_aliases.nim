# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Coordinate builtin aliases: the CUDA, GLSL, and WGSL names of the public
## contract table in `crucible_gpu_builtins.nim` as `{.builtin.}` symbols.
##
## Each alias is a `let` whose initializer references its canonical MSL symbol:
## - the declaration itself records the alias relation, enumerated by
##   `builtinAliasPairs` for the alias-canonicalization pass
## - the canonical set the kernel-only-reads check and the printers key on is
##   `CanonicalCoordBuiltins`, enumerated from the catalog declarations
## - the pass rewrites every alias read to the MSL name before legalization
##   and printing

import ./builtins_catalog
import ./builtins_pragmas
import std/[macros, tables]

# ═══════════════════════════════════════════════════════════════════════
# Alias declarations
# ═══════════════════════════════════════════════════════════════════════

let threadIdx* {.builtin, compileTime.}: ivec3 = thread_position_in_threadgroup
  ## CUDA name of `thread_position_in_threadgroup`.
let blockIdx* {.builtin, compileTime.}: ivec3 = threadgroup_position_in_grid
  ## CUDA name of `threadgroup_position_in_grid`.
let blockDim* {.builtin, compileTime.}: ivec3 = threads_per_threadgroup
  ## CUDA name of `threads_per_threadgroup`.
let gridDim* {.builtin, compileTime.}: ivec3 = threadgroups_per_grid
  ## CUDA name of `threadgroups_per_grid`.

let gl_GlobalInvocationID* {.builtin, compileTime.}: ivec3 = thread_position_in_grid
  ## GLSL name of `thread_position_in_grid`.
let gl_WorkGroupID* {.builtin, compileTime.}: ivec3 = threadgroup_position_in_grid
  ## GLSL name of `threadgroup_position_in_grid`.
let gl_LocalInvocationID* {.builtin, compileTime.}: ivec3 = thread_position_in_threadgroup
  ## GLSL name of `thread_position_in_threadgroup`.
let gl_WorkGroupSize* {.builtin, compileTime.}: ivec3 = threads_per_threadgroup
  ## GLSL name of `threads_per_threadgroup`.
let gl_NumWorkGroups* {.builtin, compileTime.}: ivec3 = threadgroups_per_grid
  ## GLSL name of `threadgroups_per_grid`.
let gl_LocalInvocationIndex* {.builtin.}: int = thread_index_in_threadgroup
  ## GLSL name of `thread_index_in_threadgroup`.

let global_id* {.builtin, compileTime.}: ivec3 = thread_position_in_grid
  ## WGSL name of `thread_position_in_grid`.
let workgroup_id* {.builtin, compileTime.}: ivec3 = threadgroup_position_in_grid
  ## WGSL name of `threadgroup_position_in_grid`.
let local_invocation_id* {.builtin, compileTime.}: ivec3 = thread_position_in_threadgroup
  ## WGSL name of `thread_position_in_threadgroup`.
let num_workgroups* {.builtin, compileTime.}: ivec3 = threadgroups_per_grid
  ## WGSL name of `threadgroups_per_grid`.
let local_invocation_index* {.builtin.}: int = thread_index_in_threadgroup
  ## WGSL name of `thread_index_in_threadgroup`.

# ═══════════════════════════════════════════════════════════════════════
# Relation derived from the declarations
# ═══════════════════════════════════════════════════════════════════════

proc unwrapConvNode(n: NimNode): NimNode =
  ## Peels the implicit conversion nodes sem wraps around an initializer.
  result = n
  while result.kind in {nnkHiddenSubConv, nnkHiddenStdConv, nnkConv}:
    result = result[^1]

macro builtinAliasPairs*(syms: varargs[typed]): untyped =
  ## Args:
  ## - `syms`: the `{.builtin.}` alias lets, each initialized from its canonical symbol
  ##
  ## Returns the (alias, canonical) name pairs read from the initializers at compile time, consumed by the alias-canonicalization pass.
  result = newNimNode(nnkBracket)
  for s in syms:
    let impl = s.getImpl
    let init = unwrapConvNode(impl[^1])
    if init.kind != nnkSym:
      error("builtin alias " & s.strVal &
        " must initialize from its canonical symbol, got " & $init.kind)
    result.add newPar(newLit(s.strVal), newLit(init.strVal))

macro builtinSymNames*(syms: varargs[typed]): untyped =
  ## Args:
  ## - `syms`: the canonical `{.builtin.}` catalog lets and procs, generic
  ##   procs included (an overload choice unwraps to its first candidate)
  ##
  ## Returns their declared names as a bracket of string literals, one per symbol.
  result = newNimNode(nnkBracket)
  for s in syms:
    var n = s
    if n.kind in {nnkOpenSymChoice, nnkClosedSymChoice}:
      n = n[0]
    result.add newLit(n.strVal)

const BuiltinAliasPairs* = builtinAliasPairs(
  threadIdx, blockIdx, blockDim, gridDim,
  gl_GlobalInvocationID, gl_WorkGroupID, gl_LocalInvocationID,
  gl_WorkGroupSize, gl_NumWorkGroups, gl_LocalInvocationIndex,
  global_id, workgroup_id, local_invocation_id, num_workgroups,
  local_invocation_index)

const CanonicalCoordBuiltins* = builtinSymNames(
  thread_position_in_grid,
  threadgroup_position_in_grid,
  thread_position_in_threadgroup,
  threads_per_threadgroup,
  threadgroups_per_grid,
  thread_index_in_threadgroup)

proc aliasCanonicalTable*(): Table[string, string] =
  ## Returns the map from alias name to canonical MSL name over the coordinate builtins.
  for pair in BuiltinAliasPairs:
    result[pair[0]] = pair[1]

proc isCanonicalCoordBuiltin*(name: string): bool =
  ## Returns true when `name` is one of the canonical coordinate builtins, the MSL names in the IR after alias canonicalization.
  name in CanonicalCoordBuiltins
