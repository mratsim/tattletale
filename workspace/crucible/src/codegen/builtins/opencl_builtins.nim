# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

## OpenCL built-in identifiers used within `opencl:` macro bodies.
## These are dummy symbols so that the typed macro expansion succeeds.
## They are replaced by actual OpenCL C builtins during codegen.

when not declaredInScope(opencl):
  ## Work-item dimension helpers
  ## Procs (not templates): a template body would be expanded away by the
  ## typed `opencl:` body sem and the call would never reach codegen. The
  ## names are registered in NimGpuFnBuiltins so calls forward to the
  ## OpenCL C builtins by name (the bodies below are never emitted).
  proc get_global_id*(dim: uint32): uint32 = discard
  proc get_local_id*(dim: uint32): uint32 = discard
  proc get_group_id*(dim: uint32): uint32 = discard
  proc get_local_size*(dim: uint32): uint32 = discard
  proc get_global_size*(dim: uint32): uint32 = discard
  proc get_num_groups*(dim: uint32): uint32 = discard

  ## Synchronization
  template barrier*(flags: uint32): void = discard

  ## Memory fences
  template mem_fence*(flags: uint32): void = discard
  template read_mem_fence*(flags: uint32): void = discard
  template write_mem_fence*(flags: uint32): void = discard

  ## Math builtins (explicit overloads)
  template clamp*(x, minVal, maxVal: float32): float32 = discard
  template clamp*(x, minVal, maxVal: int32): int32 = discard
  template select*(f, t: float32, cond: bool): float32 = discard
  template select*(f, t: int32, cond: bool): int32 = discard

  ## Atomic operations
  template atomic_add*(obj: ptr uint32, operand: uint32): uint32 = discard
  template atomic_sub*(obj: ptr uint32, operand: uint32): uint32 = discard
  template atomic_xchg*(obj: ptr uint32, value: uint32): uint32 = discard

  ## Vector type helpers
  template workgroup_barrier*(flags: uint32): void = discard
