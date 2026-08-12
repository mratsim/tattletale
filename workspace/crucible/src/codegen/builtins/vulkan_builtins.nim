## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

## Vulkan (GLSL) built-in identifiers used within `vulkan:` macro bodies.
## These are dummy symbols so that the typed macro expansion succeeds.
## They are replaced by actual GLSL builtins during codegen.

# Vector type definitions (dummy for macro typing)
type
  uvec3* = array[3, uint32]


when not declaredInScope(vulkan):
  # Work-item builtins. `let {.builtin.}` rather than value templates,
  # since a discard-body template has no type in the typed `vulkan` macro.
  let gl_GlobalInvocationID* {.builtin, compileTime.} = default(uvec3)
  let gl_LocalInvocationID* {.builtin, compileTime.} = default(uvec3)
  let gl_WorkGroupID* {.builtin, compileTime.} = default(uvec3)
  let gl_WorkGroupSize* {.builtin, compileTime.} = default(uvec3)
  let gl_NumWorkGroups* {.builtin, compileTime.} = default(uvec3)
  let gl_LocalInvocationIndex* {.builtin, compileTime.} = 0'u32

  ## Synchronization
  template barrier*(): void = discard
  template memoryBarrierBuffer*(): void = discard
  template memoryBarrierShared*(): void = discard
  template groupMemoryBarrier*(): void = discard

  ## Atomic operations
  template atomicAdd*(obj: ptr uint32, operand: uint32): uint32 = discard
  template atomicSub*(obj: ptr uint32, operand: uint32): uint32 = discard
  template atomicExchange*(obj: ptr uint32, value: uint32): uint32 = discard
