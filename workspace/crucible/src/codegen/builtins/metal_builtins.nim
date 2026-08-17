# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

const MetalAttributeNames* = [
  "thread_position_in_threadgroup",
  "threadgroup_position_in_grid",
  "threads_per_threadgroup",
  "threadgroups_per_grid",
  "thread_position_in_grid",
]
  ## MSL thread-position attribute names, the single source of truth for the Metal index builtins.
  ## The dummies below declare matching identifiers. The metal_lang printer
  ## reads this list for its attribute-param appending and its reserved-name checks,
  ## so the names never live in the compiler as a second copy.

type
  MetalThreadPositionInThreadgroup* = object
    x*, y*, z*: uint32
  MetalThreadgroupPositionInGrid* = object
    x*, y*, z*: uint32
  MetalThreadsPerThreadgroup* = object
    x*, y*, z*: uint32
  MetalThreadgroupsPerGrid* = object
    x*, y*, z*: uint32
  MetalThreadPositionInGrid* = object
    x*, y*, z*: uint32

## Dummy objects to make the MSL thread-position attribute names typable
## in the typed `metal:` macro, with `uint32` fields matching the MSL attribute components.
## They cannot be `const`, because the typed code would evaluate the values
## before the macro can work with them. The `{.builtin.}` pragma follows the cuda_builtins and wgsl_builtins pattern.
## The dummy identifiers must match `MetalAttributeNames` above.
let thread_position_in_threadgroup* {.builtin, compileTime.} = MetalThreadPositionInThreadgroup()
let threadgroup_position_in_grid* {.builtin, compileTime.} = MetalThreadgroupPositionInGrid()
let threads_per_threadgroup* {.builtin, compileTime.} = MetalThreadsPerThreadgroup()
let threadgroups_per_grid* {.builtin, compileTime.} = MetalThreadgroupsPerGrid()
let thread_position_in_grid* {.builtin, compileTime.} = MetalThreadPositionInGrid()
