# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

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

## Dummy objects to make the MSL thread-position attribute names typecheck
## in the typed `metal:` macro, with `uint32` fields matching the MSL attribute components.
## They cannot be `const`, because the typed code would evaluate the values
## before the macro can work with them.
let thread_position_in_threadgroup* {.builtin, compileTime.} = MetalThreadPositionInThreadgroup()
let threadgroup_position_in_grid* {.builtin, compileTime.} = MetalThreadgroupPositionInGrid()
let threads_per_threadgroup* {.builtin, compileTime.} = MetalThreadsPerThreadgroup()
let threadgroups_per_grid* {.builtin, compileTime.} = MetalThreadgroupsPerGrid()
let thread_position_in_grid* {.builtin, compileTime.} = MetalThreadPositionInGrid()
