# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

type
  MetalGid = object
    x*, y*, z*: int

## Metal-native global thread id: `gid = blockIdx * blockDim + threadIdx`
## (the `[[thread_position_in_grid]]` idiom). Dummy object so that `gid.x/y/z` is typable inside the `metal:` macro.
## The metal_lang printer substitutes the composite expression at emission.
## Fields are plain `int` (Nim int64), so `gid.x` mixes with the layout templates' `int or Int` params
## without casts, matching the CUDA-family index dummies.
##
## `threadIdx`/`blockIdx`/`blockDim`/`gridDim` are not redefined here.
## The dummy objects from cuda_builtins.nim are shared by name (existing kernel source ports unchanged).
## The metal_lang printer maps each name to its MSL attribute parameter (tid/bid/bdim/gdim).
## Re-registering the names would make every use ambiguous across backends
## (two modules exporting the same `let` name break the use site).
let gid* {.builtin, compileTime.} = MetalGid()
