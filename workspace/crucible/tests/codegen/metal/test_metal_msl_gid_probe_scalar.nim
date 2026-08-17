# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Module-scope `gid` probe for the MSL printer, scalar variant.
##
## A user module-scope `let gid` referenced inside a `metal:` block must never be rewritten to the builtin composite.
## The printer rejects the identifier, exactly like a kernel-local symbol named `gid`.
##
## This probe lives in its own module.
## A module-scope `let gid` shadows the builtin `gid` for the whole module.
## A declaration here would break the genuine-builtin kernels in test_metal_msl_compile.nim.

import workspace/crucible

let gid = 7'u32
  ## Module-scope user symbol named like the Metal index builtin.

## The positive control proves the `metal:` macro compiles in this module.
## The probe's failure below is therefore attributable to the `gid` reference.
const controlMsl = metal:
  proc controlKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 7'u32

static:
  # The module-scope user `gid` referenced inside `metal:` must raise.
  # If a regression rewrites it to the builtin composite, the block compiles
  # and this assert fails.
  doAssert not compiles(block:
    const k = metal:
      proc gidProbeKernel(output: ptr UncheckedArray[uint32]) {.global.} =
        output[0] = uint32(gid)
    discard k)
