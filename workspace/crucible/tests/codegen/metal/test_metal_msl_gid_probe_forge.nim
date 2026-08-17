# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Forged-pragma `gid` probe for the MSL printer.
##
## A user module-scope `let gid {.builtin.}` must never be treated as a backend builtin dummy.
## The `builtin` pragma alone is forgeable, because crucible exports a user-defined pragma template.
## The printer must reject the identifier, exactly like a kernel-local symbol named `gid`.
##
## This probe lives in its own module.
## A module-scope `let gid` shadows the builtin `gid` for the whole module.
## A declaration here would break the genuine-builtin kernels in test_metal_msl_compile.nim.

import workspace/crucible

let gid {.builtin.} = 7'u32
  ## Forged user symbol. Carries the `builtin` pragma but is defined in this user module.

## The positive control proves the `metal:` macro compiles in this module.
## The probe's failure below is therefore attributable to the `gid` reference.
const controlMsl = metal:
  proc controlKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 7'u32

static:
  # The forged-pragma module-scope `gid` referenced inside `metal:` must raise.
  # If a regression treats the pragma alone as dummy identity, the block compiles
  # and this assert fails.
  doAssert not compiles(block:
    const k = metal:
      proc gidForgeProbeKernel(output: ptr UncheckedArray[uint32]) {.global.} =
        output[0] = uint32(gid)
    discard k)
