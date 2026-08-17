# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal anti-regression, gid forged-pragma probe.
##
## A user module-scope `let gid {.builtin.}` must never be treated
## as a backend builtin dummy. The `builtin` pragma alone is forgeable,
## because crucible exports a user-defined pragma template.
## A rewrite compiles cleanly and silently changes device results,
## the silent-miscompile class this test pins. The printer rejects
## the identifier exactly like a kernel-local symbol named `gid`.
##
## The probe is standalone. It declares a module-scope `let gid`
## shadowing the builtin `gid` for the whole module, so the shadowing
## is confined to the probe's own compilation.
##
## The negative probe is a compile-time assert using `doAssert not compiles`
## in a `static:` block. The binary builds only when the reference raises,
## so a successful build is the pass.
## The runtime body is the tester convention.
##
## Tested ABI (macOS, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_anti_regression_gid_forge.nim

import std/strutils

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

proc runTest() =
  # The compile-time assert above is the real verification. This body follows
  # the tester convention and pins the positive control's MSL emission.
  doAssert "controlKernel" in controlMsl
  echo "anti-regression gid forge: OK"

when isMainModule:
  runTest()
