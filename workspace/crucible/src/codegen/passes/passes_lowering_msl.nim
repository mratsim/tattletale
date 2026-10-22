# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            MSL lowering passes
#
# ############################################################
#
# Project the crucible IR onto Metal's calling convention:
# device-function `{.smem.}` vars hoist to threadgroup params,
# coordinate builtins materialize as appended params,
# pointer-carrying structs expand to ptr/scalar leaves.

import ./pass_datatypes
import ./pass_registry
import ./passes_msl
import ./passes_legalizations

proc registerMetalLoweringPasses*(reg: var PassRegistry) =
  reg.registerExpandStructKernelParams()
  reg.register("smemArena", pkTransform, phaseMain,
    "Hoists device functions' {.smem.} vars to threadgroup params as Metal doesn't support threadgroup allocation in device functions",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      smemArenaImpl(ctx)
  )
  reg.register("materializeIndexBuiltinParams", pkTransform, phaseMain,
    "Materializes coordinate-builtin binding: builtin params appended to pParams, call-site forwarding args",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      materializeIndexBuiltinParamsImpl(ctx)
  )
