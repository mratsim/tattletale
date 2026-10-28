# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## MSL lowering: per-target transforms for the Metal storage model,
## registered by the Metal pipeline and run before printing.
##
##   kernel body → mslHoistProgramScopeConstants → mslRewriteSmemLocals → printed MSL source
##
## - `mslHoistProgramScopeConstants`: `{.const_mem.}` locals relocate into
##   `CirModule.globals` in encounter order, every proc and scope depth covered,
##   the printer emits the relocated declarations as program-scope `constant`
##   variables
## - program-scope `constant` storage is visible to all functions, a device
##   function reads the constants of the kernels it serves
## - `mslRewriteSmemLocals`: a kernel's top-level `{.smem.}` local stays
##   in-body, Metal allocates the kernel-scope `threadgroup` storage,
##   no launch-time staging
## - a device function's top-level `{.smem.}` local drops: the inline splice
##   already merged the callee's body into the calling kernel
## - an initializer on a `{.smem.}` local is discarded
##
## Refusal homes:
## - a const_mem local without an initializer refuses in the pass, the printer
##   keeps a nil-init shape guard on the relocated declarations
## - an initializer on a relocated smem local is discarded, a device-function
##   smem local drops without a parameter, both reproduce the printer shapes
##   that the extraction replaces
## - a const_mem local surviving outside the relocation refuses at the printer,
##   a surviving smem local drops, the Metal compiler rejects the dangling read

import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerMslLoweringPasses*(reg: var PassRegistry) =
  ## Registers the MSL per-target lowering passes:
  ## - `mslHoistProgramScopeConstants` then `mslRewriteSmemLocals`, before printing
  ##
  ## Registration contract: the passes sit after the reserved-word rename,
  ## relocated declaration and appended parameter names match the renamed body references.
  reg.register("mslHoistProgramScopeConstants", pkTransform, phaseMain,
    "Relocates const_mem locals into the module's program scope",
    run = mslHoistProgramScopeConstants
  )
  reg.register("mslRewriteSmemLocals", pkTransform, phaseMain,
    "Keeps kernel smem locals in-body as threadgroup storage, drops device-fn smem locals",
    run = mslRewriteSmemLocals
  )

# ═══════════════════════════════════════════════════════════════════════
# Const_mem locals to program-scope constants
# ═══════════════════════════════════════════════════════════════════════

proc mslHoistProgramScopeConstants(m: CirModule): string =
  ## Relocates the `{.const_mem.}` locals of every proc into `m.globals` in encounter order,
  ## the declarations dropped from their bodies.
  ##
  ## Contract:
  ## - every proc and scope depth covered, a device-function or nested-scope
  ##   local relocates like a kernel-top-level one
  ## - the initializer is required, a const_mem local without one refuses
  ## - runs after the reserved-word rename, relocated declaration names match
  ##   the renamed body references
  hoistConstMemLocals(m)

# ═══════════════════════════════════════════════════════════════════════
# Smem locals to threadgroup parameters
# ═══════════════════════════════════════════════════════════════════════

proc mslRewriteSmemLocals(m: CirModule): string =
  ## Rewrites the top-level `{.smem.}` locals of every proc:
  ##
  ## - a kernel keeps the local in its body: the printer emits kernel-scope
  ##   `threadgroup` storage Metal allocates per threadgroup,
  ##   no launch-time staging and no parameter
  ## - a device function loses the declaration without gaining a parameter:
  ##   the inline splice already merged the callee's body into the calling kernel,
  ##   which owns the arena
  ## - an initializer on a `{.smem.}` local is discarded
  ##
  ## - the rewrite covers top-level body locals only, a nested one refuses
  ##   at the printer
  ## - the pass runs after the reserved-word rename: the kept locals carry
  ##   the renamed names their body already references
  for fn in m.procs:
    var i = 0
    while i < fn.body.len:
      let s = fn.body[i]
      if s.kind == sVar and s.space == caThreadgroup:
        if fn.isKernel:
          s.init = nil
          i.inc
        else:
          fn.body.delete i
      else:
        i.inc
  return ""
