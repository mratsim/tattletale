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
## Metal has no function-scope threadgroup storage: a kernel's `{.smem.}`
## local becomes a trailing `threadgroup` parameter, staged by the launch
## through the threadgroup memory argument, not a buffer slot.
##
## - `mslHoistProgramScopeConstants`: `{.const_mem.}` locals relocate into
##   `CirModule.globals` in encounter order, every proc and scope depth covered,
##   the printer emits the relocated declarations as program-scope `constant`
##   variables
## - program-scope `constant` storage is visible to all functions, a device
##   function reads the constants of the kernels it serves
## - `mslRewriteSmemLocals`: top-level `{.smem.}` locals leave every proc body,
##   a kernel gains one `threadgroup` parameter per local in body order while
##   a device function gains none
## - the signature order is regular parameters, threadgroup parameters,
##   builtin attributed parameters, and only the regular parameters take
##   buffer slots
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
    "Relocates smem kernel locals into trailing threadgroup kernel parameters",
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
  ## Relocates the top-level `{.smem.}` locals of every proc, one trailing
  ## `threadgroup` parameter per relocated local:
  ##
  ## - kernels gain one parameter per local in body order, a threadgroup
  ##   pointer to the local's declared type
  ## - the printer reads the pointee's array-ness to emit the pointer form
  ##   for an array local and the reference form otherwise
  ## - device functions lose the declaration without gaining a parameter,
  ##   their body keeps referencing the dropped name
  ## - an initializer on a relocated local is discarded
  ##
  ## - the relocation covers top-level body locals only, a nested one refuses
  ##   at the printer
  ## - the pass runs after the reserved-word rename, the appended parameters
  ##   use the renamed names their body already references
  for fn in m.procs:
    var i = 0
    while i < fn.body.len:
      let s = fn.body[i]
      if s.kind == sVar and s.space == caThreadgroup:
        if fn.isKernel:
          fn.params.add CirParam(name: s.name, typ: tPtr(caThreadgroup, s.declType))
        fn.body.delete i
      else:
        i.inc
  return ""
