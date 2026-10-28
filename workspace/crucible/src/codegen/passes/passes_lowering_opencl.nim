# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## OpenCL lowering: per-target transforms for the OpenCL C storage model,
## registered by the OpenCL pipeline and run before printing.
##
##   kernel body → openclHoistProgramScopeConstants → printed OpenCL C source
##
## - `openclHoistProgramScopeConstants`: `{.const_mem.}` locals relocate into
##   `CirModule.globals` in encounter order, every proc and scope depth covered,
##   the printer emits the relocated declarations as program-scope
##   `__constant` between the typedefs and the forward declarations
## - program-scope `__constant` storage is visible to all functions, a device
##   function reads the constants of the kernels it serves
##
## `{.smem.}` locals are untouched, a mechanical `__local` kernel-body
## declaration the printer emits directly.
##
## Refusal homes:
## - a const_mem local without an initializer refuses in the pass, the printer
##   keeps a nil-init shape guard on the relocated declarations
## - a const_mem local surviving outside the relocation refuses at the printer,
##   the walk covers every proc and scope

import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerOpenclLoweringPasses*(reg: var PassRegistry) =
  ## Registers the OpenCL per-target lowering pass:
  ## - `openclHoistProgramScopeConstants`, before printing
  ##
  ## Registration contract: the pass sits after the reserved-word rename,
  ## relocated declaration names match the renamed body references.
  reg.register("openclHoistProgramScopeConstants", pkTransform, phaseMain,
    "Relocates const_mem locals into the module's program scope",
    run = openclHoistProgramScopeConstants
  )

# ═══════════════════════════════════════════════════════════════════════
# Const_mem locals to program-scope constants
# ═══════════════════════════════════════════════════════════════════════

proc openclHoistProgramScopeConstants(m: CirModule): string =
  ## Relocates the `{.const_mem.}` locals of every proc into `m.globals`.
  ## Encounter order, the declarations dropped from their bodies.
  ##
  ## Contract:
  ## - every proc and scope depth covered, a device-function or nested-scope
  ##   local relocates like a kernel-top-level one
  ## - the initializer is required, a const_mem local without one refuses
  ## - runs after the reserved-word rename, relocated declaration names match
  ##   the renamed body references
  hoistConstMemLocals(m)
