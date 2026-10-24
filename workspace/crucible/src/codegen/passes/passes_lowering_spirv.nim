# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            SPIR-V lowering passes
#
# ############################################################
#
# Lowers the crucible IR for direct SPIR-V binary emission.
# The bundle registers the SPIR-V pipeline's passes:
# - the shared lowering trio plus insertPointerDeref
# - no text passes:
#   binary output has no keyword collisions
#   composite stores come from the printer itself
#   Function storage is the native pointer ABI, no inout workaround needed
# - no instantiateImplicitAddressSpaceGenerics:
#   its gpuIndex retyping over pointers yields the container type
#   struct constructions then grow unchecked-array fields
#   the emitted composites mismatch the member types

import ./pass_datatypes
import ./pass_registry
import ./passes_lowering
import ./passes_low_bools
import ./passes_low_compound_assign
import ./passes_low_pointers
import ./passes_low_views

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSpirvLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = false)
  reg.registerSharedLoweringPasses()
  # Compound assignments rewrite to plain assignments of the binop before
  # pointer-deref insertion, both copies of the address skeleton retype alike
  reg.registerLowerCompoundAssign()
  reg.registerInsertPointerDeref()
  # Bool kernel parameters declare i32 in the push-constant interface,
  # loads convert at the use boundary
  reg.registerPatchBoolToI32()
