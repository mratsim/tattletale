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
# Lower the crucible IR for direct SPIR-V binary emission, registered by the SPIR-V pipeline only.
#
# The bundle registers the shared lowering trio: lowerArraysAndViews,
# the shared lowering passes, insertPointerDeref. The text passes stay out:
# - renameReservedIdentifiers: binary output has no keyword collisions
# - decomposeArrayAssign: the printer emits composite stores itself
# - qualifyVarParams: GLSL-only inout text-legality workaround, SPIR-V has the native
#   Function-storage pointer ABI
# - the GLSL keyword validations
#
# instantiateImplicitAddressSpaceGenerics stays out while master's getOperandType
# retypes struct field inits through the pointee container: its gpuIndex branch
# over a pointer yields the container itself, so a `Vec2(x: a[0], ...)` construction
# grows unchecked-array fields and the emitted composites mismatch the member types.
# The frontend tags kernel pointer params with their space and the printer picks
# storage classes by role, so no per-space instantiation is needed today.

import ./pass_datatypes
import ./pass_registry
import ./passes_lowering
import ./passes_low_pointers
import ./passes_low_views

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSpirvLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = false)
  reg.registerSharedLoweringPasses()
  reg.registerInsertPointerDeref()
