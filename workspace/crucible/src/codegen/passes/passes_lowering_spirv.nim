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
# Compared to the GLSL bundle the SPIR-V bundle drops the text-only passes:
# - renameReservedIdentifiers: binary output has no keyword collisions
# - decomposeArrayAssign: the printer emits composite stores itself
# - qualifyVarParams: GLSL-only inout text-legality workaround, SPIR-V has the native Function-storage pointer ABI
# - the GLSL keyword validations
#
# The bundle also skips instantiateImplicitAddressSpaceGenerics:
# - the frontend stamps kernel pointer params with their space
# - the printer picks storage classes by role, no per-space instantiation needed
# - the pass retypes struct field inits through getOperandType, whose gpuIndex
#   branch over a pointer yields the pointee container itself
# - a `Vec2(x: a[0], ...)` construction then grows unchecked-array fields,
#   the emitted composites mismatch their member types

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
