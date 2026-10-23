# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            CUDA lowering passes
#
# ############################################################
#
# Project the crucible IR onto CUDA's calling convention:
# non-lvalue passByRef args materialize at call sites, CUDA
# keeps `const Type&` params natively.

import std / [sets]
import ../ir/ast_datatypes
import ./pass_datatypes
import ./pass_registry
import ./passes_lowering
import ./passes_instantiations
import ./passes_optimizations

{.experimental: "codeReordering".}
const CudaReserved = [
  "__global__", "__device__", "__shared__", "__constant__"
].toHashSet()

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerCudaLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = true)
  reg.registerSharedLoweringPasses()
  # Consumes final argument types (post lowerArraysAndViews)
  reg.registerInstantiateGenerics()
  reg.registerMaterializePassByRefArgs()
  reg.registerRenameReservedIdentifiers("cudaReservedIdentifiers", CudaReserved)
