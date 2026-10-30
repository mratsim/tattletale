# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## CUDA lowering: per-target registration for the CUDA storage model.
##
## The shared pipeline contributes the reserved-word rename only, every other
## CUDA-C++ transform stays printer-side (backends/):
##
## - the reserved-word rename registers here, through the shared renaming
##   pass with `CudaReserved` as data: the four execution-space qualifiers
##   the CUDA-C++ emitter writes
## - call-site materialization lives in the frontend: every `var T` call
##   argument lowers to an address-of over the lvalue, and Nim sem rejects
##   a computed value for a `var T` parameter before anything reaches CIRv2
## - by-value array parameters decay to `const T name[]` at the printer:
##   the shared pipeline passes a `ctArray` parameter through unchanged,
##   the C++ parameter form belongs to the printer's parameter table
## - the printer owns everything else: the warp-reduce scan plus template emission,
##   the braced init lists, the coordinate arithmetic,
##   the qualifier set, kernel linkage, the barrier and atomic names.
##   The clang syntax fixtures exercise that template text verbatim

import ../ir/cirv2_lang
import ./pass_datatypes
import ./passes_low_reserved

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerCudaLoweringPasses*(reg: var PassRegistry) =
  ## Registers the CUDA per-target lowering stage, the reserved-word rename
  ## with `CudaReserved` as data.
  ##
  ## Registration contract: the rename sits after the shared lowering passes
  ## and before printing, the position every target uses.
  reg.registerRenameReservedIdentifiers("cudaReservedIdentifiers", CudaReserved)
