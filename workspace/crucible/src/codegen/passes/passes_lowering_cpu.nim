# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU lowering registration for the CPU pipeline.
##
##   kernel body → registerCpuLoweringPasses → printed Nim or C text
##
## Registers nothing in v0.
##
## - CIRv2 identifiers are Nim-derived, the reserved-word renaming the six
##   text backends need has no CPU equivalent
## - target specifics live in the Nim and C emitters
## - registration stays in the pipeline for symmetry with the other targets,
##   later CPU passes slot in without reshaping the pipeline
##
## Emitter specifics:
## - coordinate builtins map to launch-site expressions
## - vectors, barriers, shuffles, reductions, atomics refuse there in v0

import ../ir/cirv2_lang
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

proc registerCpuLoweringPasses*(reg: var PassRegistry) =
  ## Registers the CPU lowering passes. Empty in v0, see the module header.
  discard
