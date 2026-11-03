# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU lowering registration for the CPU pipeline.
##
##   kernel body → registerCpuLoweringPasses → printed C text
##
## Two passes in v0, both reusing shared machinery:
##
## - the C reserved-word rename (`CReserved`), CIRv2 identifiers are
##   Nim-derived and a Nim identifier may be a C keyword
## - the array-assign decomposition, C assigns arrays only element-wise
##
## Target specifics live in the C emitters.
##
## Emitter specifics:
## - coordinate builtins map to launch-site expressions
## - vectors, barriers, shuffles, reductions, atomics refuse there in v0

import std/macros
import ./pass_datatypes
import ./passes_low_reserved
import ./passes_low_array_assign

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

proc registerCpuLoweringPasses*(reg: var PassRegistry) =
  reg.registerRenameReservedIdentifiers("renameCpuReservedIdentifiers", CReserved)
  reg.registerDecomposeArrayAssign()
