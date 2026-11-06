# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

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
