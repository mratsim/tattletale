## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, sets, tables]
import ../ir/gpu_types
import ./pass_datatypes

export pass_datatypes

# ═══════════════════════════════════════════════════════════════════
# Default pipeline factory
# ═══════════════════════════════════════════════════════════════════

proc new*(_: typedesc[PassRegistry]): PassRegistry =
  ## Creates an empty registry. Backend macros call
  ## registerLegalizationPasses / registerValidationPasses
  ## and optionally registerOptimizationPasses to add passes.
  PassRegistry(passes: @[], donePasses: initHashSet[string]())
