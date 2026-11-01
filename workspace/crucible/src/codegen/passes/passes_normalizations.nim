# Tattletale
# Copyright (c) 2026 Mamy Andre-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Normalization stage over CIRv2: canonical reshaping.
## It establishes the forms every later stage assumes.
## Target-independent, every backend registers them identically.

import ./pass_datatypes
import ./passes_norm_compound_assign
import ./passes_norm_expr_forms
import ./passes_norm_unroll

func registerNormalizationPasses*(reg: var PassRegistry) =
  # settleExprForms decides the literal widths, the bool operator domain,
  # constructor field order, before the later stages read them
  reg.registerSettleExprForms()
  # bindCompoundAssignTemps runs first among the reshaping passes:
  # shared-tree detection compares references
  reg.registerCompoundAssignTemps()
  reg.registerMarkExplicitUnrolls()
