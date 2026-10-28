# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Legalization stage over CIRv2: conforms the IR to the model every later
## stage assumes. Target-independent, every backend registers them identically.

import ./pass_datatypes
import ./passes_norm_aliases
import ./passes_legal_inline

func registerLegalizationPasses*(reg: var PassRegistry) =
  # alias canonicalization runs first: the inline splice's
  # kernel-only-reads check keys on canonical names, the `dependsOn` entry
  # records that dependency
  reg.registerAliasCanonicalization()
  reg.registerInlineSplice()
