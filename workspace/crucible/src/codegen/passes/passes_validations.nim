# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Validation stage over CIRv2: the verifier as registry passes.
## Post-validation runs after the transforms, right before printing.

import ../ir/cirv2_lang
import ../ir/cirv2_verify
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerValidationPostPasses*(reg: var PassRegistry) =
  reg.register("verify", pkValidation, phaseMain,
    "Verifies the module, refuses on the first invariant violation",
    run = verifyPass)

proc verifyPass(m: CirModule): string =
  ## Registry adapter over `verifyMessage`.
  verifyMessage(m)

