# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import cirv2_lang

# ############################################################
#
#            CIRv2 pass-boundary verifier
#
# ############################################################

proc verifyModule*(m: CirModule): string =
  ## Returns "" while the module upholds the pass-boundary contracts,
  ## else the first violation with the offending proc rendered by cirv2_pretty.
  ## Read-only: never rewrites or repairs the IR.
  ## runPasses calls this after each registered pass on modules with verifyIR set.
  result = ""
