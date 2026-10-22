# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            OpenCL lowering passes
#
# ############################################################
#
# Project the crucible IR onto OpenCL C's calling convention:
# passByRef structs pass as hidden const pointers and structs
# carrying pointers expand to ptr/scalar leaves, since OpenCL
# params cross by value.

import ./pass_datatypes
import ./pass_registry
import ./passes_optimizations
import ./passes_legalizations

proc registerOpenclLoweringPasses*(reg: var PassRegistry) =
  reg.register("materializePassByRefArgs", pkTransform, phaseMain,
    "Wraps non-lvalue passByRef args in gpuMaterialize nodes",
    dependsOn = @["ensureBlock"],
    run = materializePassByRefArgs
  )
  reg.registerExpandStructKernelParams()
