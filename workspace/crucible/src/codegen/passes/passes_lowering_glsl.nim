# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            GLSL lowering passes
#
# ############################################################
#
# Lower the crucible IR onto GLSL's storage model, registered by
# the Vulkan pipeline only:
# - buffer references cannot be indexed directly, pointer parents
#   get an explicit deref the renderer composes as `.x`
# - kVar params bind as `inout`, retyped to their pointee

import ../ir/gpu_ast_datatypes
import ./pass_datatypes
import ./pass_registry
import ./passes_lowering
import ./passes_glsl

proc registerGlslLoweringPasses*(reg: var PassRegistry) =
  # GLSL buffer references cannot be indexed directly, the deref must be
  # explicit in the IR for the renderer to compose `.x`.
  reg.register(
    name = "insertPointerDeref",
    kind = pkTransform,
    phase = phaseMain,
    description = "Wraps pointer-typed index and field parents in explicit gpuDeref",
    dependsOn = @[
      "rewriteIndexDeref"],
    run = insertPointerDeref
  )
  reg.register(
    name = "qualifyVarParams",
    kind = pkTransform,
    phase = phaseMain,
    description = "Retypes kVar params to their pointee, unwraps their body derefs and call-site gpuAddr args",
    dependsOn = @[],
    run = qualifyVarParams
  )
