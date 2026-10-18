# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../ir/gpu_types
import ./pass_datatypes
import ./pass_registry
import ./passes_legalizations
import ./passes_validations
import ./passes_preprocessing
import ./passes_lowering # insertPointerDeref

proc registerVulkanPasses*(reg: var PassRegistry) =
  ## Register Vulkan-specific passes: GLSL keyword rejection, pointer deref
  ## insertion for buffer-reference rendering.
  reg.register("rejectVulkanKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved GLSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["extern", "interface", "buffer"], "GLSL")
  )
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
