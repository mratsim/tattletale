# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../ir/gpu_types
import ./pass_datatypes
import ./pass_registry
import ./passes_validations
import ./passes_optimizations

proc registerCudaPasses*(reg: var PassRegistry) =
  ## Register CUDA-specific passes: byref-arg materialization, keyword rejection.
  reg.register("materializePassByRefArgs", pkTransform, phaseMain,
    "Wraps non-lvalue passByRef args in gpuMaterialize nodes",
    dependsOn = @["ensureBlock"],
    run = materializePassByRefArgs
  )
  reg.register("rejectCUDAKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved CUDA keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["__global__", "__device__", "__shared__", "__constant__"], "CUDA")
  )
