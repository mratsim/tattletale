## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan IR legalization passes, registered from the `vulkan:` codegen path.
##
## Four passes lower GLSL-illegal IR shapes to legal Vulkan GLSL: device-fn
## `var T` params, struct values carrying pointer fields, device-fn `ptr`
## params, and fp16-subgroup lane assumptions. All are guarded on `crucibleCompileTarget == ctVulkan`
## so the other backends never see them.

import ../ir/gpu_types
import ../builtins/builtins_compilermagic
import ./pass_datatypes
import ./passes_legalization_vulkan_ptr_in_struct
import ./passes_legalization_vulkan_subgroup_guard

# ═════════════════════════════════════════════════════════════════════════
#  Registration
# ═════════════════════════════════════════════════════════════════════════

proc registerLegalizationVulkanPasses*(reg: var PassRegistry) =
  ## Registers the Vulkan-only legalizations. Called from the `vulkan:`
  ## codegen path after registerCommonPasses, so the blit/constexpr
  ## normalization is already in place. Runs phaseMain (transform) and is
  ## guarded on crucibleCompileTarget == ctVulkan so the other backends are
  ## untouched.
  reg.register("vulkanVarParamsToValue", pkTransform, phaseMain,
    "Device-fn var params → value params (+return by value; array-typed var-param fns inlined)",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      if crucibleCompileTarget == ctVulkan:
        convertVarParams(ctx)
  )
  reg.register("vulkanFlattenStructPtrValues", pkTransform, phaseMain,
    "Flatten struct-with-ptr-field values into leaf scalars + SSBO ptr expressions",
    dependsOn = @["vulkanVarParamsToValue"],
    run = proc(ctx: var GpuContext): void =
      if crucibleCompileTarget == ctVulkan:
        flattenStructPtrValues(ctx)
  )
  reg.register("vulkanBindDeviceFnPtrParams", pkTransform, phaseMain,
    "Per-call-site device-fn ptr-param binding with ident→expression substitution",
    dependsOn = @["vulkanFlattenStructPtrValues"],
    run = proc(ctx: var GpuContext): void =
      if crucibleCompileTarget == ctVulkan:
        bindDeviceFnPtrParams(ctx)
  )
  reg.register("vulkanSubgroupGuard32", pkTransform, phaseMain,
    "Fail-loudly gl_SubgroupSize<32 guard + gl_SubgroupInvocationID lane id on the fp16-subgroup shuffle path (GPU-B-001)",
    dependsOn = @["vulkanBindDeviceFnPtrParams"],
    run = proc(ctx: var GpuContext): void =
      if crucibleCompileTarget == ctVulkan:
        subgroupGuard32(ctx)
  )
