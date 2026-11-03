# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./lang_cuda, ./lang_wgsl, ./lang_opencl, ./lang_glsl, ./lang_msl, ./lang_hlsl,
       ./lang_spirv_shader
import ../ir/cirv2_lang
import ../../crucible_codegen

from ./lang_hlsl import HlslFlavor, fvD3D12

# ############################################################
#
#            Backend API
#
# ############################################################

proc codegen*(m: CirModule, B: static CodegenKind, flavor: static HlslFlavor = fvD3D12): string =
  ## Prints `m` as the source text that `B` names.
  ##
  ## - the `metal:`-family macros call this proc after the pass registry
  ## - `m` is never mutated
  ## - `flavor` selects the HLSL resource-binding flavor, `cgHlsl` only:
  ##   any other backend refuses a non-default flavor
  when B == cgSpirv:
    {.error: "codegen: cgSpirv prints through codegenSpirvShader".}
  when B == cgCpu:
    {.error: "codegen: cgCpu prints through the cpu launch emitters".}
  when B == cgHlsl:
    lang_hlsl.codegen(m, flavor)
  else:
    when flavor != fvD3D12:
      {.error: "codegen: `flavor` applies only to cgHlsl".}
    case B
    of cgCuda: lang_cuda.codegen(m)
    of cgWebGPU: lang_wgsl.codegen(m)
    of cgOpenCL: lang_opencl.codegen(m)
    of cgGlsl: lang_glsl.codegen(m)
    of cgMetal: lang_msl.codegen(m)
    of cgHlsl: lang_hlsl.codegen(m, flavor)
    of cgSpirv: ""
    of cgCpu: ""

proc codegenSpirvShader*(m: CirModule): seq[uint32] =
  lang_spirv_shader.codegen(m)
