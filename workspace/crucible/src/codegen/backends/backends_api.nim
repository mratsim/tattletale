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

proc codegenSource*(m: CirModule, B: static CodegenKind): string =
  case B
  of cgCuda: lang_cuda.codegen(m)
  of cgWebGPU: lang_wgsl.codegen(m)
  of cgOpenCL: lang_opencl.codegen(m)
  of cgGlsl: lang_glsl.codegen(m)
  of cgMetal: lang_msl.codegen(m)
  of cgHlsl: lang_hlsl.codegen(m)
  of cgSpirv:
    raiseAssert("codegenSource: cgSpirv is SPIR-V words through codegenIR, not text")
  of cgCpu:
    raiseAssert("codegenSource: cgCpu does not codegen a module, it lowers through launchCpuFn")

proc codegenSource*(m: CirModule, B: static CodegenKind, flavor: static HlslFlavor): string =
  when B != cgHlsl:
    raiseAssert("codegenSource: the flavor overload prints cgHlsl only, got " & $B)
  lang_hlsl.codegen(m, flavor)

proc codegenIR*(m: CirModule, B: static CodegenKind): seq[uint32] =
  case B
  of cgSpirv: lang_spirv_shader.codegen(m)
  else:
    raiseAssert("codegenIR: cgSpirv is the only IR artifact, " & $B & " prints through codegenSource")
