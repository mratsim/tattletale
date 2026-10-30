# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Text backend dispatch: one entry point for the six text printers.
##
## - `codegen` prints `ctx.module` as the source text that a static `Backend` names
## - each `metal:`-family macro passes its backend as a compile-time constant,
##   so the compiler folds the dispatch to a direct call
## - `codegenSpirvShader` is separate: it returns binary SPIR-V words,
##   a binary artifact that no text printer produces

import ./cuda_lang, ./wgsl_lang, ./opencl_lang, ./glsl_lang, ./msl_lang, ./hlsl_lang,
       ./spirv_shader_lang
import ../ir/ast_datatypes
from ./hlsl_lang import HlslFlavor, fvD3D12

type Backend* = enum
  ## Names the text backend a `codegen` call prints, one value per `*_lang` printer.
  backendCuda, backendWebGpu, backendOpenCL, backendGLSL, backendMetal, backendHlsl

proc codegen*(ctx: var GpuContext, B: static Backend, flavor: static HlslFlavor = fvD3D12): string =
  ## Prints the module in `ctx` as the source text that `B` names.
  ##
  ## - the `metal:`-family macros call this proc after the pass registry
  ## - `ctx.module` is never mutated
  ## - `flavor` selects the HLSL resource-binding flavor, `backendHlsl` only:
  ##   any other backend refuses a non-default flavor
  when B == backendHlsl:
    hlsl_lang.codegen(ctx, flavor)
  else:
    when flavor != fvD3D12:
      {.error: "codegen: `flavor` applies only to backendHlsl".}
    case B
    of backendCuda: cuda_lang.codegen(ctx)
    of backendWebGpu: wgsl_lang.codegen(ctx)
    of backendOpenCL: opencl_lang.codegen(ctx)
    of backendGLSL: glsl_lang.codegen(ctx)
    of backendMetal: msl_lang.codegen(ctx)
    of backendHlsl: hlsl_lang.codegen(ctx, flavor)

proc codegenSpirvShader*(ctx: var GpuContext): seq[uint32] =
  spirv_shader_lang.codegen(ctx)
