# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./cuda_lang, ./wgsl_lang, ./opencl_lang, ./glsl_lang, ./msl_lang
import ../ir/ast_datatypes

proc codegenCuda*(ctx: var GpuContext): string =
  cuda_lang.codegen(ctx)

proc codegenWebGpu*(ctx: var GpuContext): string =
  wgsl_lang.codegen(ctx)

proc codegenOpenCL*(ctx: var GpuContext): string =
  opencl_lang.codegen(ctx)

proc codegenGLSL*(ctx: var GpuContext): string =
  glsl_lang.codegen(ctx)

proc codegenMetal*(ctx: var GpuContext): string =
  msl_lang.codegen(ctx)
