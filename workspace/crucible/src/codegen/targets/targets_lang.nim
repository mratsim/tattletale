# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./cuda_lang, ./wgsl_lang, ./opencl_lang, ./vulkan_lang, ./metal_lang
import ../ir/gpu_types

proc codegenCuda*(ctx: var GpuContext, ast: GpuAst, kernel: string = ""): string =
  cuda_lang.preprocess(ctx, ast, kernel)
  result = cuda_lang.codegen(ctx)

proc codegenWebGpu*(ctx: var GpuContext, ast: GpuAst, kernel: string = ""): string =
  wgsl_lang.preprocess(ctx, ast, kernel)
  result = wgsl_lang.codegen(ctx)

proc codegenOpenCL*(ctx: var GpuContext, ast: GpuAst, kernel: string = ""): string =
  opencl_lang.preprocess(ctx, ast, kernel)
  result = opencl_lang.codegen(ctx)

proc codegenVulkan*(ctx: var GpuContext, ast: GpuAst, kernel: string = ""): string =
  vulkan_lang.preprocess(ctx, ast, kernel)
  result = vulkan_lang.codegen(ctx)

proc codegenMetal*(ctx: var GpuContext, ast: GpuAst, kernel: string = ""): string =
  ## Lowers `ast` to MSL. Runs the Metal preprocessing passes, then emits the shader source for `kernel`
  ## (all kernels when empty).
  metal_lang.preprocess(ctx, ast, kernel)
  result = metal_lang.codegen(ctx)
