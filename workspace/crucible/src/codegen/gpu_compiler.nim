# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU kernel code generator, emits native GPU code: CUDA, OpenCL, GLSL, WebGPU, Metal.
##
## Public API, one import: `import workspace/crucible`.
## The curated surface lives in `src/crucible_public_api.nim`, the GPU builtin vocabulary
## in `src/crucible_gpu_builtins.nim`, both re-exported here in full.
## Everything not exported there is internal, only compiler tests reach into the internals.
##
## Pipeline:
## - the `metal:`-family macros lower the typed body to CIRv2
## - the pass registry runs in stage order: legalize -> normalize -> optimize -> lower -> validate
## - `backends_api` dispatches the final module to the `*_lang` printer
##
## Modules: `src/codegen/frontend/nim_to_crucible`, the pass stages under
## `src/codegen/passes/`, the printers under `src/codegen/backends/*_lang`.

import std/macros

import ../builtins/builtins
export builtins
import ../builtins/builtins_compilermagic
export builtins_compilermagic

import ./ir/ast_datatypes
import ./frontend/nim_to_crucible
import ./passes/pass_registry
import ./passes/passes_legalizations
import ./passes/passes_normalizations
import ./passes/passes_optimizations
import ./passes/passes_lowering
import ./passes/passes_lowering_hlsl
import ./passes/passes_lowering_glsl
import ./passes/passes_lowering_wgsl
import ./passes/passes_lowering_msl
import ./passes/passes_lowering_opencl
import ./passes/layout_std430
import ./passes/callgraph_analysis
import ./passes/scan_features
import ./passes/passes_validations
import ./passes/passes_low_reserved
import ./backends/backends_api
import ./ir/cirv2_pretty
import ./passes/passes_lowering_spirv
from ./backends/hlsl_lang import HlslFlavor, fvD3D12, fvVulkan

macro codegenMetal(body: typed): string =
  ## Compiles a `metal:` block body into Metal Shading Language (MSL) code:
  ## CIRv2 lowering, the pass registry in stage order, then the MSL printer.
  ## A refusal or a verifier violation anywhere in the chain is a compile-time
  ## error naming the construct.
  let (m, lerr) = lowerCirv2Unverified(body)
  if lerr.len > 0:
    error(lerr)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("metalReservedIdentifiers", MetalReserved)
  reg.registerMslLoweringPasses()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegenMetal())

macro metal*(body: untyped): string =
  ## Converts the body of this macro into Metal Shading Language (MSL) code.
  # untyped -> typed macro delegation: the outer macro assigns crucibleCompileTarget
  # before the typed macro runs, so the body resolves with the target already set
  crucibleCompileTarget = ctMetal
  result = newCall(bindSym"codegenMetal", body)

macro codegenCuda(body: typed): string =
  ## Compiles a `cuda:` block body into CUDA C++ source: CIRv2 lowering,
  ## the pass registry in stage order, then the CUDA printer.
  ## Refusals and verifier violations in the chain are compile-time
  ## errors naming the construct.
  let (m, lerr) = lowerCirv2Unverified(body)
  if lerr.len > 0:
    error(lerr)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  # feature scan after the last transform: analysis results are valid only
  # if no transform runs after them (the registry never invalidates)
  reg.registerCudaFeatureScan()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegenCuda())

macro cuda*(body: untyped): string =
  ## Converts the body of this macro into CUDA code.
  crucibleCompileTarget = ctCuda
  result = newCall(bindSym"codegenCuda", body)

macro codegenOpenCL(body: typed): string =
  ## Compiles an `opencl:` block body into OpenCL C code: CIRv2 lowering,
  ## the pass registry in stage order, then the OpenCL printer.
  ## Refusals and verifier violations in the chain are compile-time
  ## errors naming the construct.
  let (m, lerr) = lowerCirv2Unverified(body)
  if lerr.len > 0:
    error(lerr)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("openclReservedIdentifiers", OpenclReserved)
  reg.registerOpenclLoweringPasses()
  # feature scan after the last transform: analysis results are valid only
  # if no transform runs after them (the registry never invalidates)
  reg.registerOpenclFeatureScan()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegenOpenCL())

macro opencl*(body: untyped): string =
  ## Converts the body of this macro into OpenCL C code.
  crucibleCompileTarget = ctOpenCL
  result = newCall(bindSym"codegenOpenCL", body)

macro codegenGLSL(body: typed): string =
  ## Compiles a `glsl:` block body into GLSL compute shader code:
  ## CIRv2 lowering, the pass registry in stage order, then the GLSL printer.
  ## A refusal or a verifier violation anywhere in the chain is a compile-time
  ## error naming the construct.
  let (m, lerr) = lowerCirv2Unverified(body)
  if lerr.len > 0:
    error(lerr)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  # registerRenameReservedIdentifiers runs first: the GLSL lowering passes
  # register after the rename, so the relocated program-scope declarations
  # reference the renamed names that their body text already uses
  reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  reg.registerGlslLoweringPasses()
  # analysis passes after the last transform: analysis results are valid
  # only if no transform runs after them (the registry never invalidates)
  reg.registerGlslFeatureScan()
  reg.registerGlslLayoutPass()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegenGLSL())

macro glsl*(body: untyped): string =
  ## Converts the body of this macro into GLSL compute shader code.
  crucibleCompileTarget = ctGlsl
  result = newCall(bindSym"codegenGLSL", body)

macro codegenSpirvShader(body: typed): seq[uint32] =
  ## Compiles a `spirv:` block body into binary SPIR-V words for the Vulkan
  ## environment through CIRv2 lowering, the pass registry in stage order,
  ## then the SPIR-V encoder.
  ##
  ## Any refusal or verifier violation in the chain is a compile-time error
  ## naming the construct.
  let (m, lerr) = lowerCirv2Unverified(body)
  if lerr.len > 0:
    error(lerr)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerSpirvLoweringPasses()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegenSpirvShader())

macro spirv*(body: untyped): seq[uint32] =
  ## Converts the body of this macro into binary SPIR-V words (Vulkan environment).
  crucibleCompileTarget = ctSpirvShader
  result = newCall(bindSym"codegenSpirvShader", body)

macro codegenWebGpu(body: typed): string =
  ## Compiles a `webgpu:` block body into WebGPU WGSL code: CIRv2 lowering,
  ## the pass registry in stage order, then the WGSL printer.
  ## A refusal or a verifier violation anywhere in the chain is a compile-time
  ## error naming the construct.
  let (m, lerr) = lowerCirv2Unverified(body)
  if lerr.len > 0:
    error(lerr)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  # WGSL reserves the double-underscore identifier head, every `__`-prefixed name gains a `wgsl_` prefix
  reg.registerRenameReservedIdentifiers("wgslReservedIdentifiers", WgslReserved, "wgsl_")
  reg.registerWgslLoweringPasses()
  # analysis passes after the last transform: analysis results are valid
  # only if no transform runs after them (the registry never invalidates)
  reg.registerWgslFeatureScan()
  reg.registerWgslLayoutPass()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegenWebGpu())

macro webgpu*(body: untyped): string =
  ## Converts the body of this macro into WebGPU WGSL code.
  crucibleCompileTarget = ctWebGPU
  result = newCall(bindSym"codegenWebGpu", body)

macro codegenHlsl(body: typed, flavor: static HlslFlavor = fvD3D12): string =
  ## Compiles an `hlsl:` block body into HLSL compute shader code.
  ##
  ## - CIRv2 lowering, the pass registry in stage order with the HLSL
  ##   per-target lowering passes and the call-graph analysis, then the HLSL
  ##   printer for `flavor`
  ## - a refusal or a verifier violation in the chain is a compile-time
  ##   error naming the construct
  let (m, lerr) = lowerCirv2Unverified(body)
  if lerr.len > 0:
    error(lerr)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerHlslLoweringPasses()
  # analysis passes after the last transform: analysis results are valid
  # only if no transform runs after them (the registry never invalidates)
  reg.registerHlslCallGraphPass()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegenHlsl(flavor))

macro hlsl*(flavor: untyped, body: untyped): string =
  ## Converts the body of this macro into HLSL compute shader code.
  ##
  ## - resource binding per `flavor`, D3D12-native by default,
  ##   fvVulkan emits the zero-descriptor flavor for the engine's address route
  ## - `flavor` stays untyped so overload resolution never sees the body,
  ##   a typed parameter would evaluate it outside any DSL block
  var name = ""
  if flavor.kind in {nnkIdent, nnkSym}:
    name = flavor.strVal
  elif flavor.kind == nnkDotExpr:
    name = flavor[1].strVal
  var flav = fvD3D12
  case name
  of "fvD3D12":
    discard
  of "fvVulkan":
    flav = fvVulkan
  else:
    error("hlsl: unknown flavor `" & name & "`, expected fvD3D12 or fvVulkan")
  crucibleCompileTarget = ctHlsl
  result = newCall(bindSym"codegenHlsl", body, newLit(flav))

macro hlsl*(body: untyped): string =
  ## Converts the body of this macro into HLSL compute shader code,
  ## D3D12-native resource binding.
  crucibleCompileTarget = ctHlsl
  result = newCall(bindSym"codegenHlsl", body)
