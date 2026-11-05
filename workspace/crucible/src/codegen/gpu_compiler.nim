# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
import std/macros

import ../builtins/builtins
export builtins
import ../builtins/builtins_compilermagic
import ../crucible_codegen
export builtins_compilermagic

import ./frontend/nim_to_crucible
import ./passes/pass_registry
import ./passes/passes_normalizations
import ./passes/passes_legalization
import ./passes/passes_lowering
import ./passes/passes_lowering_hlsl
import ./passes/passes_lowering_glsl
import ./passes/passes_lowering_wgsl
import ./passes/passes_lowering_msl
import ./passes/passes_lowering_opencl
import ./passes/passes_lowering_cfamily
import ./passes/analysis_hlsl
import ./passes/analysis_module_requirements
import ./passes/passes_validations
import ./passes/passes_low_reserved
import ./passes/passes_low_pointers
import ./passes/passes_low_array_assign
import ./passes/passes_low_address_spaces
import ./backends/backends_api
import ./ir/cirv2_pretty

from ./backends/lang_hlsl import HlslFlavor, fvD3D12, fvVulkan

# ############################################################
#
#            GPU compiler driver
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Per-target pass pipelines
# ═══════════════════════════════════════════════════════════════════════

proc registerFrontendPasses(reg: var PassRegistry) =
  registerRewriteStmtsExprs(reg)
  registerValueNormalizations(reg)
  registerReshapingPasses(reg)

macro codegenMetal(body: typed): string =
  ## Compiles a `metal:` block body into Metal Shading Language (MSL) code:
  ## CIRv2 lowering, the pass pipeline, then the MSL printer.
  ## A refusal in the chain is a compile-time error naming the construct.
  var reg = PassRegistry.new()
  registerFrontendPasses(reg)
  registerSharedLoweringPasses(reg)
  # Rename position: before every hoist and explosion. The rename walks types,
  # procs and proc bodies, never `m.globals`. A declaration hoisted earlier
  # leaves the rename's reach unrenamed.
  reg.registerRenameReservedIdentifiers("metalReservedIdentifiers", MetalReserved)
  registerDecomposeArrayAssign(reg)
  registerExplodeStructWithPtrFields(reg)
  registerSmemInitRewrite(reg)
  registerMslHoistProgramScopeConstants(reg)
  registerMslHoistDeviceFnSmem(reg)
  registerMslRewriteSmemLocals(reg)
  registerMslThreadCoordinateBuiltins(reg)
  registerInstantiateAddressSpaceGenerics(reg)
  registerValidationPostPasses(reg)
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  result = newLit(codegen(m, cgMetal))

macro metal*(body: untyped): string =
  ## Converts the body of this macro into Metal Shading Language (MSL) code.
  # untyped -> typed macro delegation: the outer macro assigns crucibleCompileTarget
  # before the typed macro runs, so the body resolves with the target already set
  crucibleCompileTarget = ctMetal
  result = newCall(bindSym"codegenMetal", body)

macro codegenCuda(body: typed): string =
  ## Compiles a `cuda:` block body into CUDA C++ source: CIRv2 lowering,
  ## the pass pipeline, then the CUDA printer.
  ## Refusals in the chain are compile-time
  ## errors naming the construct.
  var reg = PassRegistry.new()
  registerFrontendPasses(reg)
  registerSharedLoweringPasses(reg)
  registerDecomposeArrayAssign(reg)
  # Rename position: after decomposeArrayAssign, incidental placement.
  # Decompose creates only `__builtin_memcpy` calls, no generated name
  # collides with the execution-space qualifiers of CudaReserved.
  reg.registerRenameReservedIdentifiers("cudaReservedIdentifiers", CudaReserved)
  registerSmemInitRewrite(reg)
  registerInstantiateAddressSpaceGenerics(reg)
  registerCudaFeatureScan(reg)
  registerValidationPostPasses(reg)
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  result = newLit(codegen(m, cgCuda))

macro cuda*(body: untyped): string =
  ## Converts the body of this macro into CUDA code.
  crucibleCompileTarget = ctCuda
  result = newCall(bindSym"codegenCuda", body)

macro codegenOpenCL(body: typed): string =
  ## Compiles an `opencl:` block body into OpenCL C code: CIRv2 lowering,
  ## the pass pipeline, then the OpenCL printer.
  ## Refusals in the chain are compile-time
  ## errors naming the construct.
  var reg = PassRegistry.new()
  registerFrontendPasses(reg)
  registerSharedLoweringPasses(reg)
  reg.registerRenameReservedIdentifiers("openclReservedIdentifiers", OpenclReserved)
  registerDecomposeArrayAssign(reg)
  registerExplodeStructWithPtrFields(reg)
  registerSmemInitRewrite(reg)
  registerOpenclHoistProgramScopeConstants(reg)
  registerOpenclLowerByrefParams(reg)
  registerInstantiateAddressSpaceGenerics(reg)
  registerOpenclFeatureScan(reg)
  registerValidationPostPasses(reg)
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  result = newLit(codegen(m, cgOpenCL))

macro opencl*(body: untyped): string =
  ## Converts the body of this macro into OpenCL C code.
  crucibleCompileTarget = ctOpenCL
  result = newCall(bindSym"codegenOpenCL", body)

macro codegenGLSL(body: typed): string =
  ## Compiles a `glsl:` block body into GLSL compute shader code:
  ## CIRv2 lowering, the pass pipeline, then the GLSL printer.
  ## A refusal anywhere in the chain is a compile-time
  ## error naming the construct.
  var reg = PassRegistry.new()
  registerFrontendPasses(reg)
  registerSharedLoweringPasses(reg)
  reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  registerGlslQualifyVarParams(reg)
  registerSmemInitRewrite(reg)
  registerGlslHoistProgramScopeLocals(reg)
  registerInstantiateAddressSpaceGenerics(reg)
  registerGlslRefBlockPass(reg)
  registerGlslFeatureScan(reg)
  registerValidationPostPasses(reg)
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  result = newLit(codegen(m, cgGlsl))

macro glsl*(body: untyped): string =
  ## Converts the body of this macro into GLSL compute shader code.
  crucibleCompileTarget = ctGlsl
  result = newCall(bindSym"codegenGLSL", body)

macro codegenSpirvShader(body: typed): seq[uint32] =
  ## Compiles a `spirv:` block body into binary SPIR-V words for the Vulkan
  ## environment through CIRv2 lowering, the pass pipeline,
  ## then the SPIR-V encoder.
  ##
  ## Any refusal in the chain is a compile-time error
  ## naming the construct.
  var reg = PassRegistry.new()
  registerFrontendPasses(reg)
  registerSharedLoweringPasses(reg)
  # No rename and no address-space callee instantiation: the encoder consumes typed words.
  registerValidationPostPasses(reg)
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  result = newLit(codegenSpirvShader(m))

macro spirv*(body: untyped): seq[uint32] =
  ## Converts the body of this macro into binary SPIR-V words (Vulkan environment).
  crucibleCompileTarget = ctSpirvShader
  result = newCall(bindSym"codegenSpirvShader", body)

macro codegenWebGpu(body: typed): string =
  ## Compiles a `webgpu:` block body into WebGPU WGSL code: CIRv2 lowering,
  ## the pass pipeline, then the WGSL printer.
  ## A refusal anywhere in the chain is a compile-time
  ## error naming the construct.
  var reg = PassRegistry.new()
  registerFrontendPasses(reg)
  registerSharedLoweringPasses(reg)
  registerFoldMaxMinToBuiltins(reg)
  reg.registerRenameReservedIdentifiers("wgslReservedIdentifiers", WgslReserved, "wgsl_")
  registerExplodeStructWithPtrFields(reg)
  registerWgslExplodePointerLocals(reg)
  registerSmemInitRewrite(reg)
  registerWgslHoistProgramScopeLocals(reg)
  registerInstantiateAddressSpaceGenerics(reg)
  registerWgslInlinePointerParamCalls(reg)
  registerWgslFeatureScan(reg)
  registerValidationPostPasses(reg)
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  result = newLit(codegen(m, cgWebGPU))

macro webgpu*(body: untyped): string =
  ## Converts the body of this macro into WebGPU WGSL code.
  crucibleCompileTarget = ctWebGPU
  result = newCall(bindSym"codegenWebGpu", body)

macro codegenHlsl(body: typed, flavor: static HlslFlavor = fvD3D12): string =
  ## Compiles an `hlsl:` block body into HLSL compute shader code.
  ##
  ## - CIRv2 lowering, the pass pipeline with the HLSL
  ##   per-target lowering passes and the call-graph analysis, then the HLSL
  ##   printer for `flavor`
  ## - a refusal in the chain is a compile-time
  ##   error naming the construct
  var reg = PassRegistry.new()
  registerFrontendPasses(reg)
  registerSharedLoweringPasses(reg)
  registerFoldMaxMinToBuiltins(reg)
  registerHlslMaterializeNumWorkgroups(reg)
  registerHlslExplodePointerLocals(reg)
  # Rename position: after the pointer-local explosion, one named deviation
  # from the shared explode template's documented contract (explode after
  # rename). The rename matches names over declarations and uses.
  # Spliced alias symbols rename together and the emitted names stay consistent.
  reg.registerRenameReservedIdentifiers("hlslReservedIdentifiers", HlslReserved)
  registerHlslHoistProgramScopeConstants(reg)
  registerInstantiateAddressSpaceGenerics(reg)
  registerHlslCallGraphPass(reg)
  registerValidationPostPasses(reg)
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  # a macro's static param is not a compile-time constant in the macro body,
  # so each flavor binds to the `codegen` static param as a literal below
  let printed = case flavor
    of fvD3D12: codegen(m, cgHlsl, fvD3D12)
    of fvVulkan: codegen(m, cgHlsl, fvVulkan)
  result = newLit(printed)

macro hlsl*(flavor: untyped, body: untyped): string =
  ## Converts the body of this macro into HLSL compute shader code.
  ##
  ## - resource binding per `flavor`, D3D12-native by default,
  ##   fvVulkan emits the zero-descriptor flavor for the engine's address route
  ## - `flavor` is untyped so overload resolution never sees the body,
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
    error("unknown flavor `" & name & "`, expected fvD3D12/fvVulkan")
  crucibleCompileTarget = ctHlsl
  result = newCall(bindSym"codegenHlsl", body, newLit(flav))

macro hlsl*(body: untyped): string =
  ## Converts the body of this macro into HLSL compute shader code,
  ## D3D12-native resource binding.
  crucibleCompileTarget = ctHlsl
  result = newCall(bindSym"codegenHlsl", body)
