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
import ./passes/pass_datatypes
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
  registerResolveBoolDomain(reg)
  registerValueNormalizations(reg)
  registerReshapingPasses(reg)

proc registerGpuPasses*(reg: var PassRegistry, B: CodegenKind) {.compileTime.} =
  ## Registers `B`'s per-target passes on top of the shared passes.
  registerFrontendPasses(reg)
  registerSharedLoweringPasses(reg)
  case B
  of cgMetal:
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
  of cgCuda:
    reg.registerRenameReservedIdentifiers("cudaReservedIdentifiers", CudaReserved)
    registerDecomposeArrayAssign(reg)
    registerSmemInitRewrite(reg)
    registerInstantiateAddressSpaceGenerics(reg)
    registerCudaFeatureScan(reg)
    registerValidationPostPasses(reg)
  of cgOpenCL:
    reg.registerRenameReservedIdentifiers("openclReservedIdentifiers", OpenclReserved)
    registerDecomposeArrayAssign(reg)
    registerExplodeStructWithPtrFields(reg)
    registerSmemInitRewrite(reg)
    registerOpenclHoistProgramScopeConstants(reg)
    registerOpenclLowerByrefParams(reg)
    registerInstantiateAddressSpaceGenerics(reg)
    registerOpenclFeatureScan(reg)
    registerValidationPostPasses(reg)
  of cgGlsl:
    reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
    registerGlslQualifyVarParams(reg)
    registerSmemInitRewrite(reg)
    registerGlslHoistProgramScopeLocals(reg)
    registerInstantiateAddressSpaceGenerics(reg)
    registerGlslRefBlockPass(reg)
    registerGlslFeatureScan(reg)
    registerValidationPostPasses(reg)
  of cgSpirv:
    registerValidationPostPasses(reg)
  of cgWebGPU:
    registerFoldMaxMinToBuiltins(reg)
    reg.registerRenameReservedIdentifiers("wgslReservedIdentifiers", WgslReserved, "wgsl_")
    registerExplodeStructWithPtrFields(reg)
    registerWgslInlinePointerParamCalls(reg)
    registerWgslExplodePointerLocals(reg)
    registerSmemInitRewrite(reg)
    registerWgslHoistProgramScopeLocals(reg)
    registerInstantiateAddressSpaceGenerics(reg)
    registerWgslFeatureScan(reg)
    registerValidationPostPasses(reg)
  of cgHlsl:
    registerFoldMaxMinToBuiltins(reg)
    registerHlslMaterializeNumWorkgroups(reg)
    registerHlslExplodePointerLocals(reg)
    reg.registerRenameReservedIdentifiers("hlslReservedIdentifiers", HlslReserved)
    registerHlslHoistProgramScopeConstants(reg)
    registerInstantiateAddressSpaceGenerics(reg)
    registerHlslCallGraphPass(reg)
    registerValidationPostPasses(reg)
  else:
    raiseAssert("registerGpuPasses: no pass pipeline for " & $B)

macro metalImpl(body: typed): string =
  var reg = PassRegistry()
  reg.registerGpuPasses(cgMetal)
  let m = lowerCirv2(body)
  reg.runPasses(m)
  result = newLit(codegenSource(m, cgMetal))

macro metal*(body: untyped): string =
  crucibleCompileTarget = ctMetal
  result = newCall(bindSym"metalImpl", body)

macro cudaImpl(body: typed): string =
  var reg = PassRegistry()
  reg.registerGpuPasses(cgCuda)
  let m = lowerCirv2(body)
  reg.runPasses(m)
  result = newLit(codegenSource(m, cgCuda))

macro cuda*(body: untyped): string =
  crucibleCompileTarget = ctCuda
  result = newCall(bindSym"cudaImpl", body)

macro openclImpl(body: typed): string =
  var reg = PassRegistry()
  reg.registerGpuPasses(cgOpenCL)
  let m = lowerCirv2(body)
  reg.runPasses(m)
  result = newLit(codegenSource(m, cgOpenCL))

macro opencl*(body: untyped): string =
  crucibleCompileTarget = ctOpenCL
  result = newCall(bindSym"openclImpl", body)

macro glslImpl(body: typed): string =
  var reg = PassRegistry()
  reg.registerGpuPasses(cgGlsl)
  let m = lowerCirv2(body)
  reg.runPasses(m)
  result = newLit(codegenSource(m, cgGlsl))

macro glsl*(body: untyped): string =
  crucibleCompileTarget = ctGlsl
  result = newCall(bindSym"glslImpl", body)

macro spirvImpl(body: typed): seq[uint32] =
  var reg = PassRegistry()
  reg.registerGpuPasses(cgSpirv)
  let m = lowerCirv2(body)
  reg.runPasses(m)
  result = newLit(codegenIR(m, cgSpirv))

macro spirv*(body: untyped): seq[uint32] =
  crucibleCompileTarget = ctSpirvShader
  result = newCall(bindSym"spirvImpl", body)

macro webgpuImpl(body: typed): string =
  var reg = PassRegistry()
  reg.registerGpuPasses(cgWebGPU)
  let m = lowerCirv2(body)
  reg.runPasses(m)
  result = newLit(codegenSource(m, cgWebGPU))

macro webgpu*(body: untyped): string =
  crucibleCompileTarget = ctWebGPU
  result = newCall(bindSym"webgpuImpl", body)

macro hlslImpl(body: typed, flavor: static HlslFlavor = fvD3D12): string =
  var reg = PassRegistry()
  reg.registerGpuPasses(cgHlsl)
  let m = lowerCirv2(body)
  reg.runPasses(m)
  let printed = case flavor
    of fvD3D12: codegenSource(m, cgHlsl)
    of fvVulkan: codegenSource(m, cgHlsl, fvVulkan)
  result = newLit(printed)

macro hlsl*(flavor: untyped, body: untyped): string =
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
  result = newCall(bindSym"hlslImpl", body, newLit(flav))

macro hlsl*(body: untyped): string =
  crucibleCompileTarget = ctHlsl
  result = newCall(bindSym"hlslImpl", body)
