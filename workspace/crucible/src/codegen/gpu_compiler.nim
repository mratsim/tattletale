# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU kernel code generator, emits native GPU code: CUDA, OpenCL, GLSL, WebGPU, Metal.
## GPU kernel code generator, emits native GPU code: CUDA, OpenCL, GLSL, WebGPU, Metal.
##
## Public API, one import: `import workspace/crucible`.
## The curated surface lives in `src/crucible_public_api.nim`, the GPU builtin vocabulary
## in `src/crucible_gpu_builtins.nim`, both re-exported here in full.
## Everything not exported there is internal, only compiler tests reach into the internals.
##
## Pipeline:
## - the `metal:`-family macros lower the typed body to CIRv2
## - `buildPipeline` materializes the target's pass pipeline from the table below
## - `runPasses` executes it, `backends_api` dispatches the final module to the `*_lang` printer
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
import ./passes/passes_lowering
import ./passes/passes_lowering_hlsl
import ./passes/passes_lowering_glsl
import ./passes/passes_lowering_wgsl
import ./passes/passes_lowering_msl
import ./passes/passes_lowering_opencl
import ./passes/passes_lowering_cfamily
import ./passes/layout_std430
import ./passes/analysis_hlsl
import ./passes/scan_features
import ./passes/passes_validations
import ./passes/passes_low_reserved
import ./passes/passes_low_pointers
import ./passes/passes_low_array_assign
import ./passes/passes_low_address_spaces
import ./backends/backends_api
import ./ir/cirv2_pretty
from ./backends/hlsl_lang import HlslFlavor, fvD3D12, fvVulkan

# ═══════════════════════════════════════════════════════════════════════
# The pipeline table
# ═══════════════════════════════════════════════════════════════════════

type
  PipelineBind* = proc(reg: var PassRegistry, flavor: HlslFlavor) {.nimcall.}
    ## Registers one pass into `reg`. The flavor parameter reaches the one
    ## flavor-dependent row (`hlslClassifyParams`), every other binder ignores it.

proc attachLegalize(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerLegalizationPasses()

proc attachNormalize(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerNormalizationPasses()

proc attachLowerSpans(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerSharedLoweringPasses()

proc attachRenameMetal(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerRenameReservedIdentifiers("metalReservedIdentifiers", MetalReserved)

proc attachRenameCuda(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerRenameReservedIdentifiers("cudaReservedIdentifiers", CudaReserved)

proc attachRenameOpencl(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerRenameReservedIdentifiers("openclReservedIdentifiers", OpenclReserved)

proc attachRenameGlsl(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)

proc attachRenameWgsl(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerRenameReservedIdentifiers("wgslReservedIdentifiers", WgslReserved, "wgsl_")

proc attachRenameHlsl(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerRenameReservedIdentifiers("hlslReservedIdentifiers", HlslReserved)

proc attachDecompose(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerDecomposeArrayAssign()

proc attachExplodeStructs(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerExplodeStructWithPtrFields()

proc attachSpaceGenerics(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerInstantiateAddressSpaceGenerics()

proc attachMslHoistProgramScopeConstants(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerMslHoistProgramScopeConstants()

proc attachMslHoistDeviceFnSmem(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerMslHoistDeviceFnSmem()

proc attachMslRewriteSmemLocals(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerMslRewriteSmemLocals()

proc attachMslThreadCoordinateBuiltins(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerMslThreadCoordinateBuiltins()

proc attachOpenclHoistConst(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerOpenclHoistProgramScopeConstants()

proc attachOpenclByref(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerOpenclLowerByrefParams()

proc attachSmemInitRewrite(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerSmemInitRewrite()

proc attachGlslQualifyVarParams(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerGlslQualifyVarParams()

proc attachGlslExplodePointerLocals(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerGlslExplodePointerLocals()

proc attachGlslHoistLocals(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerGlslHoistProgramScopeLocals()

proc attachWgslExplodePointerLocals(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerWgslExplodePointerLocals()

proc attachWgslHoistLocals(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerWgslHoistProgramScopeLocals()

proc attachWgslInlinePointerCalls(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerWgslInlinePointerParamCalls()

proc attachHlslMaterializeNumWorkgroups(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerHlslMaterializeNumWorkgroups()

proc attachHlslExplodePointerLocals(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerHlslExplodePointerLocals()

proc attachHlslHoistConst(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerHlslHoistProgramScopeConstants()

proc attachGlslRefBlocks(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerGlslRefBlockPass()

proc attachOpenclScan(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerOpenclFeatureScan()

proc attachCudaScan(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerCudaFeatureScan()

proc attachGlslScan(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerGlslFeatureScan()

proc attachGlslLayout(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerGlslLayoutPass()

proc attachWgslStructRefs(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerWgslStructRefPass()

proc attachWgslScan(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerWgslFeatureScan()

proc attachWgslLayout(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerWgslLayoutPass()

proc attachHlslParamClass(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerHlslParamClassPass(flavor)

proc attachHlslStructRefs(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerHlslStructRefPass()

proc attachHlslCallGraph(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerHlslCallGraphPass()

proc attachValidate(reg: var PassRegistry, flavor: HlslFlavor) =
  reg.registerValidationPostPasses()

const pipelineTable: array[CompileTarget, seq[PipelineBind]] = [
    # Shared legalization, normalization and span lowering open every column.
    # Analysis rows close every column that has them, check-only validation rows
    # run last. Every compile machine-checks the grouping through runPasses,
    # which refuses a transform registered after an analysis.
    ctNone: @[],
    ctCuda: @[
      attachLegalize, attachNormalize, attachLowerSpans,
      # Rename position: after decomposeArrayAssign, incidental placement.
      # Decompose creates only `__builtin_memcpy` calls, no generated name
      # collides with the execution-space qualifiers of CudaReserved.
      attachDecompose, attachRenameCuda, attachSmemInitRewrite,
      attachSpaceGenerics, attachCudaScan,
      attachValidate,
    ],
    ctMetal: @[
      attachLegalize, attachNormalize, attachLowerSpans,
      # Rename position: before every hoist and explosion. The rename walks types,
      # procs and proc bodies, never `m.globals`. A declaration hoisted earlier
      # leaves the rename's reach unrenamed.
      attachRenameMetal,
      attachDecompose, attachExplodeStructs, attachSmemInitRewrite,
      attachMslHoistProgramScopeConstants, attachMslHoistDeviceFnSmem,
      attachMslRewriteSmemLocals, attachMslThreadCoordinateBuiltins,
      attachSpaceGenerics,
      attachValidate,
    ],
    ctOpenCL: @[
      attachLegalize, attachNormalize, attachLowerSpans,
      attachRenameOpencl,
      attachDecompose, attachExplodeStructs, attachSmemInitRewrite,
      attachOpenclHoistConst, attachOpenclByref, attachSpaceGenerics,
      attachOpenclScan,
      attachValidate,
    ],
    ctGlsl: @[
      attachLegalize, attachNormalize, attachLowerSpans,
      attachRenameGlsl,
      attachGlslQualifyVarParams, attachGlslExplodePointerLocals,
      attachSmemInitRewrite, attachGlslHoistLocals, attachSpaceGenerics,
      attachGlslRefBlocks, attachGlslScan, attachGlslLayout,
      attachValidate,
    ],
    ctWebGPU: @[
      attachLegalize, attachNormalize, attachLowerSpans,
      attachRenameWgsl,
      attachExplodeStructs, attachWgslExplodePointerLocals,
      attachSmemInitRewrite, attachWgslHoistLocals, attachSpaceGenerics,
      attachWgslInlinePointerCalls,
      attachWgslStructRefs, attachWgslScan, attachWgslLayout,
      attachValidate,
    ],
    ctHlsl: @[
      attachLegalize, attachNormalize, attachLowerSpans,
      # Rename position: after the pointer-local dissolution, one named deviation
      # from the shared dissolve template's documented contract (dissolve after
      # rename). The rename matches names over declarations and uses.
      # Spliced alias symbols rename together and the emitted names stay consistent.
      attachHlslMaterializeNumWorkgroups, attachHlslExplodePointerLocals,
      attachRenameHlsl, attachHlslHoistConst, attachSpaceGenerics,
      attachHlslParamClass, attachHlslStructRefs, attachHlslCallGraph,
      attachValidate,
    ],
    ctSpirvShader: @[
      attachLegalize, attachNormalize, attachLowerSpans,
      # No rename and no address-space callee instantiation: the encoder consumes typed words.
      attachValidate,
    ],
  ]

proc buildPipeline*(target: CompileTarget, flavor: HlslFlavor = fvD3D12): PassRegistry =
  ## Builds the target's pass pipeline from the table, in table order:
  ## registration order is execution order.
  result = PassRegistry.new()
  for attach in pipelineTable[target]:
    attach(result, flavor)

macro codegenMetal(body: typed): string =
  ## Compiles a `metal:` block body into Metal Shading Language (MSL) code:
  ## CIRv2 lowering, the pass pipeline, then the MSL printer.
  ## A refusal anywhere in the chain is a compile-time
  ## error naming the construct.
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  var reg = buildPipeline(ctMetal)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegen(backendMetal))

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
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  var reg = buildPipeline(ctCuda)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegen(backendCuda))

macro cuda*(body: untyped): string =
  ## Converts the body of this macro into CUDA code.
  crucibleCompileTarget = ctCuda
  result = newCall(bindSym"codegenCuda", body)

macro codegenOpenCL(body: typed): string =
  ## Compiles an `opencl:` block body into OpenCL C code: CIRv2 lowering,
  ## the pass pipeline, then the OpenCL printer.
  ## Refusals in the chain are compile-time
  ## errors naming the construct.
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  var reg = buildPipeline(ctOpenCL)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegen(backendOpenCL))

macro opencl*(body: untyped): string =
  ## Converts the body of this macro into OpenCL C code.
  crucibleCompileTarget = ctOpenCL
  result = newCall(bindSym"codegenOpenCL", body)

macro codegenGLSL(body: typed): string =
  ## Compiles a `glsl:` block body into GLSL compute shader code:
  ## CIRv2 lowering, the pass pipeline, then the GLSL printer.
  ## A refusal anywhere in the chain is a compile-time
  ## error naming the construct.
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  var reg = buildPipeline(ctGlsl)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegen(backendGLSL))

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
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  var reg = buildPipeline(ctSpirvShader)
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
  ## the pass pipeline, then the WGSL printer.
  ## A refusal anywhere in the chain is a compile-time
  ## error naming the construct.
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  var reg = buildPipeline(ctWebGPU)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  result = newLit(ctx.codegen(backendWebGpu))

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
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  var reg = buildPipeline(ctHlsl, flavor)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var ctx = GpuContext(module: m)
  # a macro's static param is not a compile-time constant in the macro body,
  # so each flavor binds to the `codegen` static param as a literal below
  let printed = case flavor
    of fvD3D12: ctx.codegen(backendHlsl, fvD3D12)
    of fvVulkan: ctx.codegen(backendHlsl, fvVulkan)
  result = newLit(printed)

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
