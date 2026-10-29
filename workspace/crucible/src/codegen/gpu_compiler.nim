# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, sequtils, tables]

import ./ir/ast_datatypes
import ./ir/ast_type_constructors
import ./backends/backends_api
import ./passes/passes_lowering
import ./passes/passes_low_pointers
import ./passes/passes_low_views
import ./passes/passes_low_array_assign
import ./passes/passes_legal_if_expr
import ./passes/passes_legal_block_expr
import ./passes/passes_lowering_cuda
import ./passes/passes_lowering_wgsl
import ./passes/passes_lowering_opencl
import ./passes/passes_lowering_glsl
import ./passes/passes_lowering_spirv
import ./passes/passes_validations_glsl
import ./passes/passes_lowering_hlsl
from ./backends/hlsl_lang import HlslFlavor, fvD3D12
import ./passes/passes_lowering_msl
import ./frontend/nim_to_crucible
import ./backends/lang_utils
import workspace/crucible/vendor/wgpu
import ./passes/pass_datatypes
import ./passes/pass_registry
import ./passes/passes_legalizations
import ./passes/passes_validations
import ./passes/passes_optimizations
import ./passes/passes_lowering
import ./passes/passes_normalizations

import ../builtins/builtins # all the builtins for the backend to make the Nim compiler happy
export builtins
import ../builtins/builtins_compilermagic
export builtins_compilermagic

proc farmTopLevel(ctx: var GpuContext, ast: GpuAst, kernel: string, varBlock: var GpuAst) =
  ## Registers top-level procs in allFnTab, global fns in fnTab,
  ## top-level vars in `varBlock`. A non-empty `kernel` narrows fnTab to that kernel.
  case ast.kind
  of gpuProc:
    ctx.allFnTab[ast.pName] = ast
    if ast.isGlobal() and (kernel.len == 0 or ast.pName.ident() == kernel):
      ctx.fnTab[ast.pName] = ast
  of gpuBlock:
    for ch in ast:
      ctx.farmTopLevel(ch, kernel, varBlock)
  of gpuVar, gpuConstexpr:
    varBlock.statements.add ast
  of gpuTypeDef, gpuAlias:
    error "Unexpected type def / alias def found. These should be in `ctx.types` now: " & $ast
  else:
    discard

proc ingestTopLevel*(ctx: var GpuContext, ast: GpuAst, typesFirst: static bool) =
  ## Farms the AST into ctx (fnTab, allFnTab, globalBlocks) ahead of runPasses.
  ## typesFirst: C-family block layout plus callee harvest, WGSL the reverse
  ## and no harvest (its callees inline).
  for k, v in pairs(ctx.genericInsts):
    ctx.allFnTab[k] = v
  var varBlock = GpuAst(kind: gpuBlock)
  ctx.farmTopLevel(ast, "", varBlock)
  when typesFirst:
    for k, typ in ctx.types:
      ctx.globalBlocks.add typ
    # program-scope storage is read-only on the C-family targets:
    # - Metal requires the constant space at program scope
    # - CUDA writes __constant__, OpenCL writes constant
    # - WGSL module-scope vars are per-invocation private and keep their own semantics
    for stmt in varBlock.statements.mitems:
      if stmt.kind == gpuVar and stmt.addressSpace == asRMEM:
        stmt.addressSpace = asConstant
        stmt.vType = stmt.vType.withAddressSpace(asConstant)
        stmt.vName.symbol.typ = stmt.vType
    ctx.globalBlocks.add varBlock
  else:
    ctx.globalBlocks.add varBlock
    var typBlock = GpuAst(kind: gpuBlock)
    for k, typ in ctx.types:
      typBlock.statements.add typ
    ctx.globalBlocks.add typBlock

func registerCommonPasses*(reg: var PassRegistry) =
  ## Registers passes common to all backends, through optimization.
  ## The backend bundles own lowering. Post-validation runs after them.
  reg.registerValidationPrePasses()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()

macro toCrucibleAst*(body: typed): GpuAst =
  ## Converts GPU code to IR (GpuAst) without running any passes.
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  var gpuAst = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  # Clear scope tables before newLit (can't serialize seq/Symbol refs)
  # Scope syms are stored on GpuContext, not gpuBlock, so no walk needed.
  newLit(gpuAst)

macro codegenCuda(body: typed): string =
  ## Compiles a `cuda:` block body into CUDA code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerCudaLoweringPasses()
  reg.registerValidationPostPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  ctx.ingestTopLevel(gpuAst, typesFirst = true)
  runPasses(ctx, reg)
  result = newLit(ctx.codegenCuda())

macro cuda*(body: untyped): string =
  ## Converts the body of this macro into CUDA code.
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctCuda
  result = newCall(bindSym"codegenCuda", body)

macro codegenOpenCL(body: typed): string =
  ## Compiles an `opencl:` block body into OpenCL C code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerOpenclLoweringPasses()
  reg.registerValidationPostPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  ctx.ingestTopLevel(gpuAst, typesFirst = true)
  runPasses(ctx, reg)
  result = newLit(ctx.codegenOpenCL())

macro opencl*(body: untyped): string =
  ## Converts the body of this macro into OpenCL C code.
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctOpenCL
  result = newCall(bindSym"codegenOpenCL", body)

macro codegenGLSL(body: typed): string =
  ## Compiles a `glsl:` block body into GLSL compute shader code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerGlslLoweringPasses()
  reg.registerValidationPostPasses()
  reg.registerGlslValidationPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  ctx.ingestTopLevel(gpuAst, typesFirst = true)
  runPasses(ctx, reg)
  let src = ctx.codegenGLSL()
  result = newLit(src)

macro glsl*(body: untyped): string =
  ## Converts the body of this macro into GLSL compute shader code.
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctGlsl
  result = newCall(bindSym"codegenGLSL", body)

macro codegenSpirvShader(body: typed): seq[uint32] =
  ## Compiles a `spirv:` block body into binary SPIR-V words (Vulkan environment).
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerSpirvLoweringPasses()
  reg.registerValidationPostPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  ctx.ingestTopLevel(gpuAst, typesFirst = true)
  runPasses(ctx, reg)
  result = newLit(ctx.codegenSpirvShader())

macro spirv*(body: untyped): seq[uint32] =
  ## Converts the body of this macro into binary SPIR-V words (Vulkan environment).
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctSpirvShader
  result = newCall(bindSym"codegenSpirvShader", body)

macro codegenWebGpu(body: typed): string =
  ## Compiles a `webgpu:` block body into WebGPU WGSL code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerWgslLoweringPasses()
  reg.registerValidationPostPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  ctx.ingestTopLevel(gpuAst, typesFirst = false)
  runPasses(ctx, reg)
  result = newLit(ctx.codegenWebGpu())

macro webgpu*(body: untyped): string =
  ## Converts the body of this macro into WebGPU WGSL code.
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctWebGPU
  result = newCall(bindSym"codegenWebGpu", body)

macro codegenMetal(body: typed): string =
  ## Compiles a `metal:` block body into Metal Shading Language (MSL) code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerMetalLoweringPasses()
  reg.registerValidationPostPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  ctx.ingestTopLevel(gpuAst, typesFirst = true)
  runPasses(ctx, reg)
  result = newLit(ctx.codegenMetal())

macro metal*(body: untyped): string =
  ## Converts the body of this macro into Metal Shading Language (MSL) code.
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctMetal
  result = newCall(bindSym"codegenMetal", body)

macro codegenHlsl(flavor: static HlslFlavor, body: typed): string =
  ## Compiles an `hlsl:` block body into HLSL compute shader code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerHlslLoweringPasses()
  reg.registerValidationPostPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  ctx.ingestTopLevel(gpuAst, typesFirst = true)
  runPasses(ctx, reg)
  result = newLit(ctx.codegenHlsl(flavor))

macro hlsl*(flavor: static HlslFlavor = fvD3D12, body: untyped): string =
  ## Converts the body of this macro into HLSL compute shader code,
  ## resource binding per `flavor`, D3D12-native by default.
  ##
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctHlsl
  # `flavor` is a static macro param, it reads as the value inside the body
  result = newCall(bindSym"codegenHlsl", newLit(flavor), body)

macro hlsl*(body: untyped): string =
  ## Converts the body of this macro into HLSL compute shader code,
  ## D3D12-native resource binding.
  crucibleCompileTarget = ctHlsl
  result = newCall(bindSym"codegenHlsl", newLit(fvD3D12), body)
