# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, tables]

import ./ir/gpu_types
import ./targets/targets_lang
import ./ir/nim_to_gpu
import workspace/crucible/vendor/wgpu
import ./passes/pass_datatypes
import ./passes/pass_registry
import ./passes/passes_legalizations
import ./passes/passes_legalization_vulkan
import ./passes/passes_validations
import ./passes/passes_optimizations
import ./passes/passes_lowering
import ./passes/passes_normalizations
import ./passes/passes_preprocessing

import ./builtins/builtins # all the builtins for the backend to make the Nim compiler happy
export builtins
import ./builtins/builtins_compilermagic
export builtins_compilermagic

template registerCommonPasses*(reg: var PassRegistry) =
  ## Register passes common to all backends.
  reg.registerValidationPrePasses()
  reg.registerNormalizationPasses()
  reg.registerLegalizationPasses()
  reg.registerPreprocessingPasses()
  reg.registerOptimizationPasses()
  reg.registerLoweringPasses()
  reg.registerValidationPostPasses()

macro toGpuAst*(body: typed): GpuAst =
  ## Converts GPU code to IR (GpuAst) without running any passes.
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  var gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  # Clear scope tables before newLit (can't serialize seq/Symbol refs)
  # Scope syms are stored on GpuContext, not gpuBlock, so no walk needed.
  newLit(gpuAst)

macro codegenCuda(body: typed): string =
  ## Compiles a `cuda:` block body into CUDA code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.register("materializePassByRefArgs", pkTransform, phaseMain,
    "Wraps non-lvalue passByRef args in gpuMaterialize nodes",
    dependsOn = @["ensureBlock"],
    run = materializePassByRefArgs
  )
  reg.register("rejectCUDAKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved CUDA keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["__global__", "__device__", "__shared__", "__constant__"], "CUDA")
  )
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  result = newLit(ctx.codegenCuda(gpuAst))

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
  reg.register("materializePassByRefArgs", pkTransform, phaseMain,
    "Wraps non-lvalue passByRef args in gpuMaterialize nodes",
    dependsOn = @["ensureBlock"],
    run = materializePassByRefArgs
  )
  reg.register("rejectOpenCLKeywords", pkValidation, phaseEarly,
    "Rejects identifiers that are reserved OpenCL C keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["kernel", "__kernel", "global", "__global",
        "local", "__local", "constant", "__constant",
        "read_only", "write_only", "read_write"], "OpenCL C")
  )
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  result = newLit(ctx.codegenOpenCL(gpuAst))

macro opencl*(body: untyped): string =
  ## Converts the body of this macro into OpenCL C code.
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctOpenCL
  result = newCall(bindSym"codegenOpenCL", body)

macro codegenVulkan(body: typed): string =
  ## Compiles a `vulkan:` block body into GLSL compute shader code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  # Vulkan-only legalizations: var params → value, struct-with-ptr-field
  # values → flattened leaves, device-fn ptr params → per-call-site SSBO
  # binding (all gated on ctVulkan inside the passes).
  reg.registerLegalizationVulkanPasses()
  reg.register("rejectVulkanKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved GLSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["extern", "interface", "buffer"], "GLSL")
  )
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  result = newLit(ctx.codegenVulkan(gpuAst))

macro vulkan*(body: untyped): string =
  ## Converts the body of this macro into GLSL compute shader code.
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctVulkan
  result = newCall(bindSym"codegenVulkan", body)

macro codegenWebGpu(body: typed): string =
  ## Compiles a `webgpu:` block body into WebGPU WGSL code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.register("rejectWGSLKeywords", pkValidation, phaseEarly,
    "Rejects identifiers that are reserved WGSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["override", "storage", "uniform", "workgroup"], "WGSL")
  )
  reg.registerWgslPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  result = newLit(ctx.codegenWebGpu(gpuAst))

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
  reg.register("rejectMetalKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved MSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["kernel", "device", "constant", "threadgroup"], "MSL")
  )
  reg.registerMetalPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  result = newLit(ctx.codegenMetal(gpuAst))

macro metal*(body: untyped): string =
  ## Converts the body of this macro into Metal Shading Language (MSL) code.
  # The untyped -> typed macro delegation dance
  # allows delaying compileTime const resolution
  # until after crucibleCompileTarget is updated
  crucibleCompileTarget = ctMetal
  result = newCall(bindSym"codegenMetal", body)

proc codegen*(gen: GpuGenericsInfo, ast: GpuAst, kernel: string = "",
              backend: BackendKind = bkCuda): string =
  ## Generates the code based on the given AST (optionally at runtime) and restricts
  ## it to a single global kernel (WebGPU) if any given.
  ## Default backend is CUDA.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  case backend
  of bkCuda:
    reg.register("materializePassByRefArgs", pkTransform, phaseMain,
      "Wraps non-lvalue passByRef args in gpuMaterialize nodes",
      dependsOn = @["ensureBlock"],
      run = materializePassByRefArgs
    )
    reg.register("rejectCUDAKeywords", pkValidation, phaseMain,
      "Rejects identifiers that are reserved CUDA keywords",
      proc(ctx: var GpuContext): void =
        ctx.checkReservedKeywords(["__global__", "__device__", "__shared__", "__constant__"], "CUDA")
    )
  of bkWGSL:
    reg.register("rejectWGSLKeywords", pkValidation, phaseEarly,
      "Rejects identifiers that are reserved WGSL keywords",
      proc(ctx: var GpuContext): void =
        ctx.checkReservedKeywords(["override", "storage", "uniform", "workgroup"], "WGSL")
    )
    reg.registerWgslPasses()
  of bkOpenCL:
    reg.register("materializePassByRefArgs", pkTransform, phaseMain,
      "Wraps non-lvalue passByRef args in gpuMaterialize nodes",
      dependsOn = @["ensureBlock"],
      run = materializePassByRefArgs
    )
    reg.register("rejectOpenCLKeywords", pkValidation, phaseEarly,
      "Rejects identifiers that are reserved OpenCL C keywords",
      proc(ctx: var GpuContext): void =
        ctx.checkReservedKeywords(["kernel", "__kernel", "global", "__global",
          "local", "__local", "constant", "__constant",
          "read_only", "write_only", "read_write"], "OpenCL C")
    )
  of bkVulkan:
    reg.register("rejectVulkanKeywords", pkValidation, phaseMain,
      "Rejects identifiers that are reserved GLSL keywords",
      proc(ctx: var GpuContext): void =
        ctx.checkReservedKeywords(["extern", "interface", "buffer"], "GLSL")
    )
  of bkMetal:
    # The runtime path fills `genericInsts`, not `allFnTab`, so the keyword pass
    # above iterates nothing here. Reserved identifiers are rejected at emission
    # instead. The printer's `checkReservedIdent` guards params, locals, fields, and function names.
    reg.register("rejectMetalKeywords", pkValidation, phaseMain,
      "Rejects identifiers that are reserved MSL keywords",
      proc(ctx: var GpuContext): void =
        ctx.checkReservedKeywords(["kernel", "device", "constant", "threadgroup"], "MSL")
    )
    reg.registerMetalPasses()
  # Clone IR before running passes — passes mutate in place,
  # so without cloning, one call contaminates later calls for a different backend.
  let astCopy = ast.clone()
  for fn in gen.procs:
    let fnCopy = fn.clone()
    ctx.genericInsts[fnCopy.pName] = fnCopy
  for typ in gen.types:
    let typCopy = typ.clone()
    case typCopy.kind
    of gpuTypeDef:
      ctx.types[typCopy.tTyp] = typCopy
    of gpuAlias:
      ctx.types[typCopy.aTyp] = typCopy
    else: raiseAssert "Unexpected node kind assigning to `types`: " & $typ
  runPasses(ctx, reg)
  case backend
  of bkCuda:
    result = ctx.codegenCuda(astCopy, kernel)
  of bkWGSL:
    result = ctx.codegenWebGpu(astCopy, kernel)
  of bkOpenCL:
    result = ctx.codegenOpenCL(astCopy, kernel)
  of bkVulkan:
    result = ctx.codegenVulkan(astCopy, kernel)
  of bkMetal:
    result = ctx.codegenMetal(astCopy, kernel)
