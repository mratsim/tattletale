# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, sequtils, tables]

import ./ir/gpu_types
import ./targets/targets_lang
import ./ir/nim_to_gpu
import ./passes/pass_datatypes
import ./passes/pass_registry
import ./passes/passes_legalizations
import ./passes/passes_validations
import ./passes/passes_optimizations
import ./passes/passes_lowering

import ./builtins/builtins # all the builtins for the backend to make the Nim compiler happy
export builtins

macro toGpuAst*(body: typed): GpuAst =
  ## Converts GPU code to IR (GpuAst) without running any passes.
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  newLit(gpuAst)

macro cuda*(body: typed): string =
  ## Converts the body of this macro into CUDA code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerValidationPrePasses()
  reg.register("rejectCUDAKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved CUDA keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["__global__", "__device__", "__shared__", "__constant__"], "CUDA")
  )
  reg.registerLegalizationPasses()
  reg.register("materializePassByRefArgs", pkTransform, phaseMain,
    "Wraps non-lvalue passByRef args in gpuMaterialize nodes",
    dependsOn = @["ensureBlock"],
    run = materializePassByRefArgs
  )
  reg.registerLoweringPasses()
  reg.registerValidationPostPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  let body = ctx.codegenCuda(gpuAst)
  result = newLit(body)

macro vulkan*(body: typed): string =
  ## Converts the body of this macro into GLSL compute shader code.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerValidationPrePasses()
  reg.registerLegalizationPasses()
  reg.registerValidationPostPasses()
  reg.register("rejectVulkanKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved GLSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["extern", "interface", "buffer"], "GLSL")
  )
  reg.registerLegalizationPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  let body = ctx.codegenVulkan(gpuAst)
  result = newLit(body)

proc codegen*(gen: GpuGenericsInfo, ast: GpuAst, kernel: string = "",
              backend: BackendKind = bkCuda): string =
  ## Generates the code based on the given AST (optionally at runtime) and restricts
  ## it to a single global kernel (WebGPU) if any given.
  ## Default backend is CUDA.
  var ctx = GpuContext()
  # Clone IR before backend preprocessing — backend preprocessors mutate in place,
  # so without cloning, one call to codegen(..., backend=A) contaminates later calls for backend=B.
  # TODO: Make codegen idempotent
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
  case backend
  of bkCuda:
    result = ctx.codegenCuda(astCopy, kernel)
  of bkWGSL:
    result = ctx.codegenWebGpu(astCopy, kernel)
  of bkOpenCL:
    result = ctx.codegenOpenCL(astCopy, kernel)
  of bkVulkan:
    result = ctx.codegenVulkan(astCopy, kernel)

when isMainModule:
  # Mini example
  let kernel = cuda:
    proc square(x: float32): float32 {.device.} =
      if x < 0.0'f32:
        result = 0.0'f32
      else:
        result = x * x

    proc computeSquares(
      output: ptr float32,
      input: ptr float32,
      n: int32
    ) {.global.} =
      let idx: uint32 = blockIdx.x * blockDim.x + threadIdx.x
      if idx < n:
        var temp: float32 = 0.0'f32
        for i in 0..<4:
          temp += square(input[idx + i * n])
        output[idx] = temp

  echo kernel
