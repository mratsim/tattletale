## WGSL: reduction builtin emission pin.
##
## The `webgpu` DSL macro requires the wgpu-native shared library, which is
## absent on this machine, so this pin drives the same compiler pipeline the
## macro runs (gpu_compiler minus the lib check) and asserts the exact
## emitted text: the `enable subgroups;` module feature plus the
## `subgroupShuffleDown`/`subgroupShuffle` spellings.
##
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_reduction_builtins.nim

import std/[strutils, macros]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/passes/passes_preprocessing
import workspace/crucible/src/codegen/targets/targets_lang
import workspace/crucible/src/codegen/ir/nim_to_gpu
import workspace/crucible/src/codegen/ir/gpu_types

macro webgpuEmit*(body: typed): string =
  ## The `webgpu` macro from gpu_compiler minus the wgpu-native lib check.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerWgslPasses()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  result = newLit(ctx.codegenWebGpu(gpuAst))

const kernelCode = webgpuEmit:
  proc reductionKernel(output: ptr UncheckedArray[float32]) {.global.} =
    let acc = output[0]
    let v = simdShuffleDown(acc, 1'u32)
    output[1] = v
    let w = simdShuffle(v, 0'u32)
    output[2] = w

proc runTest() =
  doAssert "enable subgroups;" in kernelCode,
    "WGSL missing the `enable subgroups;` feature line:\n" & kernelCode
  doAssert "subgroupShuffleDown(acc, 1u)" in kernelCode,
    "WGSL subgroupShuffleDown spelling not emitted:\n" & kernelCode
  doAssert "subgroupShuffle(v, 0u)" in kernelCode,
    "WGSL subgroupShuffle spelling not emitted:\n" & kernelCode
  echo "  OK — WGSL reduction builtin emission (enable subgroups; + shuffle spellings)"

when isMainModule:
  runTest()
