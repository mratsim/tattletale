## Runtime codegen entry
##
## Test that generated source drops static arguments from both the callee
## signature and the kernel's call site.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/types/t_ir_codegenEntryPrune.nim

import std/[macros, strutils, tables]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/ir/nim_to_gpu

proc mix(scale: static int, outP: ptr UncheckedArray[float32], a, b: int) =
  outP[0] = float32(scale) * float32(a + b)

macro parseWithInsts(body: typed): untyped =
  ## Runs the frontend like `toGpuAst` but also hands out the instantiations
  ## the way the runtime codegen entry receives them
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = nim_to_gpu.toGpuAst(ctx, typeReg, body)
  ctx.types = typeReg.types
  var gen = GpuGenericsInfo()
  for k, v in ctx.genericInsts:
    gen.procs.add v
  result = nnkTupleConstr.newTree(newLit(gpuAst), newLit(gen))

block: # static: the validation passes raise through the compile-time `error` macro
  static:
    let (ast, gen) = parseWithInsts:
      proc kern(outP: ptr UncheckedArray[float32]) {.global.} =
        mix(3, outP, 10, 4)

    let src = codegen(gen, ast, "kern", bkCuda)
    # A definition opens a brace after the params, the forward declaration
    # of the same signature ends on `;`
    doAssert "(float* outP, int a, int b){" in src,
      "callee definition emitted with runtime params only, got:\n" & src
    doAssert "scale" notin src,
      "static param name gone from the emitted source:\n" & src
    doAssert "(outP, 10, 4);" in src,
      "call site keeps the runtime args in order:\n" & src
    doAssert "(3, outP" notin src,
      "static arg erased from the call site:\n" & src

echo "  OK — runtime codegen entry prunes the static param, emits the callee, keeps call arity"
