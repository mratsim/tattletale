## explodePointerStructParams lowers a fixed-size array field of pointer-free
## elements to one by-value array param, body accesses rewrite onto it.
## An array field carrying pointer-bearing elements fails compilation:
## manual_explode_array_ptr_field.nim covers that case.
##
## Run:
##   nim c -r --hints:off --warnings:off workspace/crucible/tests/ir/legalizations/test_explode_array_fields.nim (from tattletale)

import std/[algorithm, macros, sequtils, sets, strutils, tables]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/nim_to_gpu
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/passes/passes_legalizations

type
  Buf = object
    data: ptr UncheckedArray[float32]
    extents: array[2, int32]

proc typeStr(t: GpuType): string =
  case t.kind
  of gtPtr:
    let inner = if t.to.kind == gtUA: t.to.uaTo else: t.to
    result = "ptr<" & typeStr(inner) & ">"
  of gtUA: result = "ua<" & typeStr(t.uaTo) & ">"
  of gtArray: result = typeStr(t.aTyp) & "[" & $t.aLen & "]"
  of gtObject: result = t.name
  of gtInt32: result = "int32"
  of gtFloat32: result = "float32"
  else: result = $t.kind

macro explodeReport(body: typed): string =
  ## Runs the body through the legalization passes and reports the exploded
  ## parameter list of each kernel.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerExplodePointerStructParams()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  var report = ""
  for fnKey, fn in ctx.allFnTab:
    if fn.kind == gpuProc and attGlobal in fn.pAttributes:
      report.add "kernel " & fn.pName.ident() & "\n"
      for i, p in fn.pParams:
        report.add "  param " & $i & ": " & p.ident.ident() &
          " : " & typeStr(p.typ) & "\n"
  result = newLit(report)

proc runTest() =
  block:
    const report = explodeReport:
      proc bufKernel(outArr: ptr UncheckedArray[float32], b: Buf) {.global.} =
        let i = b.extents[0] + b.extents[1]
        outArr[i] = b.data[0]

    # the array field is one by-value value-leaf param, body accesses
    # rewrite onto it
    doAssert "param 0: outArr : ptr<float32>" in report, report
    doAssert "param 1: b_data : ptr<float32>" in report, report
    doAssert "param 2: b_extents : int32[2]" in report, report
    doAssert "Buf" notin report,
      "whole Buf param survived:\n" & report

when isMainModule:
  runTest()
