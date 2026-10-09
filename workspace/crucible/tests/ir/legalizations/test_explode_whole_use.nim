## explodePointerStructParams explodes a struct kernel param in field order
## into leaf params, body field chains rewrite onto them. A whole-value use
## of the exploded param fails compilation:
## manual_explode_whole_use.nim covers that case.
##
## Run:
##   nim c -r --hints:off --warnings:off workspace/crucible/tests/ir/legalizations/test_explode_whole_use.nim (from tattletale)

import std/[algorithm, macros, sequtils, sets, strutils, tables]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/nim_to_gpu
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/passes/passes_legalizations

type
  Layout2D = object
    shape: tuple[rows, cols: int32]
    stride: tuple[rowS, colS: int32]

  View[T] = object
    data: ptr UncheckedArray[T]
    layout: Layout2D

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
        report.add "  param " & $i & ": " & p.ident.ident() & "\n"
  result = newLit(report)

proc runTest() =
  block:
    const report = explodeReport:
      proc k(outArr: ptr UncheckedArray[float32], dst: View[float32]) {.global.} =
        outArr[dst.layout.stride.colS] = dst.data[0]
    doAssert "param 1: dst_data" in report, report
    doAssert "param 2: dst_layout" in report, report

when isMainModule:
  runTest()
