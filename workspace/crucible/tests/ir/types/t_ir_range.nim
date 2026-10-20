## deEmbedForRangeAdjustment pass test
##
## Test that an embedded `+ 1` end adjustment resolves to the bare end
## with rkInclusive, non-+1 ends pass through untouched
##
## - an embedded `+ 1` end forces rkInclusive, mirrored `1 + x` included
## - a non-+1 end binop passes through untouched
## - toGpuAst derives rkInclusive from `..` and rkExclusive from `..<`
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/types/t_ir_range.nim

import workspace/crucible/src/codegen/ir/gpu_ast_datatypes
import workspace/crucible/src/codegen/passes/passes_normalizations
import workspace/crucible/src/codegen/gpu_compiler

proc getProc(ir: GpuAst): GpuAst =
  doAssert ir.kind == gpuBlock and ir.statements[0].kind == gpuProc
  result = ir.statements[0]

proc findFor(fn: GpuAst): GpuAst =
  # the proc body is the for loop directly or a block holding it
  if fn.pBody.kind == gpuFor:
    return fn.pBody
  for stmt in fn.pBody.statements:
    if stmt.kind == gpuFor:
      return stmt
  doAssert false, "no gpuFor node found in the proc body"


block:
  # an embedded `x + 1` end resolves to `x` and forces rkInclusive
  let intTyp = GpuType(kind: gtInt32)
  let fVar = GpuAst(kind: gpuIdent, symbol: newSymbol("i", iSym = "i_pos", typ = intTyp))
  let fStart = GpuAst(kind: gpuLit, lValue: "0", lType: intTyp)
  let xVal = GpuAst(kind: gpuIdent, symbol: newSymbol("x", iSym = "x_pos", typ = intTyp))
  let lit1 = GpuAst(kind: gpuLit, lValue: "1", lType: intTyp)
  let plusOp = GpuAst(kind: gpuIdent, symbol: newSymbol("+", iSym = "+_pos"))
  let adjustedEnd = GpuAst(kind: gpuBinOp, bType: nil, bOp: plusOp, bLeft: xVal, bRight: lit1)
  var forLoop = GpuAst(kind: gpuFor, fVar: fVar, fStart: fStart, fEnd: adjustedEnd,
                       fRangeKind: rkExclusive, fBody: GpuAst(kind: gpuBlock))

  deEmbedForRangeAdjustmentImpl(forLoop)
  doAssert forLoop.fEnd.kind == gpuIdent and forLoop.fEnd.symbol.name == "x",
    "`x + 1` end must resolve to `x`, got " & $forLoop.fEnd.kind
  doAssert forLoop.fRangeKind == rkInclusive,
    "an embedded `+ 1` adjustment forces rkInclusive, got " & $forLoop.fRangeKind

block:
  # a mirrored `1 + x` end resolves the same way
  let intTyp = GpuType(kind: gtInt32)
  let fVar = GpuAst(kind: gpuIdent, symbol: newSymbol("i", iSym = "i_mir", typ = intTyp))
  let fStart = GpuAst(kind: gpuLit, lValue: "0", lType: intTyp)
  let xVal = GpuAst(kind: gpuIdent, symbol: newSymbol("x", iSym = "x_mir", typ = intTyp))
  let lit1 = GpuAst(kind: gpuLit, lValue: "1", lType: intTyp)
  let plusOp = GpuAst(kind: gpuIdent, symbol: newSymbol("+", iSym = "+_mir"))
  let adjustedEnd = GpuAst(kind: gpuBinOp, bType: nil, bOp: plusOp, bLeft: lit1, bRight: xVal)
  var forLoop = GpuAst(kind: gpuFor, fVar: fVar, fStart: fStart, fEnd: adjustedEnd,
                       fRangeKind: rkExclusive, fBody: GpuAst(kind: gpuBlock))

  deEmbedForRangeAdjustmentImpl(forLoop)
  doAssert forLoop.fEnd.kind == gpuIdent and forLoop.fEnd.symbol.name == "x",
    "`1 + x` end must resolve to `x`, got " & $forLoop.fEnd.kind
  doAssert forLoop.fRangeKind == rkInclusive,
    "a mirrored `+ 1` adjustment forces rkInclusive, got " & $forLoop.fRangeKind

block:
  # a non-+1 end binop passes through untouched
  let intTyp = GpuType(kind: gtInt32)
  let fVar = GpuAst(kind: gpuIdent, symbol: newSymbol("i", iSym = "i_h4", typ = intTyp))
  let fStart = GpuAst(kind: gpuLit, lValue: "0", lType: intTyp)
  let opMult = GpuAst(kind: gpuIdent, symbol: newSymbol("*", iSym = "*_h4"))
  let nVal = GpuAst(kind: gpuIdent, symbol: newSymbol("n", iSym = "n_h4", typ = intTyp))
  let lit2 = GpuAst(kind: gpuLit, lValue: "2", lType: intTyp)
  let adjustedEnd = GpuAst(kind: gpuBinOp, bType: nil, bOp: opMult, bLeft: nVal, bRight: lit2)
  var forLoop = GpuAst(kind: gpuFor, fVar: fVar, fStart: fStart, fEnd: adjustedEnd,
                       fRangeKind: rkInclusive, fBody: GpuAst(kind: gpuBlock))

  deEmbedForRangeAdjustmentImpl(forLoop)
  doAssert forLoop.fRangeKind == rkInclusive, "Non-+1 binop should not change range kind"
  doAssert forLoop.fEnd.kind == gpuBinOp, "fEnd should remain a binop (not flattened)"

block:
  # toGpuAst derives rkInclusive from `..`
  let ir = toGpuAst:
    proc rangeTestInc(x: int32) {.device.} =
      for i in 0 .. x:
        var y = x
  let fn = ir.getProc()
  let forNode = fn.findFor()
  doAssert forNode.fRangeKind == rkInclusive,
    "0 .. x must produce rkInclusive, got " & $forNode.fRangeKind

block:
  # toGpuAst derives rkExclusive from `..<`
  let ir = toGpuAst:
    proc rangeTestExc(x: int32) {.device.} =
      for i in 0 ..< x:
        var y = x
  let fn = ir.getProc()
  let forNode = fn.findFor()
  doAssert forNode.fRangeKind == rkExclusive,
    "0 ..< x must produce rkExclusive, got " & $forNode.fRangeKind
echo "t_ir_range: adjustment resolution and range-kind derivation verified"
