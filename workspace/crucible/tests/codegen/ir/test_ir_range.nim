## Phase 4: deEmbedForRangeAdjustment pass test
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_range.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_normalizations

# ═══════════════════════════════════════════════════════════════════════
# 1. Inclusive range (rkInclusive) is preserved
# ═══════════════════════════════════════════════════════════════════════
block:
  let intTyp = GpuType(kind: gtInt32)
  let fSym = newSymbol("i", iSym = "i_h1", typ = intTyp)
  let fVar = GpuAst(kind: gpuIdent, symbol: fSym)
  let fStart = GpuAst(kind: gpuLit, lValue: "0", lType: intTyp)
  let fEnd = GpuAst(kind: gpuLit, lValue: "10", lType: intTyp)
  var forLoop = GpuAst(kind: gpuFor, fVar: fVar, fStart: fStart, fEnd: fEnd,
                       fRangeKind: rkInclusive, fBody: GpuAst(kind: gpuBlock))

  deEmbedForRangeAdjustmentImpl(forLoop)
  doAssert forLoop.fRangeKind == rkInclusive, "Inclusive range should remain rkInclusive"
  doAssert forLoop.fEnd.lValue == "10", "End value 10 should be preserved"
  echo "  OK — inclusive range preserved"

# ═══════════════════════════════════════════════════════════════════════
# 2. Exclusive range (rkExclusive) is preserved
# ═══════════════════════════════════════════════════════════════════════
block:
  let intTyp = GpuType(kind: gtInt32)
  let fSym = newSymbol("i", iSym = "i_h2", typ = intTyp)
  let fVar = GpuAst(kind: gpuIdent, symbol: fSym)
  let fStart = GpuAst(kind: gpuLit, lValue: "0", lType: intTyp)
  let fEnd = GpuAst(kind: gpuLit, lValue: "10", lType: intTyp)
  var forLoop = GpuAst(kind: gpuFor, fVar: fVar, fStart: fStart, fEnd: fEnd,
                       fRangeKind: rkExclusive, fBody: GpuAst(kind: gpuBlock))

  deEmbedForRangeAdjustmentImpl(forLoop)
  doAssert forLoop.fRangeKind == rkExclusive, "Exclusive range should remain rkExclusive"
  echo "  OK — exclusive range preserved"

# ═══════════════════════════════════════════════════════════════════════
# 3. No false-positive on non-+1 binop
# ═══════════════════════════════════════════════════════════════════════
block:
  let intTyp = GpuType(kind: gtInt32)
  let fSym = newSymbol("i", iSym = "i_h4", typ = intTyp)
  let fVar = GpuAst(kind: gpuIdent, symbol: fSym)
  let fStart = GpuAst(kind: gpuLit, lValue: "0", lType: intTyp)
  let opSym = newSymbol("*", iSym = "*_h4")
  let opMult = GpuAst(kind: gpuIdent, symbol: opSym)
  let nSym = newSymbol("n", iSym = "n_h4", typ = intTyp)
  let nVal = GpuAst(kind: gpuIdent, symbol: nSym)
  let lit2 = GpuAst(kind: gpuLit, lValue: "2", lType: intTyp)
  let adjustedEnd = GpuAst(kind: gpuBinOp, bType: nil, bOp: opMult, bLeft: nVal, bRight: lit2)
  var forLoop = GpuAst(kind: gpuFor, fVar: fVar, fStart: fStart, fEnd: adjustedEnd,
                       fRangeKind: rkInclusive, fBody: GpuAst(kind: gpuBlock))

  deEmbedForRangeAdjustmentImpl(forLoop)
  doAssert forLoop.fRangeKind == rkInclusive, "Non-+1 binop should not change range kind"
  doAssert forLoop.fEnd.kind == gpuBinOp, "fEnd should remain a binop (not flattened)"
  echo "  OK — no false positive on non-+1 binop"

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All deEmbedForRangeAdjustment tests passed."
