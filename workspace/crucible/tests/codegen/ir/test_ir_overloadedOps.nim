## Phase 4: resolveOverloadedOperators pass test
##
## Verifies:
## - gpuBinOp with bIsOverloaded=true is converted to gpuCall
## - gpuBinOp with bIsOverloaded=false remains gpuBinOp
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_overloadedOps.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_normalizations

# ═══════════════════════════════════════════════════════════════════════
# 1. bIsOverloaded=false remains gpuBinOp
# ═══════════════════════════════════════════════════════════════════════
block:
  var binOp = GpuAst(kind: gpuBinOp, bIsOverloaded: false,
                     bLeft: GpuAst(kind: gpuLit, lValue: "1"),
                     bRight: GpuAst(kind: gpuLit, lValue: "2"))

  var ctx = GpuContext()
  resolveOverloadedOperatorsImpl(ctx, binOp)
  doAssert binOp.kind == gpuBinOp,
    "bIsOverloaded=false should remain gpuBinOp, got " & $binOp.kind
  echo "  OK — bIsOverloaded=false remains gpuBinOp"

# ═══════════════════════════════════════════════════════════════════════
# 2. bIsOverloaded=true is converted to gpuCall
# ═══════════════════════════════════════════════════════════════════════
block:
  let opSym = newSymbol("+", iSym = "+_h2", symKind = gsProc)
  let opPlus = GpuAst(kind: gpuIdent, symbol: opSym)
  var binOp = GpuAst(kind: gpuBinOp, bOp: opPlus, bIsOverloaded: true,
                     bLeft: GpuAst(kind: gpuLit, lValue: "1"),
                     bRight: GpuAst(kind: gpuLit, lValue: "2"))

  var ctx = GpuContext()
  resolveOverloadedOperatorsImpl(ctx, binOp)
  doAssert binOp.kind == gpuCall,
    "bIsOverloaded=true should be converted to gpuCall, got " & $binOp.kind
  doAssert binOp.cName.symbol.name == "+",
    "Call name should be the operator, got '" & binOp.cName.symbol.name & "'"
  doAssert binOp.cArgs.len == 2, "Call should have 2 arguments"
  echo "  OK — bIsOverloaded=true converted to gpuCall"

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All resolveOverloadedOperators tests passed."
