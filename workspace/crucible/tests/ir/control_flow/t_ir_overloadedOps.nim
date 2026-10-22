## resolveOverloadedOperators binop-to-call rewrite on the raw IR
##
## Test that resolveOverloadedOperators turns overloaded gpuBinOp nodes into
## gpuCall nodes and leaves plain gpuBinOp nodes unchanged
##
## - bIsOverloaded=false keeps kind gpuBinOp
## - bIsOverloaded=true becomes gpuCall named by the operator symbol
## - the rewritten gpuCall carries the two operands as cArgs
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_overloadedOps.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_normalizations {.all.}
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

# ═══════════════════════════════════════════════════════════════════════
# 1. bIsOverloaded=false remains gpuBinOp
# ═══════════════════════════════════════════════════════════════════════
static:
  var binOp = GpuAst(kind: gpuBinOp, bType: nil, bIsOverloaded: false,
                     bLeft: GpuAst(kind: gpuLit, lValue: "1"),
                     bRight: GpuAst(kind: gpuLit, lValue: "2"))

  let res = registerResolveOverloadedOperators.runPassOnNode(binOp)[0]
  doAssert binOp.kind == gpuBinOp,
    "bIsOverloaded=false should remain gpuBinOp, got " & $binOp.kind
  echo "  OK — bIsOverloaded=false remains gpuBinOp"

# ═══════════════════════════════════════════════════════════════════════
# 2. bIsOverloaded=true is converted to gpuCall
# ═══════════════════════════════════════════════════════════════════════
static:
  let opSym = newSymbol("+", iSym = "+_h2", symKind = gsProc)
  let opPlus = GpuAst(kind: gpuIdent, symbol: opSym)
  var binOp = GpuAst(kind: gpuBinOp, bType: nil, bOp: opPlus, bIsOverloaded: true,
                     bLeft: GpuAst(kind: gpuLit, lValue: "1"),
                     bRight: GpuAst(kind: gpuLit, lValue: "2"))

  let res = registerResolveOverloadedOperators.runPassOnNode(binOp)[0]
  doAssert res.kind == gpuCall,
    "bIsOverloaded=true should be converted to gpuCall, got " & $res.kind
  doAssert res.cName.symbol.name == "+",
    "Call name should be the operator, got '" & res.cName.symbol.name & "'"
  doAssert res.cArgs.len == 2, "Call should have 2 arguments"
  echo "  OK — bIsOverloaded=true converted to gpuCall"

