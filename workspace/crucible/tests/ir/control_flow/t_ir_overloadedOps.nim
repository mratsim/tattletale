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
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_overloadedOps.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/pass_registry

proc runOnCtxBody(ctx: var GpuContext, n: var GpuAst) =
  ## Runs one registered pass over `n` as a fn body in `ctx`, the public-API path.
  var fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol("f", iSym = "f")),
                  pRetType: GpuType(kind: gtVoid),
                  pBody: GpuAst(kind: gpuBlock, statements: @[n]))
  ctx.allFnTab[fn.pName] = fn
  var reg = PassRegistry.new()
  reg.registerResolveOverloadedOperators()
  ctx.runPasses(reg)
  n = fn.pBody.statements[0]

# ═══════════════════════════════════════════════════════════════════════
# 1. bIsOverloaded=false remains gpuBinOp
# ═══════════════════════════════════════════════════════════════════════
block:
  var binOp = GpuAst(kind: gpuBinOp, bType: nil, bIsOverloaded: false,
                     bLeft: GpuAst(kind: gpuLit, lValue: "1"),
                     bRight: GpuAst(kind: gpuLit, lValue: "2"))

  var ctx = GpuContext()
  runOnCtxBody(ctx, binOp)
  doAssert binOp.kind == gpuBinOp,
    "bIsOverloaded=false should remain gpuBinOp, got " & $binOp.kind
  echo "  OK — bIsOverloaded=false remains gpuBinOp"

# ═══════════════════════════════════════════════════════════════════════
# 2. bIsOverloaded=true is converted to gpuCall
# ═══════════════════════════════════════════════════════════════════════
block:
  let opSym = newSymbol("+", iSym = "+_h2", symKind = gsProc)
  let opPlus = GpuAst(kind: gpuIdent, symbol: opSym)
  var binOp = GpuAst(kind: gpuBinOp, bType: nil, bOp: opPlus, bIsOverloaded: true,
                     bLeft: GpuAst(kind: gpuLit, lValue: "1"),
                     bRight: GpuAst(kind: gpuLit, lValue: "2"))

  var ctx = GpuContext()
  runOnCtxBody(ctx, binOp)
  doAssert binOp.kind == gpuCall,
    "bIsOverloaded=true should be converted to gpuCall, got " & $binOp.kind
  doAssert binOp.cName.symbol.name == "+",
    "Call name should be the operator, got '" & binOp.cName.symbol.name & "'"
  doAssert binOp.cArgs.len == 2, "Call should have 2 arguments"
  echo "  OK — bIsOverloaded=true converted to gpuCall"

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All resolveOverloadedOperators tests passed."
