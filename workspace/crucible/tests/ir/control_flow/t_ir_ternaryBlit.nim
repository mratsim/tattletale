## Expression-only if-expression branches lower to gpuTernary
##
## Test that a gpuIf(isExpr: true) with bare-expression branches converts
## to a gpuTernary that stays the assignment RHS.
## Branches carrying statements lower to a statement if instead,
## t_ir_ifExprStatementBranches verifies that shape.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_ternaryBlit.nim

import std/sequtils
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_legalizations
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

proc registerLegalize(reg: var PassRegistry) {.nimcall.} =
  reg.registerLegalizationPasses()

proc ident(sym: Symbol): GpuAst = GpuAst(kind: gpuIdent, symbol: sym)

proc lit(t: GpuType, v: string): GpuAst = GpuAst(kind: gpuLit, lType: t, lValue: v)

proc local(name: string, t: GpuType): Symbol =
  newSymbol(name, iSym = name, typ = t, symKind = gsLocal)

proc op(name: string): GpuAst =
  GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = name))

static:
  let i32t = initGpuType(gtInt32)
  let xSym = local("x", i32t)
  let uSym = local("u", i32t)
  let cond = GpuAst(kind: gpuLit, lType: GpuType(kind: gtBool), lValue: "true")
  let thenBranch = GpuAst(kind: gpuBinOp, bOp: op("+"), bType: i32t,
                          bLeft: ident(uSym), bRight: lit(i32t, "1"))
  let elseBranch = GpuAst(kind: gpuBinOp, bOp: op("*"), bType: i32t,
                          bLeft: ident(uSym), bRight: lit(i32t, "2"))
  let k = GpuAst(kind: gpuProc,
                 pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k")),
                 pRetType: GpuType(kind: gtVoid),
                 pBody: GpuAst(kind: gpuBlock, statements: @[
                   GpuAst(kind: gpuAssign,
                          aLeft: ident(xSym), aOp: "=",
                          aRight: GpuAst(kind: gpuIf, ifIsExpr: true,
                                         ifCond: cond, ifThen: thenBranch,
                                         ifElse: elseBranch))]))
  let lowered = registerLegalize.runPass(k)
  let stmts = lowered.pBody.statements

  let assignIdx = stmts.findIt(it.kind == gpuAssign and it.aLeft.symbol.iSym == "x")
  doAssert assignIdx >= 0, "the assignment survives the blit"
  doAssert stmts[assignIdx].aRight.kind == gpuTernary,
    "expression-only branches convert to a ternary, got " & $stmts[assignIdx].aRight.kind
  echo "  OK — expression-only branches convert to a ternary"

echo "t_ir_ternaryBlit: all static assertions passed"
