## Statement-carrying if-expression branches lower to a statement if
## assigning a fresh temp
##
## Test:
##
##   `result = if cond: (block: (t = 1.0; t + 1)) else: u * 2`
##
## lowers to a `var _ifres_0` declaration, a statement if assigning
## the temp in each branch, and `result = _ifres_0`.
## The unselected branch's statements never run, so the user write
## `t = 1.0` executes only when the then branch is selected.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_ifExprStatementBranches.nim

import std/[sequtils, strutils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_legalizations
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

proc registerLegalize(reg: var PassRegistry) {.nimcall.} =
  reg.registerLegalizationPasses()

static:
  let ir = toCrucibleAst:
    proc k(u: float32): float32 {.device.} =
      var t = 0.0
      result = if true: (block: (t = 1.0; t + 1)) else: u * 2
  let fn = ir.statements[0]
  let lowered = registerLegalize.runPass(fn)
  let stmts = lowered.pBody.statements

  # result = _ifres_0
  let assignIdx = stmts.findIt(it.kind == gpuAssign and
                               it.aLeft.kind == gpuIdent and
                               it.aLeft.symbol.iSym.startsWith("result"))
  doAssert assignIdx >= 0, "the result assignment survives"
  doAssert stmts[assignIdx].aRight.kind == gpuIdent and
           stmts[assignIdx].aRight.symbol.iSym.startsWith("_ifres_"),
    "the if-expression became a temp reference, got " & $stmts[assignIdx].aRight.kind

  # the statement if with the fresh temp assigned in both branches
  let ifIdx = stmts.findIt(it.kind == gpuIf and not it.ifIsExpr)
  doAssert ifIdx >= 0, "a statement if carries the branches, got none"
  let ifStmt = stmts[ifIdx]
  doAssert ifIdx < assignIdx, "the if statement runs before the result assignment"

  # then branch: the user write stays in-branch, the tail assigns the temp
  let thenStmts = ifStmt.ifThen.statements
  doAssert thenStmts.findIt(it.kind == gpuAssign and
                            it.aLeft.kind == gpuIdent and
                            not it.aLeft.symbol.iSym.startsWith("_ifres_")) >= 0,
    "the user write t = 1.0 stays inside the then branch"
  doAssert thenStmts[^1].kind == gpuAssign and
           thenStmts[^1].aLeft.symbol.iSym.startsWith("_ifres_"),
    "the then tail assigns the result temp"

  # else branch: the tail assigns the temp, no user write
  let elseStmts = ifStmt.ifElse.statements
  doAssert elseStmts.findIt(it.kind == gpuAssign and
                            it.aLeft.kind == gpuIdent and
                            not it.aLeft.symbol.iSym.startsWith("_ifres_")) < 0,
    "the else branch never writes t"
  doAssert elseStmts[^1].kind == gpuAssign and
           elseStmts[^1].aLeft.symbol.iSym.startsWith("_ifres_"),
    "the else tail assigns the result temp"
  echo "  OK — statement branches lower to a statement if, user writes stay in-branch"

echo "t_ir_ifExprStatementBranches: all static assertions passed"
