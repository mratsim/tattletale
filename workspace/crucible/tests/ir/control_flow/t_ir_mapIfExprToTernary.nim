## mapIfExprToTernary pass
##
## Test that gpuIf(isExpr: true) rewrites to gpuTernary, nested shapes included,
## while statement-position gpuIf nodes stay untouched.
##
## - Condition, then, and else values carry over into the ternary fields
## - A nested if-expression lowers to a nested ternary in one pass
## - gpuIf(ifIsExpr: false) nodes remain statements, never ternaries
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_mapIfExprToTernary.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_legal_if_expr {.all.}
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

# ═══════════════════════════════════════════════════════════════════════
# 1. gpuIf(isExpr: true) → gpuTernary
# ═══════════════════════════════════════════════════════════════════════
static:
  let cond = GpuAst(kind: gpuLit, lValue: "true", lType: GpuType(kind: gtBool))
  let thenVal = GpuAst(kind: gpuLit, lValue: "42", lType: GpuType(kind: gtInt32))
  let elseVal = GpuAst(kind: gpuLit, lValue: "0", lType: GpuType(kind: gtInt32))
  let ifExpr = GpuAst(kind: gpuIf, ifIsExpr: true,
                      ifCond: cond, ifThen: thenVal, ifElse: elseVal,
                      ifElifs: @[])

  let lowered = registerMapIfExprToTernary.runPassOnNode(ifExpr)[0]

  doAssert lowered.kind == gpuTernary, "gpuIf(isExpr:true) should become gpuTernary, got " & $lowered.kind
  doAssert lowered.tCond.lValue == "true"
  doAssert lowered.tThen.lValue == "42"
  doAssert lowered.tElse.kind != gpuDiscard
  doAssert lowered.tElse.lValue == "0"
  echo "  OK — gpuIf(isExpr: true) lowered to gpuTernary"

# ═══════════════════════════════════════════════════════════════════════
# 2. gpuIf(ifIsExpr: false) is NOT converted
# ═══════════════════════════════════════════════════════════════════════
static:
  let cond = GpuAst(kind: gpuLit, lValue: "true", lType: GpuType(kind: gtBool))
  let thenVal = GpuAst(kind: gpuLit, lValue: "42", lType: GpuType(kind: gtInt32))
  let stmtIf = GpuAst(kind: gpuIf, ifIsExpr: false,
                      ifCond: cond, ifThen: thenVal, ifElse: GpuAst(kind: gpuDiscard),
                      ifElifs: @[])

  let lowered = registerMapIfExprToTernary.runPassOnNode(stmtIf)[0]

  doAssert lowered.kind == gpuIf, "gpuIf(ifIsExpr:false) should remain gpuIf, got " & $lowered.kind
  doAssert lowered.ifIsExpr == false
  echo "  OK — gpuIf(ifIsExpr: false) is NOT converted"

# ═══════════════════════════════════════════════════════════════════════
# 3. Nested gpuIf(isExpr: true) (elif chains)
# ═══════════════════════════════════════════════════════════════════════
static:
  let cond1 = GpuAst(kind: gpuLit, lValue: "true", lType: GpuType(kind: gtBool))
  let v1 = GpuAst(kind: gpuLit, lValue: "10", lType: GpuType(kind: gtInt32))
  let cond2 = GpuAst(kind: gpuLit, lValue: "false", lType: GpuType(kind: gtBool))
  let v2 = GpuAst(kind: gpuLit, lValue: "20", lType: GpuType(kind: gtInt32))
  let v3 = GpuAst(kind: gpuLit, lValue: "30", lType: GpuType(kind: gtInt32))

  let innerIf = GpuAst(kind: gpuIf, ifIsExpr: true,
                       ifCond: cond2, ifThen: v2, ifElse: v3,
                       ifElifs: @[])
  let outerIf = GpuAst(kind: gpuIf, ifIsExpr: true,
                       ifCond: cond1, ifThen: v1, ifElse: innerIf,
                       ifElifs: @[])

  let lowered = registerMapIfExprToTernary.runPassOnNode(outerIf)[0]

  doAssert lowered.kind == gpuTernary, "Outer should be ternary"
  doAssert lowered.tThen.lValue == "10"
  doAssert lowered.tElse.kind == gpuTernary, "Inner should also be ternary"
  doAssert lowered.tElse.tCond.lValue == "false"
  doAssert lowered.tElse.tThen.lValue == "20"
  doAssert lowered.tElse.tElse.lValue == "30"
  echo "  OK — nested gpuIf(isExpr: true) lowered to nested gpuTernary"
