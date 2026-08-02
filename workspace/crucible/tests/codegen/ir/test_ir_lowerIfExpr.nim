## Phase 4: lowerIfExpr pass test
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_lowerIfExpr.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_normalizations

# ═══════════════════════════════════════════════════════════════════════
# 1. gpuIf(isExpr: true) → gpuTernary
# ═══════════════════════════════════════════════════════════════════════
block:
  let cond = GpuAst(kind: gpuLit, lValue: "true", lType: GpuType(kind: gtBool))
  let thenVal = GpuAst(kind: gpuLit, lValue: "42", lType: GpuType(kind: gtInt32))
  let elseVal = GpuAst(kind: gpuLit, lValue: "0", lType: GpuType(kind: gtInt32))
  var ifExpr = GpuAst(kind: gpuIf, ifIsExpr: true,
                      ifCond: cond, ifThen: thenVal, ifElse: elseVal)

  var fnBody = GpuAst(kind: gpuBlock, statements: @[ifExpr])
  lowerIfExprImpl(fnBody)

  doAssert fnBody.statements.len == 1
  let tern = fnBody.statements[0]
  doAssert tern.kind == gpuTernary, "gpuIf(isExpr:true) should become gpuTernary, got " & $tern.kind
  doAssert tern.tCond.lValue == "true"
  doAssert tern.tThen.lValue == "42"
  doAssert tern.tElse.kind != gpuDiscard
  doAssert tern.tElse.lValue == "0"
  echo "  OK — gpuIf(isExpr: true) lowered to gpuTernary"

# ═══════════════════════════════════════════════════════════════════════
# 2. gpuIf(ifIsExpr: false) is NOT converted
# ═══════════════════════════════════════════════════════════════════════
block:
  let cond = GpuAst(kind: gpuLit, lValue: "true", lType: GpuType(kind: gtBool))
  let thenVal = GpuAst(kind: gpuLit, lValue: "42", lType: GpuType(kind: gtInt32))
  var stmtIf = GpuAst(kind: gpuIf, ifIsExpr: false,
                      ifCond: cond, ifThen: thenVal, ifElse: GpuAst(kind: gpuDiscard))

  var fnBody = GpuAst(kind: gpuBlock, statements: @[stmtIf])
  lowerIfExprImpl(fnBody)

  let stmt = fnBody.statements[0]
  doAssert stmt.kind == gpuIf, "gpuIf(ifIsExpr:false) should remain gpuIf, got " & $stmt.kind
  doAssert stmt.ifIsExpr == false
  echo "  OK — gpuIf(ifIsExpr: false) is NOT converted"

# ═══════════════════════════════════════════════════════════════════════
# 3. Nested gpuIf(isExpr: true) (elif chains)
# ═══════════════════════════════════════════════════════════════════════
block:
  let cond1 = GpuAst(kind: gpuLit, lValue: "true", lType: GpuType(kind: gtBool))
  let v1 = GpuAst(kind: gpuLit, lValue: "10", lType: GpuType(kind: gtInt32))
  let cond2 = GpuAst(kind: gpuLit, lValue: "false", lType: GpuType(kind: gtBool))
  let v2 = GpuAst(kind: gpuLit, lValue: "20", lType: GpuType(kind: gtInt32))
  let v3 = GpuAst(kind: gpuLit, lValue: "30", lType: GpuType(kind: gtInt32))

  let innerIf = GpuAst(kind: gpuIf, ifIsExpr: true,
                       ifCond: cond2, ifThen: v2, ifElse: v3)
  var outerIf = GpuAst(kind: gpuIf, ifIsExpr: true,
                       ifCond: cond1, ifThen: v1, ifElse: innerIf)

  var fnBody = GpuAst(kind: gpuBlock, statements: @[outerIf])
  lowerIfExprImpl(fnBody)

  let outer = fnBody.statements[0]
  doAssert outer.kind == gpuTernary, "Outer should be ternary"
  doAssert outer.tThen.lValue == "10"
  doAssert outer.tElse.kind == gpuTernary, "Inner should also be ternary"
  doAssert outer.tElse.tCond.lValue == "false"
  doAssert outer.tElse.tThen.lValue == "20"
  doAssert outer.tElse.tElse.lValue == "30"
  echo "  OK — nested gpuIf(isExpr: true) lowered to nested gpuTernary"

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All lowerIfExpr tests passed."
