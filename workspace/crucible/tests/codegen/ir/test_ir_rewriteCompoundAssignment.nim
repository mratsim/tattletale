## Phase 6: rewriteCompoundAssignment pass tests
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_rewriteCompoundAssignment.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. x += y → x = x + y
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let leftSym = newSymbol("x", iSym = "x_h1", typ = int32)
  let left = GpuAst(kind: gpuIdent, symbol: leftSym)
  let rightLit = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
  let opIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("+=", iSym = "+="))
  var binOp = GpuAst(kind: gpuBinOp, bType: nil, bOp: opIdent, bLeft: left, bRight: rightLit)

  let result = rewriteCompoundAssignmentImpl(binOp)
  doAssert result.kind == gpuAssign, "Compound += should become gpuAssign, got: " & $result.kind
  doAssert result.aRight.kind == gpuBinOp, "RHS should be a binop"
  doAssert result.aRight.bOp.ident() == "+", "Operator should be '+', got: '" & result.aRight.bOp.ident() & "'"
  echo "  OK — x += y rewritten to x = x + y"

# ═══════════════════════════════════════════════════════════════════════════
# 2. x *= y → x = x * y
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let leftSym = newSymbol("x", iSym = "x_h2", typ = int32)
  let left = GpuAst(kind: gpuIdent, symbol: leftSym)
  let rightLit = GpuAst(kind: gpuLit, lValue: "2", lType: int32)
  let opIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("*=", iSym = "*="))
  var binOp = GpuAst(kind: gpuBinOp, bType: nil, bOp: opIdent, bLeft: left, bRight: rightLit)

  let result = rewriteCompoundAssignmentImpl(binOp)
  doAssert result.kind == gpuAssign, "Compound *= should become gpuAssign"
  doAssert result.aRight.bOp.ident() == "*", "Operator should be '*'"
  echo "  OK — x *= y rewritten to x = x * y"

# ═══════════════════════════════════════════════════════════════════════════
# 3. Comparison operators (<=, >=, ==, !=) are left unchanged
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let leftLit = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
  let rightLit = GpuAst(kind: gpuLit, lValue: "2", lType: int32)
  for op in ["<=", "==", ">=", "!="]:
    let opIdent = GpuAst(kind: gpuIdent, symbol: newSymbol(op, iSym = op))
    var binOp = GpuAst(kind: gpuBinOp, bType: nil, bOp: opIdent, bLeft: leftLit, bRight: rightLit)
    let result = rewriteCompoundAssignmentImpl(binOp)
    doAssert result.kind == gpuBinOp, "Comparison '" & op & "' should stay as gpuBinOp, got: " & $result.kind
  echo "  OK — comparison operators left unchanged"

# ═══════════════════════════════════════════════════════════════════════════
# 4. Simple binop (not compound assignment) is left unchanged
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let leftLit = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
  let rightLit = GpuAst(kind: gpuLit, lValue: "2", lType: int32)
  let opIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("+", iSym = "+"))
  var binOp = GpuAst(kind: gpuBinOp, bType: nil, bOp: opIdent, bLeft: leftLit, bRight: rightLit)

  let result = rewriteCompoundAssignmentImpl(binOp)
  doAssert result.kind == gpuBinOp, "Simple '+' should stay as gpuBinOp"
  echo "  OK — simple '+' binop left unchanged"

# ═══════════════════════════════════════════════════════════════════════════
# 5. Compound-assign rewrite PRESERVES a real (non-nil) bType
# ═══════════════════════════════════════════════════════════════════════════
# Regression: the rewrite sites copy `bType: n.bType` — a real (non-nil,
# non-void) input value must survive onto the rewritten RHS binop unchanged.
# The fixtures above all use nil, so this asserts the copy is not just
# structurally present but value-correct.
block:
  let int32 = GpuType(kind: gtInt32)
  let leftSym = newSymbol("x", iSym = "x_h5", typ = int32)
  let left = GpuAst(kind: gpuIdent, symbol: leftSym)
  let rightLit = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
  let opIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("+=", iSym = "+="))
  var binOp = GpuAst(kind: gpuBinOp, bType: int32, bOp: opIdent, bLeft: left, bRight: rightLit)

  let result = rewriteCompoundAssignmentImpl(binOp)
  doAssert result.kind == gpuAssign, "Compound += should become gpuAssign"
  doAssert result.aRight.kind == gpuBinOp, "RHS should be a binop"
  doAssert not result.aRight.bType.isNil, "rewritten RHS must carry the copied bType (got nil)"
  doAssert result.aRight.bType == int32, "rewritten RHS must preserve the input bType value"
  echo "  OK — compound-assign rewrite preserves a real bType"

echo ""
echo "  All rewriteCompoundAssignment tests passed."
