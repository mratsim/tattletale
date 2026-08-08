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
# ═══════════════════════════════════════════════════════════════════════════
# 6. Compound-assign rewrite STRIPS a leading gpuAddr on the LHS
# ═══════════════════════════════════════════════════════════════════════════
# Nim wraps non-simple `+=` lvalues in HiddenAddr (-> gpuAddr), e.g. ceramic's
# `tv[m,n]` statement-list expression. In `x = x + y` the lvalue must be used
# directly on both sides: `(&x) = ...` is not a modifiable lvalue on any
# backend and `(&x) + y` is not a valid read.
block:
  let int32 = GpuType(kind: gtInt32)
  let leftSym = newSymbol("x", iSym = "x_h6", typ = int32)
  let left = GpuAst(kind: gpuIdent, symbol: leftSym)
  let addrLeft = GpuAst(kind: gpuAddr, aOf: left)
  let rightLit = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
  let opIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("+=", iSym = "+="))
  var binOp = GpuAst(kind: gpuBinOp, bType: int32, bOp: opIdent, bLeft: addrLeft, bRight: rightLit)

  let result = rewriteCompoundAssignmentImpl(binOp)
  doAssert result.kind == gpuAssign, "Compound += should become gpuAssign"
  doAssert result.aLeft.kind == gpuIdent, "LHS gpuAddr must be stripped on assignment LHS, got: " & $result.aLeft.kind
  doAssert result.aLeft.symbol.name == "x", "assignment LHS must be the addressed ident"
  doAssert result.aRight.kind == gpuBinOp, "RHS should be a binop"
  doAssert result.aRight.bOp.ident() == "+", "Operator should be '+'"
  doAssert result.aRight.bLeft.kind == gpuIdent, "RHS read must not keep the gpuAddr, got: " & $result.aRight.bLeft.kind
  doAssert result.aRight.bLeft.symbol.name == "x", "RHS read must be the plain ident"
  echo "  OK — gpuAddr stripped from compound-assign LHS (both sides)"

# ═══════════════════════════════════════════════════════════════════════════
# 7. Block-expression LHS: addr stripped, block preserved, RHS is a CLONE
# ═══════════════════════════════════════════════════════════════════════════
# Ceramic `tv[m,n]` expands to a statement-list expression (gpuBlock isExpr)
# that Nim wraps in HiddenAddr. Legalization mutates the assignment LHS in
# place (hoisting block intermediates via statements.setLen) — the RHS read
# must be an independent clone, otherwise that mutation corrupts the RHS.
block:
  let int32 = GpuType(kind: gtInt32)
  let posSym = newSymbol("pos", iSym = "pos_h7", typ = int32, symKind = gsLocal)
  let posVar = GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: posSym),
                      vType: int32, vInit: GpuAst(kind: gpuLit, lValue: "0", lType: int32))
  let arrSym = newSymbol("arr", iSym = "arr_h7",
                          typ = GpuType(kind: gtArray, aTyp: int32, aLen: 4),
                          symKind = gsLocal)
  let tail = GpuAst(kind: gpuIndex,
                    iArr: GpuAst(kind: gpuIdent, symbol: arrSym),
                    iIndex: GpuAst(kind: gpuIdent, symbol: posSym))
  let lhsBlock = GpuAst(kind: gpuBlock, isExpr: true, statements: @[posVar, tail])
  let addrBlock = GpuAst(kind: gpuAddr, aOf: lhsBlock)
  let rightLit = GpuAst(kind: gpuLit, lValue: "1", lType: int32)
  let opIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("+=", iSym = "+="))
  var binOp = GpuAst(kind: gpuBinOp, bType: int32, bOp: opIdent, bLeft: addrBlock, bRight: rightLit)

  let result = rewriteCompoundAssignmentImpl(binOp)
  doAssert result.kind == gpuAssign, "Compound += should become gpuAssign"
  doAssert result.aLeft.kind == gpuBlock, "block LHS must survive (addr stripped), got: " & $result.aLeft.kind
  doAssert result.aLeft.statements.len == 2, "block LHS must keep its statements"
  doAssert result.aRight.kind == gpuBinOp, "RHS should be a binop"
  doAssert result.aRight.bLeft.kind == gpuBlock, "RHS read must be the block (clone), got: " & $result.aRight.bLeft.kind
  # Clone independence: legalization hoists LHS block intermediates via
  # statements.setLen — the RHS clone must be unaffected.
  result.aLeft.statements.setLen(1)
  doAssert result.aRight.bLeft.statements.len == 2,
    "RHS block must be an independent clone (LHS hoisting must not corrupt it), got: " & $result.aRight.bLeft.statements.len
  echo "  OK — block LHS: addr stripped, block preserved, RHS cloned"

echo ""
echo "  All rewriteCompoundAssignment tests passed."
