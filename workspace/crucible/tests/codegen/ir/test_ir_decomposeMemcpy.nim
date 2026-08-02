## Phase 5: decomposeMemcpyVars pass test
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_decomposeMemcpy.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. gpuVar with vRequiresMemcpy is decomposed into var decl + memcpy call
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let arrTyp = GpuType(kind: gtArray, aTyp: int32, aLen: 10)
  let initIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("init", iSym = "init_h1", typ = arrTyp))
  let varName = GpuAst(kind: gpuIdent, symbol: newSymbol("x", iSym = "x_h1", typ = arrTyp))
  var varNode = GpuAst(kind: gpuVar, vName: varName, vType: arrTyp,
                       vInit: initIdent, vRequiresMemcpy: true, vMutable: true)
  var ctx = GpuContext()

  decomposeMemcpyVarsImpl(ctx, varNode)

  doAssert varNode.kind == gpuBlock, "Should be decomposed into a block"
  doAssert varNode.statements.len == 2, "Block should have 2 statements: var decl + memcpy call"
  doAssert varNode.statements[0].kind == gpuVar, "First statement should be a var decl"
  doAssert not varNode.statements[0].vRequiresMemcpy, "Var decl should not require memcpy anymore"
  doAssert varNode.statements[0].vInit.kind == gpuDiscard, "Var decl should have no init"
  doAssert varNode.statements[1].kind == gpuCall, "Second statement should be a memcpy call"
  doAssert varNode.statements[1].cName.ident() == "__builtin_memcpy", "Should call __builtin_memcpy"
  doAssert varNode.statements[1].cArgs.len == 3, "memcpy should have 3 args: dst, src, size"
  doAssert varNode.statements[1].cArgs[0].kind == gpuAddr, "dst should be address of var"
  doAssert varNode.statements[1].cArgs[1].kind == gpuAddr, "src should be address of init"
  echo "  OK — gpuVar with RequiresMemcpy decomposed into var decl + memcpy call"

# ═══════════════════════════════════════════════════════════════════════════
# 2. gpuAssign with aRequiresMemcpy is replaced by memcpy call
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let arrTyp = GpuType(kind: gtArray, aTyp: int32, aLen: 10)
  let leftIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("a", iSym = "a_h2", typ = arrTyp))
  let rightIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("b", iSym = "b_h2", typ = arrTyp))
  var assign = GpuAst(kind: gpuAssign, aLeft: leftIdent, aRight: rightIdent,
                      aRequiresMemcpy: true)
  var ctx = GpuContext()

  decomposeMemcpyVarsImpl(ctx, assign)

  doAssert assign.kind == gpuCall, "memcpy assign should become a gpuCall"
  doAssert assign.cName.ident() == "__builtin_memcpy", "Should call __builtin_memcpy"
  doAssert assign.cArgs.len == 3, "memcpy should have 3 args"
  doAssert assign.cArgs[0].kind == gpuAddr, "first arg should be address of left"
  doAssert assign.cArgs[1].kind == gpuAddr, "second arg should be address of right"
  echo "  OK — gpuAssign with RequiresMemcpy replaced by memcpy call"

# ═══════════════════════════════════════════════════════════════════════════
# 3. gpuVar without RequiresMemcpy is unchanged
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let varName = GpuAst(kind: gpuIdent, symbol: newSymbol("y", iSym = "y_h3", typ = int32))
  let initLit = GpuAst(kind: gpuLit, lValue: "42", lType: int32)
  var varNode = GpuAst(kind: gpuVar, vName: varName, vType: int32,
                       vInit: initLit, vRequiresMemcpy: false, vMutable: false)
  var ctx = GpuContext()

  decomposeMemcpyVarsImpl(ctx, varNode)

  doAssert varNode.kind == gpuVar, "Simple var without memcpy should be unchanged"
  doAssert varNode.vInit.lValue == "42", "Initializer should be preserved"
  echo "  OK — gpuVar without RequiresMemcpy is unchanged"

# ═══════════════════════════════════════════════════════════════════════════
# 4. gpuAssign without RequiresMemcpy is unchanged
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let leftIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("c", iSym = "c_h4", typ = int32))
  let rightLit = GpuAst(kind: gpuLit, lValue: "99", lType: int32)
  var assign = GpuAst(kind: gpuAssign, aLeft: leftIdent, aRight: rightLit,
                      aRequiresMemcpy: false)
  var ctx = GpuContext()

  decomposeMemcpyVarsImpl(ctx, assign)

  doAssert assign.kind == gpuAssign, "Simple assign without memcpy should be unchanged"
  echo "  OK — gpuAssign without RequiresMemcpy is unchanged"

echo ""
echo "  All decomposeMemcpyVars tests passed."
