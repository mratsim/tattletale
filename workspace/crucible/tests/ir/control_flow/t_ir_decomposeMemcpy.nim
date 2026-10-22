## decomposeMemcpyVars pass
##
## Test that memcpy-flagged var and assign nodes decompose into plain
## declarations plus `__builtin_memcpy` calls, unflagged nodes passing through.
##
## - Flagged gpuVar becomes a plain decl plus a 3-arg memcpy over the old init
## - Flagged gpuAssign becomes a dst/src/size memcpy call outright
## - Unflagged var and assign nodes keep their kind, init, and flags
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_decomposeMemcpy.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering {.all.}

proc runOne(node: GpuAst): GpuAst =
  ## Runs the memcpy-decomposition pass over one fn holding `node`,
  ## returns the transformed statement
  var ctx = GpuContext()
  let fn = GpuAst(kind: gpuProc, pName: newGpuIdent("anon"),
                  pRetType: GpuType(kind: gtVoid),
                  pBody: GpuAst(kind: gpuBlock, statements: @[node]),
                  pParams: @[])
  ctx.allFnTab[fn.pName] = fn
  var reg = PassRegistry.new()
  reg.registerDecomposeMemcpyVars()
  ctx.runPasses(reg)
  let res = fn.pBody.statements
  if res.len == 1:
    res[0]
  else:
    GpuAst(kind: gpuBlock, blockLabel: "_memcpy", statements: res)

# ═══════════════════════════════════════════════════════════════════════
# 1. gpuVar with vRequiresMemcpy is decomposed into var decl + memcpy call
# ═══════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let arrTyp = GpuType(kind: gtArray, aTyp: int32, aLen: 10)
  let initIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("init", iSym = "init_h1", typ = arrTyp))
  let varName = GpuAst(kind: gpuIdent, symbol: newSymbol("x", iSym = "x_h1", typ = arrTyp))
  var varNode = GpuAst(kind: gpuVar, vName: varName, vType: arrTyp,
                       vInit: initIdent, vRequiresMemcpy: true, vMutable: true)
  varNode = runOne(varNode)

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

# ═══════════════════════════════════════════════════════════════════════
# 2. gpuAssign with aRequiresMemcpy is replaced by memcpy call
# ═══════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let arrTyp = GpuType(kind: gtArray, aTyp: int32, aLen: 10)
  let leftIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("a", iSym = "a_h2", typ = arrTyp))
  let rightIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("b", iSym = "b_h2", typ = arrTyp))
  var assign = GpuAst(kind: gpuAssign, aLeft: leftIdent, aRight: rightIdent,
                      aRequiresMemcpy: true)
  assign = runOne(assign)

  doAssert assign.kind == gpuCall, "memcpy assign should become a gpuCall"
  doAssert assign.cName.ident() == "__builtin_memcpy", "Should call __builtin_memcpy"
  doAssert assign.cArgs.len == 3, "memcpy should have 3 args"
  doAssert assign.cArgs[0].kind == gpuAddr, "first arg should be address of left"
  doAssert assign.cArgs[1].kind == gpuAddr, "second arg should be address of right"
  echo "  OK — gpuAssign with RequiresMemcpy replaced by memcpy call"

# ═══════════════════════════════════════════════════════════════════════
# 3. gpuVar without RequiresMemcpy is unchanged
# ═══════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let varName = GpuAst(kind: gpuIdent, symbol: newSymbol("y", iSym = "y_h3", typ = int32))
  let initLit = GpuAst(kind: gpuLit, lValue: "42", lType: int32)
  var varNode = GpuAst(kind: gpuVar, vName: varName, vType: int32,
                       vInit: initLit, vRequiresMemcpy: false, vMutable: false)
  varNode = runOne(varNode)

  doAssert varNode.kind == gpuVar, "Simple var without memcpy should be unchanged"
  doAssert varNode.vInit.lValue == "42", "Initializer should be preserved"
  echo "  OK — gpuVar without RequiresMemcpy is unchanged"

# ═══════════════════════════════════════════════════════════════════════
# 4. gpuAssign without RequiresMemcpy is unchanged
# ═══════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let leftIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("c", iSym = "c_h4", typ = int32))
  let rightLit = GpuAst(kind: gpuLit, lValue: "99", lType: int32)
  var assign = GpuAst(kind: gpuAssign, aLeft: leftIdent, aRight: rightLit,
                      aRequiresMemcpy: false)
  assign = runOne(assign)

  doAssert assign.kind == gpuAssign, "Simple assign without memcpy should be unchanged"
  echo "  OK — gpuAssign without RequiresMemcpy is unchanged"

echo ""
echo "  All decomposeMemcpyVars tests passed."
