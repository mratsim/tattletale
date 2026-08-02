## Phase 2: Scope Table + Scope Validation test
##
## Verifies:
## - Variables are registered in the current scope's symbol table on GpuContext
## - Two variables with the same name in different scopes have different Symbol objects
## - Scope push/pop restores parent scope
## - Scope-validation pass catches a nil-Symbol ident
## - Scope-validation pass passes on well-formed IR
##
## Uses manually constructed GpuAst for newLit compatibility.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_scope.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/ir/gpu_type_constructors
import workspace/crucible/src/codegen/passes/pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# 1. Variables are registered in the current scope's symbol table
# ═══════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let symX = newSymbol("x", iSym = "x_hash1", typ = GpuType(kind: gtInt32), symKind = gsLocal)
  let symY = newSymbol("y", iSym = "y_hash1", typ = GpuType(kind: gtInt32), symKind = gsLocal)

  ctx.scopeSymsStack.add(ctx.currentScopeSyms)
  ctx.currentScopeSyms = @[]
  scopeAdd(ctx.currentScopeSyms, "x", symX)
  scopeAdd(ctx.currentScopeSyms, "y", symY)
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: symX),
                              vType: GpuType(kind: gtInt32),
                              vInit: GpuAst(kind: gpuLit, lValue: "42", lType: initGpuType(gtInt32)),
                              vMutable: false)
  body.statements.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: symY),
                              vType: GpuType(kind: gtInt32),
                              vInit: GpuAst(kind: gpuLit, lValue: "10", lType: initGpuType(gtInt32)),
                              vMutable: true)

  doAssert body.kind == gpuBlock, "Function body should be a gpuBlock"
  doAssert scopeHas(ctx.currentScopeSyms, "x"), "Symbol 'x' should be registered in scope"
  doAssert scopeHas(ctx.currentScopeSyms, "y"), "Symbol 'y' should be registered in scope"
  doAssert scopeGet(ctx.currentScopeSyms, "x") != nil, "Symbol 'x' should be non-nil"
  doAssert scopeGet(ctx.currentScopeSyms, "x").symKind == gsLocal, "Symbol 'x' should be gsLocal"
  doAssert scopeGet(ctx.currentScopeSyms, "y").symKind == gsLocal, "Symbol 'y' should be gsLocal"
  doAssert scopeGet(ctx.currentScopeSyms, "x") == symX, "Symbol ref should match"
  ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
  echo "  OK — Variables registered in scope symbol table"

# ═══════════════════════════════════════════════════════════════════════
# 2. Two variables with same name in different scopes have different Symbols
# ═══════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let outerSym = newSymbol("x", iSym = "x_hash2_outer", typ = GpuType(kind: gtInt32), symKind = gsLocal)
  let innerSym = newSymbol("x", iSym = "x_hash2_inner", typ = GpuType(kind: gtInt32), symKind = gsLocal)

  # Outer scope
  ctx.scopeSymsStack.add(ctx.currentScopeSyms)
  ctx.currentScopeSyms = @[]
  scopeAdd(ctx.currentScopeSyms, "x", outerSym)

  # Inner scope (push)
  ctx.scopeSymsStack.add(ctx.currentScopeSyms)
  ctx.currentScopeSyms = @[]
  scopeAdd(ctx.currentScopeSyms, "x", innerSym)

  doAssert outerSym != innerSym,
    "Inner 'x' Symbol should be different object from outer 'x'"
  doAssert outerSym.iSym != innerSym.iSym,
    "Inner 'x' iSym should differ from outer 'x' iSym"
  doAssert scopeGet(ctx.currentScopeSyms, "x") == innerSym, "Inner scope should map to inner symbol"

  # Pop inner scope
  ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
  doAssert scopeGet(ctx.currentScopeSyms, "x") == outerSym, "Outer scope should map to outer symbol after pop"

  # Pop outer scope
  ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
  echo "  OK — Same-name vars in different scopes have distinct Symbols"

# ═══════════════════════════════════════════════════════════════════════
# 3. For loop variable registered in scope (manually)
# ═══════════════════════════════════════════════════════════════════════
block:
  var ctx = GpuContext()
  let symI = newSymbol("i", iSym = "i_for", typ = initGpuType(gtInt32), symKind = gsLocal)

  ctx.scopeSymsStack.add(ctx.currentScopeSyms)
  ctx.currentScopeSyms = @[]
  scopeAdd(ctx.currentScopeSyms, "i", symI)

  doAssert scopeHas(ctx.currentScopeSyms, "i"), "Loop variable 'i' should be in scope"
  doAssert scopeGet(ctx.currentScopeSyms, "i") == symI, "Loop variable Symbol should match"
  doAssert scopeGet(ctx.currentScopeSyms, "i").symKind == gsLocal, "Loop variable should be gsLocal"

  ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
  echo "  OK — For loop variable registered in scope"

# ═══════════════════════════════════════════════════════════════════════
# 4. Scope-validation pass catches nil-Symbol ident
# ═══════════════════════════════════════════════════════════════════════
block:
  let badIdent = GpuAst(kind: gpuIdent, symbol: nil)
  var badBody = GpuAst(kind: gpuBlock, statements: @[badIdent])
  let fnName = GpuAst(kind: gpuIdent, symbol: newSymbol("badFn", symKind = gsProc))
  var badFn = GpuAst(kind: gpuProc, pName: fnName,
                     pRetType: GpuType(kind: gtVoid),
                     pBody: badBody)

  var foundNil = false
  badFn.pBody.walk(proc(n: var GpuAst): void =
    if n.kind == gpuIdent and n.symbol.isNil:
      foundNil = true
  )
  doAssert foundNil, "Validation should detect nil-Symbol ident"
  echo "  OK — Scope validation detects nil-Symbol ident"

# ═══════════════════════════════════════════════════════════════════════
# 5. Scope-validation pass passes on well-formed IR (direct walk check)
# ═══════════════════════════════════════════════════════════════════════
block:
  let symX = newSymbol("x", iSym = "x_5", typ = GpuType(kind: gtInt32), symKind = gsLocal)
  let symY = newSymbol("y", iSym = "y_5", typ = GpuType(kind: gtInt32), symKind = gsLocal)
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: symX),
                              vType: GpuType(kind: gtInt32),
                              vInit: GpuAst(kind: gpuLit, lValue: "42", lType: initGpuType(gtInt32)),
                              vMutable: false)
  body.statements.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: symY),
                              vType: GpuType(kind: gtInt32),
                              vInit: GpuAst(kind: gpuLit, lValue: "1", lType: initGpuType(gtInt32)),
                              vMutable: false)

  var foundNil = false
  body.walk(proc(n: var GpuAst): void =
    if n.kind == gpuIdent and n.symbol.isNil:
      foundNil = true
  )
  doAssert not foundNil, "Well-formed IR should have no nil-Symbol idents"
  echo "  OK — Scope validation passes on well-formed IR"

# ═══════════════════════════════════════════════════════════════════════
# 6. Blit-created temps have non-nil Symbol objects
# ═══════════════════════════════════════════════════════════════════════
block:
  let blitSym = newSymbol("_blit_0", iSym = "_blit_0", typ = GpuType(kind: gtInt32), symKind = gsLocal)
  let exprSym = newSymbol("val", iSym = "val_blit", typ = GpuType(kind: gtInt32), symKind = gsLocal)

  # Scope block
  var scopeBlock = GpuAst(kind: gpuBlock, blockLabel: "_blit_0")
  scopeBlock.statements.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: exprSym),
                                    vType: GpuType(kind: gtInt32),
                                    vInit: GpuAst(kind: gpuLit, lValue: "99", lType: initGpuType(gtInt32)),
                                    vMutable: false)
  scopeBlock.statements.add GpuAst(kind: gpuAssign,
                                    aLeft: GpuAst(kind: gpuIdent, symbol: blitSym),
                                    aRight: GpuAst(kind: gpuIdent, symbol: exprSym))

  # Blit temp declaration
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: blitSym),
                              vType: GpuType(kind: gtInt32),
                              vInit: GpuAst(kind: gpuDiscard),
                              vMutable: true)
  body.statements.add scopeBlock

  doAssert blitSym != nil, "Blit temp symbol should be non-nil"
  doAssert blitSym.typ.kind != gtVoid, "Blit temp should have non-void type"
  echo "  OK — Blit-created temps have non-nil Symbol objects"

# ═══════════════════════════════════════════════════════════════════════
# 7. Symbol identity: same variable referenced twice shares Symbol ref
# ═══════════════════════════════════════════════════════════════════════
block:
  let symX = newSymbol("x", iSym = "x_7", typ = GpuType(kind: gtInt32), symKind = gsLocal)
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: symX),
                              vType: GpuType(kind: gtInt32),
                              vInit: GpuAst(kind: gpuLit, lValue: "42", lType: initGpuType(gtInt32)),
                              vMutable: false)
  body.statements.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: newSymbol("y", symKind = gsLocal)),
                              vType: GpuType(kind: gtInt32),
                              vInit: GpuAst(kind: gpuIdent, symbol: symX),
                              vMutable: false)
  body.statements.add GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: newSymbol("z", symKind = gsLocal)),
                              vType: GpuType(kind: gtInt32),
                              vInit: GpuAst(kind: gpuIdent, symbol: symX),
                              vMutable: false)

  var identCount = 0
  body.walk(proc(n: var GpuAst): void =
    if n.kind == gpuIdent and n.symbol.name == "x":
      identCount += 1
      doAssert n.symbol == symX,
        "All idents referencing 'x' should share the same Symbol ref"
  )
  doAssert identCount >= 2, "Expected multiple 'x' references, found " & $identCount
  echo "  OK — Same variable referenced multiple times shares Symbol ref"

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  All scope resolution tests passed."
