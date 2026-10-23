## Run:
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/ir/legalizations/t_ir_renameReservedIdentifiers.nim

import std/[sets, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering
import workspace/crucible/tests/ir/ir_testutils

const Reserved = ["layout", "buffer"].toHashSet()

proc registerRenameReserved(reg: var PassRegistry) {.nimcall.} =
  reg.registerRenameReservedIdentifiers(name = "renameReserved", reserved = Reserved)

# ═══════════════════════════════════════════════════════════════════════
# 1. fn param renamed in the decl and in body refs
# ═══════════════════════════════════════════════════════════════════════
static:
  let intTy = GpuType(kind: gtInt32)
  let sym = newSymbol("layout", iSym = "layout_param", typ = intTy)
  let param = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: intTy,
                       addressSpace: asRMEM, passByRef: false)
  # body: `layout = 3` as gpuAssign(gpuIdent(layout), 3), the body ref shares the iSym
  let bodyRef = GpuAst(kind: gpuIdent, symbol: newSymbol("layout", iSym = "layout_param", typ = intTy))
  var body = GpuAst(kind: gpuBlock)
  body.statements.add GpuAst(kind: gpuAssign, aLeft: bodyRef,
                             aRight: GpuAst(kind: gpuLit, lValue: "3", lType: intTy))
  var procNode = GpuAst(kind: gpuProc,
                        pName: GpuAst(kind: gpuIdent, symbol: newSymbol("fn")),
                        pParams: @[param], pBody: body)
  var ctx = GpuContext()
  ctx.fnTab[procNode.pName] = procNode

  registerRenameReserved.runPassPrepared(ctx)

  doAssert procNode.pParams[0].ident.ident() == "layout_nim",
    "reserved param name should rename, got: " & procNode.pParams[0].ident.ident()
  doAssert procNode.pBody.statements[0].aLeft.ident() == "layout_nim",
    "body refs of a renamed param should follow, got: " & procNode.pBody.statements[0].aLeft.ident()
  echo "  OK: param decl and body refs renamed"

# ═══════════════════════════════════════════════════════════════════════
# 2. struct field renamed in the decl, including block-wrapped type defs
# ═══════════════════════════════════════════════════════════════════════
static:
  var field = GpuTypeField(name: "buffer", typ: GpuType(kind: gtInt32))
  var typeDef = GpuAst(kind: gpuTypeDef,
                       tTyp: GpuType(kind: gtObject, name: "S"),
                       tFields: @[field])
  # WGSL wraps all type defs in one gpuBlock, GLSL puts them directly in globalBlocks
  var typBlock = GpuAst(kind: gpuBlock)
  typBlock.statements.add typeDef
  var ctx = GpuContext()
  ctx.globalBlocks.add typBlock

  registerRenameReserved.runPassPrepared(ctx)

  doAssert ctx.globalBlocks[0].statements[0].tFields[0].name == "buffer_nim",
    "reserved field name in a block-wrapped type def should rename, got: " &
      ctx.globalBlocks[0].statements[0].tFields[0].name
  echo "  OK: struct field decl renamed in block-wrapped type def"

# ═══════════════════════════════════════════════════════════════════════
# 2.b globals table (WGSL lifts kernel params there) renamed
# ═══════════════════════════════════════════════════════════════════════
static:
  let intTy = GpuType(kind: gtInt32)
  let sym = newSymbol("layout", iSym = "layout_global", typ = intTy)
  let g = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: intTy,
                   addressSpace: asRMEM, passByRef: false)
  var ctx = GpuContext()
  ctx.globals["layout_global"] = g

  registerRenameReserved.runPassPrepared(ctx)

  doAssert ctx.globals["layout_global"].ident.ident() == "layout_nim",
    "global symbol with reserved name should rename, got: " &
      ctx.globals["layout_global"].ident.ident()
  echo "  OK: globals table renamed"

# ═══════════════════════════════════════════════════════════════════════
# 3. field access renamed
# ═══════════════════════════════════════════════════════════════════════
static:
  let intTy = GpuType(kind: gtInt32)
  let fieldSym = newSymbol("layout", iSym = "s_layout", typ = intTy)
  let parentSym = newSymbol("s", iSym = "s_var", typ = GpuType(kind: gtObject, name: "S"))
  let dot = GpuAst(kind: gpuDot,
                   dParent: GpuAst(kind: gpuIdent, symbol: parentSym),
                   dField: GpuAst(kind: gpuIdent, symbol: fieldSym))
  var body = GpuAst(kind: gpuBlock)
  body.statements.add dot
  var procNode = GpuAst(kind: gpuProc,
                        pName: GpuAst(kind: gpuIdent, symbol: newSymbol("fn")),
                        pBody: body)
  var ctx = GpuContext()
  ctx.fnTab[procNode.pName] = procNode

  registerRenameReserved.runPassPrepared(ctx)

  let acc = procNode.pBody.statements[0]
  doAssert acc.dField.ident() == "layout_nim",
    "reserved field access should rename, got: " & acc.dField.ident()
  doAssert acc.dParent.ident() == "s",
    "non-colliding parent should stay, got: " & acc.dParent.ident()
  echo "  OK: field access renamed, parent untouched"

# ═══════════════════════════════════════════════════════════════════════
# 4. negative control: non-colliding names untouched
# ═══════════════════════════════════════════════════════════════════════
static:
  let intTy = GpuType(kind: gtInt32)
  let sym = newSymbol("count", iSym = "count_var", typ = intTy)
  let param = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: intTy,
                       addressSpace: asRMEM, passByRef: false)
  var procNode = GpuAst(kind: gpuProc,
                        pName: GpuAst(kind: gpuIdent, symbol: newSymbol("fn")),
                        pParams: @[param], pBody: GpuAst(kind: gpuBlock))
  var ctx = GpuContext()
  ctx.fnTab[procNode.pName] = procNode

  registerRenameReserved.runPassPrepared(ctx)

  doAssert procNode.pParams[0].ident.ident() == "count",
    "non-colliding param must stay untouched, got: " & procNode.pParams[0].ident.ident()
  echo "  OK: non-colliding names untouched"


# ═══════════════════════════════════════════════════════════════════════
# 2.c fn name renamed in the decl and at call sites
# ═══════════════════════════════════════════════════════════════════════
static:
  let intTy = GpuType(kind: gtInt32)
  # callee: proc layout_fn() with body `discard`
  var callee = GpuAst(kind: gpuProc,
                      pName: GpuAst(kind: gpuIdent, symbol: newSymbol("buffer", iSym = "fn_buffer")),
                      pBody: GpuAst(kind: gpuBlock))
  # caller body: `buffer()` as a gpuCall whose cName carries the same iSym
  let call = GpuAst(kind: gpuCall,
                    cName: GpuAst(kind: gpuIdent, symbol: newSymbol("buffer", iSym = "fn_buffer")))
  var callerBody = GpuAst(kind: gpuBlock)
  callerBody.statements.add call
  var caller = GpuAst(kind: gpuProc,
                      pName: GpuAst(kind: gpuIdent, symbol: newSymbol("caller", iSym = "fn_caller")),
                      pBody: callerBody)
  var ctx = GpuContext()
  ctx.allFnTab[callee.pName] = callee
  ctx.allFnTab[caller.pName] = caller
  ctx.fnTab[callee.pName] = callee
  ctx.fnTab[caller.pName] = caller

  registerRenameReserved.runPassPrepared(ctx)

  doAssert callee.pName.ident() == "buffer_nim",
    "reserved fn name should rename, got: " & callee.pName.ident()
  doAssert caller.pBody.statements[0].cName.ident() == "buffer_nim",
    "call sites of a renamed fn should follow, got: " & caller.pBody.statements[0].cName.ident()
  doAssert caller.pName.ident() == "caller",
    "non-colliding fn name must stay, got: " & caller.pName.ident()
  echo "  OK: fn name renamed in decl and at call sites"

# ═══════════════════════════════════════════════════════════════════════
# 5. local declarations renamed: var, for-loop variable, body refs follow
# ═══════════════════════════════════════════════════════════════════════
static:
  let intTy = GpuType(kind: gtInt32)
  # body: `var layout = 3`, `for buffer in 0 ..< 3`, a ref to the local sharing its iSym
  let varName = GpuAst(kind: gpuIdent, symbol: newSymbol("layout", iSym = "local_layout", typ = intTy))
  let varDecl = GpuAst(kind: gpuVar, vName: varName, vType: intTy,
                       vInit: GpuAst(kind: gpuLit, lValue: "3", lType: intTy),
                       vMutable: true, addressSpace: asRMEM)
  let forVar = GpuAst(kind: gpuIdent, symbol: newSymbol("buffer", iSym = "local_forvar", typ = intTy))
  let forNode = GpuAst(kind: gpuFor, fVar: forVar,
                       fStart: GpuAst(kind: gpuLit, lValue: "0", lType: intTy),
                       fEnd: GpuAst(kind: gpuLit, lValue: "3", lType: intTy),
                       fStep: GpuAst(kind: gpuLit, lValue: "1", lType: intTy),
                       fBody: GpuAst(kind: gpuBlock), fRangeKind: rkExclusive)
  # body ref to the `layout` local: fresh ident node sharing the iSym
  let refNode = GpuAst(kind: gpuIdent, symbol: newSymbol("layout", iSym = "local_layout", typ = intTy))
  # hazard negative: a non-colliding local must stay untouched
  let plainVar = GpuAst(kind: gpuVar,
                        vName: GpuAst(kind: gpuIdent, symbol: newSymbol("count", iSym = "local_count", typ = intTy)),
                        vType: intTy, vInit: GpuAst(kind: gpuDiscard),
                        vMutable: true, addressSpace: asRMEM)
  var body = GpuAst(kind: gpuBlock)
  body.statements.add @[varDecl, forNode, refNode, plainVar]
  var procNode = GpuAst(kind: gpuProc,
                        pName: GpuAst(kind: gpuIdent, symbol: newSymbol("fn")),
                        pBody: body)
  var ctx = GpuContext()
  ctx.fnTab[procNode.pName] = procNode

  registerRenameReserved.runPassPrepared(ctx)

  doAssert procNode.pBody.statements[0].vName.ident() == "layout_nim",
    "reserved local var name should rename, got: " & procNode.pBody.statements[0].vName.ident()
  doAssert procNode.pBody.statements[1].fVar.ident() == "buffer_nim",
    "reserved for-loop variable should rename, got: " & procNode.pBody.statements[1].fVar.ident()
  doAssert procNode.pBody.statements[2].ident() == "layout_nim",
    "body refs of a renamed local should follow, got: " & procNode.pBody.statements[2].ident()
  doAssert procNode.pBody.statements[3].vName.ident() == "count",
    "non-colliding local must stay untouched, got: " & procNode.pBody.statements[3].vName.ident()
  echo "  OK: local var, for-loop variable and their refs renamed, non-colliding local untouched"

echo "  OK - t_ir_renameReservedIdentifiers"
