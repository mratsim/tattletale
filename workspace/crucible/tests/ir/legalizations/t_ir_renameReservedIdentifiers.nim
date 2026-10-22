## Run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/ir/legalizations/t_ir_renameReservedIdentifiers.nim (from tattletale)

import std/[sets, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering

const Reserved = ["layout", "buffer"].toHashSet()

# ═══════════════════════════════════════════════════════════════════════════
# 1. fn param renamed in the decl and in body refs
# ═══════════════════════════════════════════════════════════════════════════
block:
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

  ctx.renameReservedIdentifiers(Reserved)

  doAssert procNode.pParams[0].ident.ident() == "layout_nim",
    "reserved param name should rename, got: " & procNode.pParams[0].ident.ident()
  doAssert procNode.pBody.statements[0].aLeft.ident() == "layout_nim",
    "body refs of a renamed param should follow, got: " & procNode.pBody.statements[0].aLeft.ident()
  echo "  OK: param decl and body refs renamed"

# ═══════════════════════════════════════════════════════════════════════════
# 2. struct field renamed in the decl, including block-wrapped type defs
# ═══════════════════════════════════════════════════════════════════════════
block:
  var field = GpuTypeField(name: "buffer", typ: GpuType(kind: gtInt32))
  var typeDef = GpuAst(kind: gpuTypeDef,
                       tTyp: GpuType(kind: gtObject, name: "S"),
                       tFields: @[field])
  # WGSL wraps all type defs in one gpuBlock, GLSL puts them directly in globalBlocks
  var typBlock = GpuAst(kind: gpuBlock)
  typBlock.statements.add typeDef
  var ctx = GpuContext()
  ctx.globalBlocks.add typBlock

  ctx.renameReservedIdentifiers(Reserved)

  doAssert ctx.globalBlocks[0].statements[0].tFields[0].name == "buffer_nim",
    "reserved field name in a block-wrapped type def should rename, got: " &
      ctx.globalBlocks[0].statements[0].tFields[0].name
  echo "  OK: struct field decl renamed in block-wrapped type def"

# ═══════════════════════════════════════════════════════════════════════════
# 2.b globals table (WGSL lifts kernel params there) renamed
# ═══════════════════════════════════════════════════════════════════════════
block:
  let intTy = GpuType(kind: gtInt32)
  let sym = newSymbol("layout", iSym = "layout_global", typ = intTy)
  let g = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: intTy,
                   addressSpace: asRMEM, passByRef: false)
  var ctx = GpuContext()
  ctx.globals["layout_global"] = g

  ctx.renameReservedIdentifiers(Reserved)

  doAssert ctx.globals["layout_global"].ident.ident() == "layout_nim",
    "global symbol with reserved name should rename, got: " &
      ctx.globals["layout_global"].ident.ident()
  echo "  OK: globals table renamed"

# ═══════════════════════════════════════════════════════════════════════════
# 3. field access renamed
# ═══════════════════════════════════════════════════════════════════════════
block:
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

  ctx.renameReservedIdentifiers(Reserved)

  let acc = procNode.pBody.statements[0]
  doAssert acc.dField.ident() == "layout_nim",
    "reserved field access should rename, got: " & acc.dField.ident()
  doAssert acc.dParent.ident() == "s",
    "non-colliding parent should stay, got: " & acc.dParent.ident()
  echo "  OK: field access renamed, parent untouched"

# ═══════════════════════════════════════════════════════════════════════════
# 4. negative control: non-colliding names untouched
# ═══════════════════════════════════════════════════════════════════════════
block:
  let intTy = GpuType(kind: gtInt32)
  let sym = newSymbol("count", iSym = "count_var", typ = intTy)
  let param = GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: intTy,
                       addressSpace: asRMEM, passByRef: false)
  var procNode = GpuAst(kind: gpuProc,
                        pName: GpuAst(kind: gpuIdent, symbol: newSymbol("fn")),
                        pParams: @[param], pBody: GpuAst(kind: gpuBlock))
  var ctx = GpuContext()
  ctx.fnTab[procNode.pName] = procNode

  ctx.renameReservedIdentifiers(Reserved)

  doAssert procNode.pParams[0].ident.ident() == "count",
    "non-colliding param must stay untouched, got: " & procNode.pParams[0].ident.ident()
  echo "  OK: non-colliding names untouched"


# ═══════════════════════════════════════════════════════════════════════════
# 2.c fn name renamed in the decl and at call sites
# ═══════════════════════════════════════════════════════════════════════════
block:
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

  ctx.renameReservedIdentifiers(Reserved)

  doAssert callee.pName.ident() == "buffer_nim",
    "reserved fn name should rename, got: " & callee.pName.ident()
  doAssert caller.pBody.statements[0].cName.ident() == "buffer_nim",
    "call sites of a renamed fn should follow, got: " & caller.pBody.statements[0].cName.ident()
  doAssert caller.pName.ident() == "caller",
    "non-colliding fn name must stay, got: " & caller.pName.ident()
  echo "  OK: fn name renamed in decl and at call sites"

echo "  OK - t_ir_renameReservedIdentifiers"
