## Phase 6: injectAddressOf pass tests
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_injectAddressOf.nim

import std / [tables, sequtils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. injectAddressOfImpl wraps ptr globals in gpuAddr
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let ptrTyp = GpuType(kind: gtPtr, to: int32)
  let sym = newSymbol("x", iSym = "x_h1", typ = ptrTyp)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  var ctx = GpuContext()
  ctx.globals["x_h1"] = GpuParam(ident: ident, typ: ptrTyp, addressSpace: asDevice)
  var n = ident
  ctx.injectAddressOfImpl(n)
  doAssert n.kind == gpuAddr, "Ptr global should become gpuAddr, got: " & $n.kind
  doAssert n.aOf.kind == gpuIdent, "gpuAddr target should be the ident"
  doAssert n.aOf.ident() == "x", "gpuAddr target ident should be 'x'"
  echo "  OK — ptr global ident wrapped in gpuAddr"

# ═══════════════════════════════════════════════════════════════════════════
# 2. injectAddressOfImpl wraps bool globals in gpuConv
# ═══════════════════════════════════════════════════════════════════════════
block:
  let boolTyp = GpuType(kind: gtBool)
  let sym = newSymbol("b", iSym = "b_h2", typ = boolTyp)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  var ctx = GpuContext()
  ctx.globals["b_h2"] = GpuParam(ident: ident, typ: boolTyp, addressSpace: asDevice)
  var n = ident
  ctx.injectAddressOfImpl(n)
  doAssert n.kind == gpuConv, "Bool global should become gpuConv, got: " & $n.kind
  doAssert n.convTo.kind == gtBool, "Conv target should be bool"
  doAssert n.convExpr.kind == gpuIdent, "Conv expr should be the ident"
  echo "  OK — bool global ident wrapped in gpuConv"

# ═══════════════════════════════════════════════════════════════════════════
# 3. injectAddressOfImpl collapses deref of ptr global
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let ptrTyp = GpuType(kind: gtPtr, to: int32)
  let sym = newSymbol("p", iSym = "p_h3", typ = ptrTyp)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  var deref = GpuAst(kind: gpuDeref, dOf: ident)
  var ctx = GpuContext()
  ctx.globals["p_h3"] = GpuParam(ident: ident, typ: ptrTyp, addressSpace: asDevice)
  var n = deref
  ctx.injectAddressOfImpl(n)
  doAssert n.kind == gpuIdent, "Deref of ptr global should collapse to ident, got: " & $n.kind
  echo "  OK — deref of ptr global collapses to ident"

# ═══════════════════════════════════════════════════════════════════════════
# 4. pullConstantPragmaVarsImpl extracts {.const_mem.} vars
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let sym = newSymbol("buf", iSym = "buf_h4", typ = int32)
  let varIdent = GpuAst(kind: gpuIdent, symbol: sym)
  let varNode = GpuAst(kind: gpuVar, vName: varIdent, vType: int32,
                       vInit: GpuAst(kind: gpuDiscard), vMutable: true,
                       addressSpace: asConstant)
  var blk = GpuAst(kind: gpuBlock)
  blk.statements.add varNode
  var ctx = GpuContext()
  ctx.pullConstantPragmaVarsImpl(blk)
  doAssert blk.statements.len == 0, "Block should be empty after extraction"
  doAssert "buf_h4" in ctx.globals, "Extracted var should be in globals"
  echo "  OK — {.const_mem.} var extracted from block to globals"

# ═══════════════════════════════════════════════════════════════════════════
# 5. removeStructPointerFieldsImpl removes ptr fields from structs
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let ptrTyp = GpuType(kind: gtPtr, to: int32)
  let structTyp = GpuType(kind: gtObject, name: "Foo")
  var typeDef = GpuAst(kind: gpuTypeDef, tTyp: structTyp)
  typeDef.tFields = @[
    GpuTypeField(name: "a", typ: int32),
    GpuTypeField(name: "p", typ: ptrTyp),
    GpuTypeField(name: "b", typ: int32),
  ]
  var blk = GpuAst(kind: gpuBlock)
  blk.statements.add typeDef
  removeStructPointerFieldsImpl(blk)
  doAssert typeDef.tFields.len == 2, "Should have 2 fields after removal, got: " & $typeDef.tFields.len
  doAssert typeDef.tFields[0].name == "a", "First field should be 'a'"
  doAssert typeDef.tFields[1].name == "b", "Second field should be 'b'"
  echo "  OK — ptr fields removed from struct definition"

echo ""
echo "  All injectAddressOf tests passed."
