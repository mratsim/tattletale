## injectAddressOf plus pullConstantPragmaVars and removeStructPointerFields
##
## Test that injectAddressOf wraps device-global idents and collapses ptr-global
## derefs to the bare ident, the const-mem and ptr-field passes holding beside it.
##
## - Ptr globals wrap in gpuAddr, bool globals stay bare (patchBoolToI32 wraps their loads)
## - `{.const_mem.}` vars lift from the block body into ctx.globals
## - Struct type defs drop pointer fields while plain fields keep their order
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_injectAddressOf.nim

import std / [tables, sequtils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering_wgsl {.all.}
import workspace/crucible/src/codegen/passes/pass_registry

proc runOnBlock(regProc: proc(reg: var PassRegistry) {.nimcall.}, blk: var GpuAst, idx: int): GpuContext =
  ## Runs one registered pass with `blk` at globalBlocks[idx].
  ## Returns the ctx for assertions on pass side effects.
  result = GpuContext()
  for i in 0 ..< idx:
    result.globalBlocks.add GpuAst(kind: gpuBlock)
  result.globalBlocks.add blk
  var reg = PassRegistry.new()
  regProc(reg)
  result.runPasses(reg)

proc runInject(ctx: var GpuContext, n: var GpuAst) =
  ## Runs the registered injectAddressOf pass over `n` as a global fn body.
  var fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol("f", iSym = "f")),
                  pRetType: GpuType(kind: gtVoid),
                  pBody: GpuAst(kind: gpuBlock, statements: @[n]))
  fn.pAttributes.incl attGlobal
  ctx.fnTab[fn.pName] = fn
  var reg = PassRegistry.new()
  reg.registerAddressOfInjection()
  ctx.runPasses(reg)
  n = fn.pBody.statements[0]

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
  runInject(ctx, n)
  doAssert n.kind == gpuAddr, "Ptr global should become gpuAddr, got: " & $n.kind
  doAssert n.aOf.kind == gpuIdent, "gpuAddr target should be the ident"
  doAssert n.aOf.ident() == "x", "gpuAddr target ident should be 'x'"
  echo "  OK — ptr global ident wrapped in gpuAddr"

# ═══════════════════════════════════════════════════════════════════════════
# 2. injectAddressOfImpl leaves bool globals alone
# ═══════════════════════════════════════════════════════════════════════════
block:
  let boolTyp = GpuType(kind: gtBool)
  let sym = newSymbol("b", iSym = "b_h2", typ = boolTyp)
  let ident = GpuAst(kind: gpuIdent, symbol: sym)
  var ctx = GpuContext()
  ctx.globals["b_h2"] = GpuParam(ident: ident, typ: boolTyp, addressSpace: asDevice)
  var n = ident
  runInject(ctx, n)
  doAssert n.kind == gpuIdent, "Bool global must stay bare, got: " & $n.kind
  echo "  OK — bool global ident left to patchBoolToI32"

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
  runInject(ctx, n)
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
  var ctx = runOnBlock(registerPullConstantPragmaVars, blk, 0)
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
  discard runOnBlock(registerRemoveStructPointerFields, blk, 1)
  doAssert typeDef.tFields.len == 2, "Should have 2 fields after removal, got: " & $typeDef.tFields.len
  doAssert typeDef.tFields[0].name == "a", "First field should be 'a'"
  doAssert typeDef.tFields[1].name == "b", "Second field should be 'b'"
  echo "  OK — ptr fields removed from struct definition"

echo ""
echo "  All injectAddressOf tests passed."
