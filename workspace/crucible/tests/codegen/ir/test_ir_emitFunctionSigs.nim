## Phase 5: emitFunctionSignatures pass test
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_emitFunctionSigs.nim

import std / [tables, sequtils, strutils]
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/passes_preprocessing

# ═══════════════════════════════════════════════════════════════════════════
# 1. emitFunctionSignatures adds sigString to fnTable entries
# ═══════════════════════════════════════════════════════════════════════════
block:
  let int32 = GpuType(kind: gtInt32)
  let void = GpuType(kind: gtVoid)
  let fnSym = newSymbol("foo", iSym = "foo_h1", symKind = gsProc)
  let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
  let p1Ident = GpuAst(kind: gpuIdent, symbol: newSymbol("a", iSym = "a_h1", typ = int32))
  let p2Ident = GpuAst(kind: gpuIdent, symbol: newSymbol("b", iSym = "b_h1", typ = int32))
  let fnBody = GpuAst(kind: gpuBlock, statements: @[])
  let fn = GpuAst(kind: gpuProc, pName: fnIdent, pRetType: void,
                 pParams: @[GpuParam(ident: p1Ident, typ: int32),
                            GpuParam(ident: p2Ident, typ: int32)],
                 pBody: fnBody)

  var ctx = GpuContext()
  ctx.fnTable["foo_h1"] = FnTableEntry(ident: fnIdent, body: fn, kind: {fkDefined}, namePolicy: npClean)

  emitFunctionSignaturesImpl(ctx)

  doAssert ctx.fnTable["foo_h1"].sigString.len > 0,
    "sigString should be set after emitFunctionSignatures"
  echo "  OK — sigString set on FnTableEntry after emitFunctionSignatures"

# ═══════════════════════════════════════════════════════════════════════════
# 2. sigString contains function name
# ═══════════════════════════════════════════════════════════════════════════
block:
  let void = GpuType(kind: gtVoid)
  let fnSym = newSymbol("bar", iSym = "bar_h2", symKind = gsProc)
  let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
  let fnBody = GpuAst(kind: gpuBlock, statements: @[])
  let fn = GpuAst(kind: gpuProc, pName: fnIdent, pRetType: void,
                 pParams: @[], pBody: fnBody)

  var ctx = GpuContext()
  ctx.fnTable["bar_h2"] = FnTableEntry(ident: fnIdent, body: fn, kind: {fkDefined}, namePolicy: npClean)

  emitFunctionSignaturesImpl(ctx)

  let sig = ctx.fnTable["bar_h2"].sigString
  doAssert sig.find("bar") >= 0, "sigString should contain function name 'bar', got: " & sig
  echo "  OK — sigString contains function name"

# ═══════════════════════════════════════════════════════════════════════════
# 3. sigComment comment node inserted in proc body
# ═══════════════════════════════════════════════════════════════════════════
block:
  let void = GpuType(kind: gtVoid)
  let fnSym = newSymbol("baz", iSym = "baz_h3", symKind = gsProc)
  let fnIdent = GpuAst(kind: gpuIdent, symbol: fnSym)
  let fnBody = GpuAst(kind: gpuBlock, statements: @[])
  let fn = GpuAst(kind: gpuProc, pName: fnIdent, pRetType: void,
                 pParams: @[], pBody: fnBody)

  var ctx = GpuContext()
  ctx.fnTable["baz_h3"] = FnTableEntry(ident: fnIdent, body: fn, kind: {fkDefined}, namePolicy: npClean)

  emitFunctionSignaturesImpl(ctx)

  doAssert fnBody.statements.len > 0, "Body should have at least 1 statement (the sig comment)"
  doAssert fnBody.statements[0].kind == gpuComment, "First statement should be a comment"
  doAssert "sig:" in fnBody.statements[0].comment, "Comment should contain 'sig:' prefix"
  echo "  OK — sig comment node inserted in proc body"

echo ""
echo "  All emitFunctionSignatures tests passed."
