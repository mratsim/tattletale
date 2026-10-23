## lowerArraysAndViews decomposes local views without span signatures
##
## Test that a kernel creating a local `let v = toOpenArray(p, a, b)`
## gets the binding decomposed into the (ptr, len) declaration pair
## even with an empty spanSigMap (no span parameter anywhere).
##
## An empty spanSigMap must not skip the local-view rewriting:
## the raw `toOpenArray` call would otherwise survive to codegen.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_localViewNoSpanSig.nim

import std / [tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_low_views
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

proc registerLower(reg: var PassRegistry) {.nimcall.} =
  reg.registerLowerArraysAndViews(decayArrays = true)

proc ident(sym: Symbol): GpuAst = GpuAst(kind: gpuIdent, symbol: sym)

proc lit(t: GpuType, v: string): GpuAst = GpuAst(kind: gpuLit, lType: t, lValue: v)

proc toOpenArrayCall(data: Symbol): GpuAst =
  GpuAst(kind: gpuCall,
         cName: GpuAst(kind: gpuIdent, symbol: newSymbol("toOpenArray", iSym = "toOpenArray")),
         cArgs: @[ident(data), lit(initGpuType(gtInt32), "0"), lit(initGpuType(gtInt32), "3")])

static:
  var ctx = GpuContext()
  let bufTyp = initGpuPtrType(initGpuType(gtInt32), implicitPtr = false)
  let bufSym = newSymbol("buf", iSym = "buf_h0", typ = bufTyp)
  ctx.globals["buf_h0"] = GpuParam(ident: ident(bufSym), typ: bufTyp, addressSpace: asDevice)
  let spanTyp = initGpuSpanType(initGpuType(gtInt32))
  let vSym = newSymbol("v", iSym = "v", typ = spanTyp)
  let viewVar = GpuAst(kind: gpuVar,
                       vName: ident(vSym),
                       vType: spanTyp,
                       vInit: toOpenArrayCall(bufSym))
  let use = GpuAst(kind: gpuIndex,
                   iArr: ident(vSym),
                   iIndex: lit(initGpuType(gtInt32), "0"))
  let k = GpuAst(kind: gpuProc,
                 pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k")),
                 pRetType: GpuType(kind: gtVoid),
                 pBody: GpuAst(kind: gpuBlock, statements: @[viewVar, use]))
  ctx.allFnTab[k.pName] = k
  registerLower.runPassPrepared(ctx)

  let stmts = ctx.allFnTab[k.pName].pBody.statements
  doAssert stmts.len == 3,
    "the view binding decomposes into a (ptr, len) pair, got " & $stmts.len & " statements"
  doAssert stmts[0].kind == gpuVar and stmts[0].vType.kind == gtPtr,
    "the view var becomes a pointer, got " & $stmts[0].vType.kind
  doAssert stmts[0].vInit.kind == gpuBinOp,
    "the init is the ptr-plus expression, got " & $stmts[0].vInit.kind
  doAssert stmts[1].kind == gpuVar and stmts[1].vName.symbol.name == "v_len",
    "the second declaration is the len binding, got " & stmts[1].vName.symbol.name
  echo "  OK — local view decomposed with empty span signatures"

echo "t_ir_localViewNoSpanSig: all static assertions passed"
