## A bare `toOpenArray(...)` statement is rejected
##
## Manual test: a `toOpenArray(...)` call used outside a span argument,
## a len call, or a view binding has no lowering.
## The pass must reject it loudly:
## a two-value expression block cannot reach codegen.
##
## Expected: compile fails at the runPassPrepared call, carrying
##   "toOpenArray result used outside a span argument"
##
## Run (must FAIL):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/manual_ir_bareToOpenArray.nim

import std / [tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/tests/ir/ir_testutils

proc registerLower(reg: var PassRegistry) {.nimcall.} =
  reg.registerLowerArraysAndViews(decayArrays = true)

static:
  var ctx = GpuContext()
  let spanTyp = initGpuSpanType(initGpuType(gtInt32))
  let k = GpuAst(kind: gpuProc,
                 pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k")),
                 pRetType: GpuType(kind: gtVoid),
                 pParams: @[GpuParam(
                   ident: GpuAst(kind: gpuIdent,
                                 symbol: newSymbol("s", iSym = "s", typ = spanTyp)),
                   typ: spanTyp, addressSpace: asRMEM, passByRef: false)],
                 pBody: GpuAst(kind: gpuBlock, statements: @[
                   GpuAst(kind: gpuCall,
                          cName: GpuAst(kind: gpuIdent,
                                        symbol: newSymbol("toOpenArray", iSym = "toOpenArray")),
                          cArgs: @[
                            GpuAst(kind: gpuIdent,
                                   symbol: newSymbol("data", iSym = "data",
                                                     typ = initGpuPtrType(initGpuType(gtInt32), implicitPtr = false))),
                            GpuAst(kind: gpuLit, lType: initGpuType(gtInt32), lValue: "0"),
                            GpuAst(kind: gpuLit, lType: initGpuType(gtInt32), lValue: "3")])]))
  ctx.allFnTab[k.pName] = k
  registerLower.runPassPrepared(ctx)
