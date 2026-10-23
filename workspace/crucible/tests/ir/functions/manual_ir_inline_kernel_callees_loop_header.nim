## inlineKernelCallees pass: a device fn call in a for-loop header refuses.
##
## Test that the inline pass rejects a kernel whose loop bound calls a device function, the call cannot hoist out of the loop scope.
##
## Expected compile error:
##   HLSL: device function call in a for-loop header cannot inline, the call would hoist out of the loop scope
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/ir/functions/manual_ir_inline_kernel_callees_loop_header.nim

import std/[tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_lowering_hlsl {.all.}
import workspace/crucible/tests/ir/ir_testutils
import workspace/crucible/src/codegen/passes/pass_registry

proc lit(v: string): GpuAst =
  GpuAst(kind: gpuLit, lValue: v, lType: GpuType(kind: gtUint32))

proc callee(): GpuAst =
  GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("limit", iSym = "limit")),
    pRetType: GpuType(kind: gtUint32),
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuReturn, rValue: lit("4"))]))

proc kernel(calleeName: string): GpuAst =
  ## `for i in 0 ..< limit(): output[i] = 0`
  GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k")),
    pRetType: GpuType(kind: gtVoid),
    pAttributes: {attGlobal},
    pParams: @[GpuParam(
      ident: GpuAst(kind: gpuIdent, symbol: newSymbol("output", iSym = "output",
        typ = initGpuPtrType(GpuType(kind: gtUncheckedArray, uaTo: GpuType(kind: gtUint32)),
          implicitPtr = false), symKind = gsGlobalKernelParam)),
      typ: initGpuPtrType(GpuType(kind: gtUncheckedArray, uaTo: GpuType(kind: gtUint32)), implicitPtr = false))],
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuFor,
        fVar: GpuAst(kind: gpuIdent, symbol: newSymbol("i", iSym = "i", typ = GpuType(kind: gtUint32))),
        fStart: lit("0"),
        fEnd: GpuAst(kind: gpuCall,
          cName: GpuAst(kind: gpuIdent, symbol: newSymbol(calleeName, iSym = calleeName)),
          cArgs: @[]),
        fStep: lit("1"),
        fRangeKind: rkExclusive,
        fBody: GpuAst(kind: gpuBlock))]))

static:
  var ctx = GpuContext()
  let fn = callee()
  let k = kernel("limit")
  ctx.allFnTab[fn.pName] = fn
  ctx.fnTab[fn.pName] = fn
  ctx.fnTab[k.pName] = k
  registerInlineKernelCallees.runPassPrepared(ctx)
