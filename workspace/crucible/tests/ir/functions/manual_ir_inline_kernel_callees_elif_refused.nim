## inlineKernelCallees pass: a device fn call in an elif condition refuses.
##
## Test that the inline pass rejects a kernel whose elif condition calls a device function, the condition evaluates only when reached.
##
## Expected compile error:
##   HLSL: device function call in an elif condition cannot inline, the condition evaluates only when reached
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/ir/functions/manual_ir_inline_kernel_callees_elif_refused.nim

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
  ## `if cond: discard elif limit() > 0: output[0] = 0`
  GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k")),
    pRetType: GpuType(kind: gtVoid),
    pAttributes: {attGlobal},
    pParams: @[GpuParam(
      ident: GpuAst(kind: gpuIdent, symbol: newSymbol("output", iSym = "output",
        typ = initGpuPtrType(GpuType(kind: gtUncheckedArray, uaTo: GpuType(kind: gtUint32)),
          implicitPtr = false), symKind = gsGlobalKernelParam)),
      typ: initGpuPtrType(GpuType(kind: gtUncheckedArray, uaTo: GpuType(kind: gtUint32)),
        implicitPtr = false))],
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuIf,
        ifCond: lit("1"),
        ifThen: GpuAst(kind: gpuBlock),
        ifElifs: @[(cond: GpuAst(kind: gpuCall,
                      cName: GpuAst(kind: gpuIdent, symbol: newSymbol(calleeName, iSym = calleeName)),
                      cArgs: @[]),
                    body: GpuAst(kind: gpuBlock))],
        ifElse: GpuAst(kind: gpuDiscard))]))

static:
  var ctx = GpuContext()
  let fn = callee()
  let k = kernel("limit")
  ctx.allFnTab[fn.pName] = fn
  ctx.fnTab[fn.pName] = fn
  ctx.fnTab[k.pName] = k
  registerInlineKernelCallees.runPassPrepared(ctx)
