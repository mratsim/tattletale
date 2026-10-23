## inlineKernelCallees pass: a device fn call in a ternary branch refuses.
##
## Test that the inline pass rejects a kernel whose ternary branch calls a device function, the branch evaluates conditionally.
##
## Expected compile error:
##   HLSL: device function call in a ternary branch cannot inline, the branch evaluates conditionally
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/ir/functions/manual_ir_inline_kernel_callees_ternary_refused.nim

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
  ## `output[0] = if cond: limit() else: 2`
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
      GpuAst(kind: gpuAssign,
        aLeft: GpuAst(kind: gpuIndex,
          iArr: GpuAst(kind: gpuIdent, symbol: newSymbol("output", iSym = "output")),
          iIndex: lit("0")),
        aRight: GpuAst(kind: gpuTernary,
          tCond: lit("1"),
          tThen: GpuAst(kind: gpuCall,
            cName: GpuAst(kind: gpuIdent, symbol: newSymbol(calleeName, iSym = calleeName)),
            cArgs: @[]),
          tElse: lit("2")))]))

static:
  var ctx = GpuContext()
  let fn = callee()
  let k = kernel("limit")
  ctx.allFnTab[fn.pName] = fn
  ctx.fnTab[fn.pName] = fn
  ctx.fnTab[k.pName] = k
  registerInlineKernelCallees.runPassPrepared(ctx)
