## Manual test: a gpuIdent block-expr tail whose symbol has a nil typ must
## raise a compile error at the getExprType read site — the ident analogue of
## the gpuBinOp bType check (SLOP-009). Pre-fix, the nil typ fell silently to
## the blitExprSlot ladder's fnRetType rung in value-returning procs; the
## enriched blit-temp error (blitExprSlot, "Cannot determine type for blit
## temp in block expression") only fired when fnRetType was also void.
##
## This variant uses a VOID-returning proc; the non-void variant lives in
## manual_getexprtype_nil_identtail_nonvoid.nim and proves the SAME error is
## raised when pRetType is gtInt32 (the silent-absorption shape). The error is
## a plain compile error — never a case-object FieldDefect.
##
## Run (expected FAIL — negative test):
##   cd tattletale
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/ir/manual_getexprtype_nil_identtail.nim
## Expect: "gpuIdent with nil or void symbol type: Cannot determine type for blit temp in " &
##         "block expression (nil/void symbol.typ on a gpuIdent is a defect — idents reaching getExprType must carry a type)"

import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types

static:
  let tailIdent = GpuAst(kind: gpuIdent,
                         symbol: newSymbol("undef", iSym = "undef", typ = nil))
  let blk = GpuAst(
    kind: gpuBlock,
    isExpr: true,
    statements: @[
      GpuAst(kind: gpuLit, lValue: "0", lType: GpuType(kind: gtInt32)),
      tailIdent
    ])
  let resId = GpuAst(kind: gpuIdent,
                     symbol: newSymbol("result", iSym = "result", typ = GpuType(kind: gtVoid)))
  let assign = GpuAst(kind: gpuAssign, aLeft: resId, aRight: blk)
  let fn = GpuAst(
    kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("negTestIdentTail", iSym = "negTestIdentTail", symKind = gsProc)),
    pRetType: GpuType(kind: gtVoid),
    pParams: @[],
    pBody: GpuAst(kind: gpuBlock, isExpr: false, statements: @[assign]))
  var gen = GpuGenericsInfo(procs: @[fn])
  discard codegen(gen, GpuAst(kind: gpuBlock), backend = bkCuda)

echo "UNREACHABLE — the nil-typed ident tail must have raised the blit-temp error"
