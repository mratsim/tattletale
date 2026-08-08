## Manual test: nil-bType gpuBinOp as block-expr tail must keep raising the
## "blit temp" error — no silent degradation, no nil fallback.
##
## The compiler has NO nil fallback: resolveType hard-errors and
## resolveProcReturnType returns gtVoid or resolveType — never nil. A nil
## bType on a block-tail gpuBinOp is a defect that must surface at the
## getExprType gpuBinOp read site ("Cannot determine type for blit temp in
## block expression") — it must NOT silently fall through to the blitExprSlot
## ladder's fnRetType rung.
##
## This variant uses a VOID-returning proc. A non-void variant lives in
## manual_getexprtype_nil_btype_nonvoid.nim: pre-fix, the fnRetType rung of
## the ladder silently absorbed the nil bType there; both variants must now
## raise the same error.
##
## Run (expected FAIL — negative test):
##   cd tattletale
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/ir/manual_getexprtype_nil_btype.nim
## Expect: "gpuBinOp with nil or void bType: Cannot determine type for blit temp in block expression (nil/void bType on a gpuBinOp is a defect — all construction sites must populate bType)"

import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types

# Hand-built IR replicating the sgemm crash shape: a proc whose body is an
# expression block tailed by a primitive gpuBinOp. Here the binop carries
# bType: nil (a defect), so getExprType cannot type the tail and must raise
# the "blit temp" error at the gpuBinOp read site. The codegen runs at
# compile time via `static:` — the blitBlockExprs pass raises during macro-
# equivalent evaluation, so `nim c` (compile only) reports the error.
static:
  let binOp = GpuAst(
    kind: gpuBinOp,
    bType: nil,
    bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("*", iSym = "*")),
    bLeft: GpuAst(kind: gpuIdent,
                  symbol: newSymbol("it", iSym = "it", typ = GpuType(kind: gtInt32))),
    bRight: GpuAst(kind: gpuLit, lValue: "1", lType: GpuType(kind: gtInt32)))

  let blk = GpuAst(
    kind: gpuBlock,
    isExpr: true,
    statements: @[
      GpuAst(kind: gpuLit, lValue: "0", lType: GpuType(kind: gtInt32)),
      binOp
    ])

  let resId = GpuAst(kind: gpuIdent,
                     symbol: newSymbol("result", iSym = "result", typ = GpuType(kind: gtVoid)))
  let assign = GpuAst(kind: gpuAssign, aLeft: resId, aRight: blk)

  let fn = GpuAst(
    kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("negTest", iSym = "negTest", symKind = gsProc)),
    pRetType: GpuType(kind: gtVoid),
    pParams: @[],
    pBody: GpuAst(kind: gpuBlock, isExpr: false, statements: @[assign]))

  var gen = GpuGenericsInfo(procs: @[fn])
  discard codegen(gen, GpuAst(kind: gpuBlock), backend = bkCuda)

echo "UNREACHABLE — the nil-bType block tail must have raised the blit-temp error"
