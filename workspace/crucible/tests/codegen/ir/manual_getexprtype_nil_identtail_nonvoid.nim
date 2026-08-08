## Manual test: a gpuIdent block-expr tail with nil symbol.typ in a
## VALUE-RETURNING proc must STILL raise the read-site error — regression for
## the silent fnRetType absorption (SLOP-009).
##
## The void-proc variant (manual_getexprtype_nil_identtail.nim) could not
## discriminate this defect: with pRetType = gtVoid the blitExprSlot ladder's
## `t = fnRetType` rung is also void, so the "blit temp" error fired anyway.
## Pre-fix, a nil-typed ident tail in a value-returning proc (pRetType =
## gtInt32) silently compiled with the proc's return type substituted for the
## ident's unknown type — the exact same silent wrong-type class that SLOP-001
## closed for gpuBinOp. The getExprType gpuIdent read site now verifies the
## symbol type and raises the error regardless of fnRetType.
##
## The block-expr is used as a STATEMENT (discarded), so blitType is void and
## the ladder must consult getExprType — the exact shape of a discarded
## block-expr in a value-returning proc.
##
## Run (expected FAIL — negative test):
##   cd tattletale
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/ir/manual_getexprtype_nil_identtail_nonvoid.nim
## Expect: "gpuIdent with nil or void symbol type: Cannot determine type for blit temp in " &
##         "block expression (nil/void symbol.typ on a gpuIdent is a defect — idents reaching getExprType must carry a type)"

import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types

# Hand-built IR: a value-returning proc (pRetType = gtInt32) whose body is a
# discarded expression block tailed by a gpuIdent with nil symbol.typ.
# getExprType cannot type the tail and must raise the read-site error at the
# gpuIdent case — NOT silently substitute fnRetType. The codegen runs at
# compile time via `static:`, so `nim c` (compile only) reports it.
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
  let fn = GpuAst(
    kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("negTestIdentTailNonVoid", iSym = "negTestIdentTailNonVoid", symKind = gsProc)),
    pRetType: GpuType(kind: gtInt32),   # NON-void — pre-fix this absorbed the nil via fnRetType
    pParams: @[],
    pBody: GpuAst(kind: gpuBlock, isExpr: false, statements: @[blk]))
  var gen = GpuGenericsInfo(procs: @[fn])
  discard codegen(gen, GpuAst(kind: gpuBlock), backend = bkCuda)

echo "UNREACHABLE — the nil-typed ident tail in a value-returning proc must have raised the read-site error"
