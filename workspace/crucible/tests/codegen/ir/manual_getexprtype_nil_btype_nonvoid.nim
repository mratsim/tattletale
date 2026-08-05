## Manual test: nil-bType gpuBinOp as block-expr tail in a VALUE-RETURNING
## proc must STILL raise the "blit temp" error — regression for the silent
## fnRetType absorption.
##
## Pre-fix, a nil bType on a block-tail gpuBinOp inside a value-returning
## proc (pRetType = gtInt32) silently fell through the blitExprSlot ladder to
## the `t = fnRetType` rung: the block compiled with the proc's return type
## substituted for the binop's unknown type. The getExprType gpuBinOp read
## site now verifies the self-carried bType and raises the same "Cannot
## determine type for blit temp in block expression" error as the void
## variant (manual_getexprtype_nil_btype.nim), regardless of fnRetType.
##
## The block-expr is used as a STATEMENT (discarded), so blitType is void and
## the ladder must consult getExprType — the exact shape of a discarded
## block-expr in a value-returning proc.
##
## Run (expected FAIL — negative test):
##   cd tattletale
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/ir/manual_getexprtype_nil_btype_nonvoid.nim
## Expect: "gpuBinOp with nil or void bType: Cannot determine type for blit temp in block expression (nil/void bType on a gpuBinOp is a defect — all construction sites must populate bType)"

import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types

# Hand-built IR: a value-returning proc (pRetType = gtInt32) whose body is a
# discarded expression block tailed by a primitive gpuBinOp with bType: nil.
# getExprType cannot type the tail and must raise the "blit temp" error at
# the gpuBinOp read site — NOT silently substitute fnRetType. The codegen
# runs at compile time via `static:`, so `nim c` (compile only) reports it.
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

  let fn = GpuAst(
    kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("negTestNonVoid", iSym = "negTestNonVoid", symKind = gsProc)),
    pRetType: GpuType(kind: gtInt32),   # NON-void — pre-fix this absorbed the nil via fnRetType
    pParams: @[],
    pBody: GpuAst(kind: gpuBlock, isExpr: false, statements: @[blk]))

  var gen = GpuGenericsInfo(procs: @[fn])
  discard codegen(gen, GpuAst(kind: gpuBlock), backend = bkCuda)

echo "UNREACHABLE — the nil-bType block tail in a value-returning proc must have raised the blit-temp error"
