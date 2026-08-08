## Compound-assign LHS through the full common pipeline (rewrite → legalization)
##
## Regression (Class A of the sgemm_1 port): ceramic `tv[m,n] += ...` — a
## statement-list-expression LHS wrapped in HiddenAddr — was by-value-blitted
## into a discarded temp and emitted as `((&_blit_N) += (...))`, which NVRTC
## rejects ("expression must be a modifiable lvalue") and which silently lost
## the accumulation. The compound-assign rewrite (registered in
## registerNormalizationPasses, BEFORE legalization) desugars `x += y` →
## `x = x + y` with the HiddenAddr stripped; blitBlockExprs then keeps the
## block-expr tail as the assignment LHS, so the store-back is direct.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_blitCompoundAssign.nim
##
##   cd workspace/crucible
##   nim c -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip \
##     tests/codegen/ir/test_ir_blitCompoundAssign.nim

import std/[strutils, tables]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/pass_datatypes

proc countExprBlocks(n: GpuAst): int =
  ## Count gpuBlock(isExpr: true) nodes in a subtree.
  if n == nil: return 0
  result = if n.kind == gpuBlock and n.isExpr: 1 else: 0
  for ch in n.items:
    result += countExprBlocks(ch)

proc findAssign(n: GpuAst): GpuAst =
  ## The rewritten compound-assign node in a subtree (pre-order): the
  ## gpuAssign whose RHS is a gpuBinOp. (Inner blit-assigns like
  ## `_blit_N = acc[pos]` also match gpuAssign — filter them out.)
  if n == nil: return nil
  if n.kind == gpuAssign and n.aRight.kind == gpuBinOp: return n
  for ch in n.items:
    result = findAssign(ch)
    if not result.isNil: return

# ── 1. Full common pipeline on the ceramic shape (hand-built IR) ──
# The LHS is gpuAddr(gpuBlock(isExpr, [pos decl, gpuIndex lvalue])) exactly as
# the frontend produces for `tv[m,n] += rhs`. After the rewrite + legalization
# the assignment LHS must be the real gpuIndex lvalue (direct store-back) —
# NOT a by-value `_blit_` temp and NOT an address-of.
static:
  let int32t = GpuType(kind: gtInt32)
  let posSym = newSymbol("pos", iSym = "pos", typ = int32t, symKind = gsLocal)
  let posVar = GpuAst(kind: gpuVar, vName: GpuAst(kind: gpuIdent, symbol: posSym),
                      vType: int32t, vInit: GpuAst(kind: gpuLit, lValue: "0", lType: int32t))
  let accSym = newSymbol("acc", iSym = "acc",
                         typ = GpuType(kind: gtArray, aTyp: int32t, aLen: 4),
                         symKind = gsLocal)
  let tail = GpuAst(kind: gpuIndex,
                    iArr: GpuAst(kind: gpuIdent, symbol: accSym),
                    iIndex: GpuAst(kind: gpuIdent, symbol: posSym))
  let lhsBlock = GpuAst(kind: gpuBlock, isExpr: true, statements: @[posVar, tail])
  let binOp = GpuAst(kind: gpuBinOp, bType: int32t,
                     bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+=", iSym = "+=")),
                     bLeft: GpuAst(kind: gpuAddr, aOf: lhsBlock),
                     bRight: GpuAst(kind: gpuLit, lValue: "1", lType: int32t))
  let body = GpuAst(kind: gpuBlock, statements: @[binOp])
  let fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol("compoundKernel", iSym = "compoundKernel", symKind = gsProc)),
                  pRetType: GpuType(kind: gtVoid),
                  pParams: @[],
                  pBody: body)
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  ctx.allFnTab[fn.pName] = fn
  runPasses(ctx, reg)
  doAssert countExprBlocks(fn.pBody) == 0,
    "compound-assign block LHS must be fully blitted, leftover: " & $countExprBlocks(fn.pBody)
  let assign = findAssign(fn.pBody)
  doAssert not assign.isNil, "expected the rewritten assignment in the body"
  doAssert assign.aLeft.kind == gpuIndex,
    "assignment LHS must stay the real lvalue (gpuIndex), got: " & $assign.aLeft.kind
  doAssert assign.aLeft.iArr.kind == gpuIdent and assign.aLeft.iArr.ident() == "acc",
    "LHS lvalue must be acc[...], got: " & $assign.aLeft.iArr.kind
  echo "  OK — compound-assign LHS: block tail kept as lvalue (direct store-back)"

# ── 2. Emission anchor through the real `cuda:` macro ──
# A plain indexed compound assignment must emit a direct read-modify-write:
# no `(&`, no leftover `+=`, no by-value temp on the LHS.
block:
  const outCuda = cuda:
    proc compoundKernel(C: ptr UncheckedArray[int32]) {.global.} =
      for i in 0 ..< 4:
        C[i] += 1
  doAssert not outCuda.contains("(&"), "no address-of on the compound-assign LHS, got: " & outCuda
  doAssert not outCuda.contains("+="), "compound assignment must be desugared, got: " & outCuda
  doAssert outCuda.contains("C[i] = (C[i] + 1)"),
    "expected direct read-modify-write `C[i] = (C[i] + 1)`, got: " & outCuda
  echo "  OK — emitted CUDA is a direct read-modify-write (no &, no +=)"

echo ""
echo "  All compound-assign LHS blit tests passed."
