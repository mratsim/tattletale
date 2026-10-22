## Compound assignment through the frontend and the common pipeline: gpuAssign(aOp) reaches the codegen boundary intact
##
## Test that gpuAssign(aOp) reaches the codegen boundary with no expansion node, and block lvalue tails stay the real lvalue
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_blitCompoundAssign.nim

import std/[strutils, tables]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/tests/ir/ir_testutils

proc countExprBlocks(n: GpuAst): int =
  if n == nil:
    return 0
  result = if n.kind == gpuBlock and n.isExpr: 1 else: 0
  for ch in n.items:
    result += countExprBlocks(ch)

proc collectAssigns(n: GpuAst, acc: var seq[GpuAst]) =
  if n == nil:
    return
  if n.kind == gpuAssign:
    acc.add n
  for ch in n.items:
    collectAssigns(ch, acc)

proc collectBinOps(n: GpuAst, acc: var seq[GpuAst]) =
  if n == nil:
    return
  if n.kind == gpuBinOp:
    acc.add n
  for ch in n.items:
    collectBinOps(ch, acc)

proc isBareIndexLhs(n: GpuAst): bool =
  n.kind == gpuIndex and n.iArr.kind in {gpuIdent, gpuDeref}

# ── 1. Frontend emission: compound assignments enter as gpuAssign(aOp) ──
static:
  let ir = toCrucibleAst:
    proc compoundKernel(C: ptr UncheckedArray[int32]) {.global.} =
      C[0] += 5
      C[0] *= 2
  var assigns: seq[GpuAst]
  ir.statements[0].pBody.collectAssigns(assigns)
  doAssert assigns.len == 2,
    "expected the two compound assignments, got: " & $assigns.len
  doAssert assigns[0].aOp == "+=", "expected aOp \"+=\", got: " & assigns[0].aOp
  doAssert isBareIndexLhs(assigns[0].aLeft),
    "the LHS must be the bare gpuIndex lvalue (no address wrapper), got: " & $assigns[0].aLeft.kind
  doAssert assigns[0].aRight.kind == gpuLit and assigns[0].aRight.lValue == "5",
    "the RHS must stay the untouched operand, no assign+binop expansion, got: " & $assigns[0].aRight.kind
  doAssert assigns[1].aOp == "*=", "expected aOp \"*=\", got: " & assigns[1].aOp
  doAssert assigns[1].aRight.kind == gpuLit and assigns[1].aRight.lValue == "2",
    "the RHS must stay the untouched operand, got: " & $assigns[1].aRight.kind
  echo "  OK — frontend emits gpuAssign(aOp) with the bare lvalue and the untouched RHS"

# ── 2. Hazard negatives: plain assigns and comparisons pass through untouched ──
static:
  let ir = toCrucibleAst:
    proc negativeKernel(C: ptr UncheckedArray[int32]) {.global.} =
      C[0] = 3
      let y = C[1] + 2
      if C[2] < y:
        C[3] = 1
  var assigns: seq[GpuAst]
  var binops: seq[GpuAst]
  ir.statements[0].pBody.collectAssigns(assigns)
  ir.statements[0].pBody.collectBinOps(binops)
  doAssert assigns.len == 2, "expected the two plain assignments, got: " & $assigns.len
  for a in assigns:
    doAssert a.aOp == "=", "plain assignment keeps aOp \"=\", got: " & a.aOp
  doAssert binops.len == 2, "expected the additive and the comparison binop, got: " & $binops.len
  doAssert binops[0].bOp.ident() == "+", "addition stays a binop, got: " & binops[0].bOp.ident()
  doAssert binops[1].bOp.ident() == "<", "comparison stays a binop, got: " & binops[1].bOp.ident()
  echo "  OK — plain assigns keep aOp \"=\", comparisons stay binops"

# ── 3. Full common pipeline: aOp survives to the codegen boundary ──
static:
  let ir = toCrucibleAst:
    proc compoundKernel(C: ptr UncheckedArray[int32]) {.global.} =
      for i in 0 ..< 4:
        C[i] += 5
        C[i] *= 2
  let fn = ir.statements[0]
  doAssert fn.kind == gpuProc, "expected gpuProc, got " & $fn.kind
  let lowered = registerCommonPasses.runPass(fn)
  var assigns: seq[GpuAst]
  lowered.pBody.collectAssigns(assigns)
  doAssert assigns.len == 2,
    "the two compound assignments must reach the codegen boundary, got: " & $assigns.len
  doAssert assigns[0].aOp == "+=", "aOp must survive the pipeline, got: " & assigns[0].aOp
  doAssert isBareIndexLhs(assigns[0].aLeft),
    "the LHS must stay the real gpuIndex lvalue, got: " & $assigns[0].aLeft.kind
  doAssert assigns[0].aRight.kind == gpuLit,
    "no expansion node may exist at the codegen boundary, got: " & $assigns[0].aRight.kind
  doAssert assigns[1].aOp == "*=", "aOp must survive the pipeline, got: " & assigns[1].aOp
  echo "  OK — aOp intact after registerCommonPasses, no expansion node"

# ── 4. Block-expression LHS through blit ──
# The frontend strips the address wrapper from var-template arguments, so the pipeline sees the bare expression block as the assignment LHS.
# Blitting materializes the intermediate statement and keeps the tail as the compound lvalue with aOp preserved.
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
  let assign = GpuAst(kind: gpuAssign, aLeft: lhsBlock,
                      aRight: GpuAst(kind: gpuLit, lValue: "1", lType: int32t),
                      aOp: "+=")
  let body = GpuAst(kind: gpuBlock, statements: @[assign])
  let fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol("compoundKernel", iSym = "compoundKernel", symKind = gsProc)),
                  pRetType: GpuType(kind: gtVoid),
                  pParams: @[],
                  pBody: body)
  let lowered = registerCommonPasses.runPass(fn)
  doAssert countExprBlocks(lowered.pBody) == 0,
    "the block-expression LHS must be fully blitted, leftover: " & $countExprBlocks(lowered.pBody)
  var assigns: seq[GpuAst]
  lowered.pBody.collectAssigns(assigns)
  var compound: GpuAst = nil
  for a in assigns:
    if a.aOp == "+=":
      compound = a
  doAssert not compound.isNil, "the compound assignment must survive blitting"
  doAssert compound.aLeft.kind == gpuIndex,
    "the assignment LHS must be the real gpuIndex lvalue (direct store-back), got: " & $compound.aLeft.kind
  doAssert compound.aLeft.iArr.kind == gpuIdent and compound.aLeft.iArr.ident() == "acc",
    "the LHS lvalue must be acc[...], got: " & $compound.aLeft.iArr.kind
  echo "  OK — block-expression LHS: tail kept as lvalue, aOp preserved"

echo ""
echo "  All compound assignment pipeline tests passed."
