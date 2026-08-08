## Regression: blitBlockExprs final recursion (DEV-003 preamble-only walk)
##
## blitFnBody's PASS 3 descends only into the NEW content produced by
## blitting — blit scope blocks and hoisted lvalue wrappers — instead of
## re-walking the whole statement list (which is O(N x blit-depth) and blows
## up on deeply-nested expression-block trees such as ceramic evalOnceAs /
## crd2idx). This test locks the preamble-recursion semantics:
##
##   1. Nested expr-block tree (toGpuAst from real Nim source): blit scope
##      blocks nest; every gpuBlock(isExpr: true) must be gone after the pass.
##   2. Hoisted lvalue wrappers (hand-built IR): an expr block used as an
##      lvalue hoists its intermediate statements into the preamble; an
##      intermediate var whose vInit is itself an expr block must be blitted
##      inline, and the aRight expr block must become a blit scope block.
##   3. Emission (cuda: macro): the emitted CUDA must contain the nested blit
##      scope blocks with the inner statement order preserved — an emission
##      anchor for the rewrite. A naive rewrite that leaves any expr block
##      unblitted fails at compile time ("Block expression survived to
##      codegen") or changes this emission.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off --debugger:native \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/ir/test_ir_blitRecursion.nim

import std/[macros, strutils, tables]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/pass_datatypes

proc countExprBlocks(n: GpuAst): int =
  ## Count gpuBlock(isExpr: true) nodes in a subtree.
  if n == nil: return 0
  result = if n.kind == gpuBlock and n.isExpr: 1 else: 0
  for ch in n.items:
    result += countExprBlocks(ch)

proc countBlitScopes(n: GpuAst): int =
  ## Count labeled blit scope blocks (`_blit_` labels) in a subtree.
  if n == nil: return 0
  if n.kind == gpuBlock and n.blockLabel.startsWith("_blit_"):
    inc result
  for ch in n.items:
    result += countBlitScopes(ch)

# ── 1. Nested expr-block tree from real Nim source ──
static:
  # Two levels of nested 2-statement expression blocks: blitting the outer
  # block creates a scope block that CONTAINS the inner block, so PASS 3 must
  # descend into the newly created preamble to blit it (and the inner one).
  let ir = toGpuAst:
    proc nestedKernel(x: int32): int32 {.device.} =
      let a = block:
        let b = block:
          let t = x + 1
          t + 2
        b + 3
      a
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  let fn = ir.statements[0]
  doAssert fn.kind == gpuProc, "Expected gpuProc, got " & $fn.kind
  ctx.allFnTab[fn.pName] = fn
  runPasses(ctx, reg)
  doAssert countExprBlocks(fn.pBody) == 0,
    "nested expr-block tree must be fully blitted, leftover: " & $countExprBlocks(fn.pBody)
  doAssert countBlitScopes(fn.pBody) >= 3,
    "expected >= 3 nested blit scope blocks, got " & $countBlitScopes(fn.pBody)

# ── 2. Hoisted lvalue wrappers (hand-built IR) ──
static:
  # gpuAssign whose aLeft is an expr block [var v (vInit = expr block), last]
  # and whose aRight is a 2-statement expr block. Blitting must:
  #   - blit the intermediate v's vInit into a scope block, hoisting
  #     `_blit_0` + scope before `var v = _blit_0`
  #   - blit aRight into `_blit_1` + scope
  #   - leave the assign as `w = _blit_1`
  let int32t = GpuType(kind: gtInt32)
  let innerBinOp = GpuAst(kind: gpuBinOp, bType: int32t,
                          bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+", iSym = "+")),
                          bLeft: GpuAst(kind: gpuIdent, symbol: newSymbol("u", iSym = "u", typ = int32t, symKind = gsLocal)),
                          bRight: GpuAst(kind: gpuLit, lValue: "1", lType: int32t))
  let innerBlk = GpuAst(kind: gpuBlock, isExpr: true, statements: @[
    GpuAst(kind: gpuLit, lValue: "0", lType: int32t),
    innerBinOp])
  let vIdent = GpuAst(kind: gpuIdent, symbol: newSymbol("v", iSym = "v", typ = int32t, symKind = gsLocal))
  let vVar = GpuAst(kind: gpuVar, vName: vIdent, vType: int32t, vInit: innerBlk, vMutable: true)
  let lastLval = GpuAst(kind: gpuIdent, symbol: newSymbol("w", iSym = "w", typ = int32t, symKind = gsLocal))
  let leftBlock = GpuAst(kind: gpuBlock, isExpr: true, statements: @[vVar, lastLval])
  let rightBlock = GpuAst(kind: gpuBlock, isExpr: true, statements: @[
    GpuAst(kind: gpuLit, lValue: "1", lType: int32t),
    GpuAst(kind: gpuLit, lValue: "2", lType: int32t)])
  let assign = GpuAst(kind: gpuAssign, aLeft: leftBlock, aRight: rightBlock)
  let body = GpuAst(kind: gpuBlock, isExpr: false, statements: @[assign])
  let fn = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent, symbol: newSymbol("lvalKernel", iSym = "lvalKernel", symKind = gsProc)),
                  pRetType: int32t,
                  pParams: @[],
                  pBody: body)
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  ctx.allFnTab[fn.pName] = fn
  runPasses(ctx, reg)
  doAssert countExprBlocks(fn.pBody) == 0,
    "lvalue-hoist shape must be fully blitted, leftover: " & $countExprBlocks(fn.pBody)
  # The hoisted intermediate var must carry a blit ref (its expr-block init
  # was blitted inline), and the assign's aRight must be a blit ref too.
  var foundV = false
  var assignRightIsBlitRef = false
  proc check(n: GpuAst) =
    if n == nil: return
    if n.kind == gpuVar and n.vName.symbol.name == "v":
      foundV = true
      doAssert n.vInit.kind == gpuIdent, "hoisted v must carry a blit ref vInit, got " & $n.vInit.kind
    if n.kind == gpuAssign and n.aLeft.kind == gpuIdent and n.aLeft.symbol.name == "w":
      assignRightIsBlitRef = n.aRight.kind == gpuIdent and n.aRight.symbol.name == "_blit_1"
    for ch in n.items:
      check(ch)
  check(fn.pBody)
  doAssert foundV, "expected the hoisted intermediate var v in the blitted body"
  doAssert assignRightIsBlitRef, "lvalue expr-block assign must end as w = _blit_1"

# ── 3. Emission anchor: cuda: macro output for the nested shape ──
const outCuda = cuda:
  proc nestedKernel(x: int32): int32 {.device.} =
    let a = block:
      let b = block:
        let t = x + 1
        t + 2
      b + 3
    a
  proc regressionKernel(C: ptr UncheckedArray[int32]) {.global.} =
    C[0] = nestedKernel(C[0])

block:
  doAssert outCuda.contains("nestedKernel"), "emission must include the nested device proc"
  doAssert outCuda.contains("_blit_0") and outCuda.contains("_blit_1") and outCuda.contains("_blit_2"),
    "emission must contain the three nested blit temps"
  # Statement order inside the innermost scope block: the let t = (x + 1)
  # must come BEFORE the blit assignment _blit_2 = (t + 2).
  let tIdx = outCuda.find("int t = (x + 1)")
  let blit2Idx = outCuda.find("_blit_2 = (t + 2)")
  doAssert tIdx >= 0 and blit2Idx >= 0, "emission must contain the innermost let + blit assign"
  doAssert tIdx < blit2Idx, "inner let must be emitted before its blit assignment"
  # The innermost scope block must be lexically inside the outer ones.
  let blit1Idx = outCuda.find("_blit_1 = (b + 3)")
  let blit0Idx = outCuda.find("_blit_0 = a")
  doAssert blit1Idx >= 0 and blit0Idx >= 0, "emission must contain the outer blit assignments"
  doAssert blit2Idx < blit1Idx and blit1Idx < blit0Idx,
    "nested blit assignments must be ordered _blit_2 < _blit_1 < _blit_0"

echo "  OK — blit recursion regression: nested scope blocks, hoisted lvalue wrappers, emission"

echo ""
echo "  All blit recursion tests passed."
