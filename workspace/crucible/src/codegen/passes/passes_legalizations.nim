## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, sets, tables]
import ../ir/gpu_types
import ./pass_datatypes
import ./passes_utils
export getExprType, dedupVarNames

proc insertResult(ctx: var GpuContext; fn: GpuAst) =
  ## Insert `var result` and `return result` into a function body.
  if fn.pRetType.kind == gtVoid: return

  proc lastIsReturn(n: GpuAst): bool =
    doAssert n.kind == gpuBlock
    if n.statements[^1].kind == gpuReturn: return true

  if not lastIsReturn(fn.pBody):
    let resSym = newSymbol("result", iSym = "result", typ = fn.pRetType, symKind = gsLocal)
    let resId = GpuAst(kind: gpuIdent, symbol: resSym)
    let res = GpuAst(kind: gpuVar, vName: resId,
                     vType: fn.pRetType,
                     vInit: GpuAst(kind: gpuDiscard),
                     vRequiresMemcpy: false,
                     vMutable: true)
    fn.pBody.statements.insert(res, 0)

    if not lastIsReturn(fn.pBody):
      fn.pBody.statements.add GpuAst(kind: gpuReturn, rValue: resId)

    for i in countdown(fn.pBody.statements.high, 0):
      let stmt = fn.pBody.statements[i]
      if stmt.kind notin {gpuVar, gpuComment, gpuDiscard, gpuReturn, gpuIf, gpuFor, gpuWhile, gpuEmit}:
        if stmt.kind == gpuBlock and stmt.isExpr:
          if stmt.statements.len == 1:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt.statements[0])
          else:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        elif stmt.kind != gpuAssign:
          fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        break

proc firstNonBlock(n: GpuAst): GpuAst =
  ## Unwrap single-statement gpuBlock wrappers.
  result = n
  while result.kind == gpuBlock and result.statements.len == 1:
    result = result.statements[0]

proc liftConstexprFrom(n: var GpuAst; lifts: var seq[GpuAst]) =
  ## Lift gpuConstexpr nodes from expression children, replacing them with their cIdent.
  case n.kind
  of gpuConstexpr:
    lifts.add n
    n = n.cIdent
  of gpuVar:
    liftConstexprFrom(n.vInit, lifts)
  of gpuAssign:
    liftConstexprFrom(n.aRight, lifts)
  of gpuDot:
    liftConstexprFrom(n.dParent, lifts)
    liftConstexprFrom(n.dField, lifts)
  of gpuIndex:
    liftConstexprFrom(n.iArr, lifts)
    liftConstexprFrom(n.iIndex, lifts)
  of gpuCall:
    for i in 0 ..< n.cArgs.len:
      liftConstexprFrom(n.cArgs[i], lifts)
  of gpuObjConstr:
    for i in 0 ..< n.ocFields.len:
      liftConstexprFrom(n.ocFields[i].value, lifts)
  of gpuReturn:
    liftConstexprFrom(n.rValue, lifts)
  of gpuAddr:
    liftConstexprFrom(n.aOf, lifts)
  of gpuDeref:
    liftConstexprFrom(n.dOf, lifts)
  of gpuConv:
    liftConstexprFrom(n.convExpr, lifts)
  of gpuCast:
    liftConstexprFrom(n.cExpr, lifts)
  of gpuBinOp:
    liftConstexprFrom(n.bLeft, lifts)
    liftConstexprFrom(n.bRight, lifts)
  of gpuPrefix:
    liftConstexprFrom(n.pVal, lifts)
  of gpuIf:
    liftConstexprFrom(n.ifCond, lifts)
    for el in n.ifElifs.mitems:
      liftConstexprFrom(el.cond, lifts)
  of gpuFor:
    liftConstexprFrom(n.fStart, lifts)
    liftConstexprFrom(n.fEnd, lifts)
  of gpuWhile:
    liftConstexprFrom(n.wCond, lifts)
  of gpuArrayLit:
    for i in 0 ..< n.aValues.len:
      liftConstexprFrom(n.aValues[i], lifts)
  of gpuBlock:
    for i in 0 ..< n.statements.len:
      if n.statements[i].kind == gpuConstexpr:
        lifts.add n.statements[i]
        n.statements[i] = GpuAst(kind: gpuDiscard)
      else:
        liftConstexprFrom(n.statements[i], lifts)
  else:
    discard

proc liftConstexpr(pbody: var GpuAst) =
  ## Extract gpuConstexpr nodes from expression children into preceding statements.
  ##
  ## Nim source example:
  ##   let x = block:
  ##     const tmp {.genSym.} = Int[8]()   ← gpuConstexpr in expression slot
  ##     tmp
  ##
  ## Before (IR):
  ##   gpuVar(vName: "x", vInit: gpuBlock(isExpr: true,
  ##     statements: @[gpuConstexpr(cIdent: "tmp"), gpuIdent("tmp")]))
  ##
  ## After (IR):
  ##   gpuConstexpr(cIdent: "tmp")              ← lifted to statement position
  ##   gpuVar(vName: "x", vInit: gpuBlock(isExpr: true,
  ##     statements: @[gpuIdent("tmp")]))
  ##
  ## Benefit: constexpr declarations are statements, not expressions.
  ## Lifting them prevents codegen from emitting `constexpr Type name = ...`
  ## inside an expression context (e.g. vInit), which would be invalid C++.
  case pbody.kind
  of gpuBlock:
    var newStmts: seq[GpuAst]
    var liftedSyms: HashSet[string]  # dedup by cIdent.symbol.iSym
    for stmt in pbody.statements.mitems:
      if stmt.kind != gpuConstexpr:
        var lifts: seq[GpuAst]
        liftConstexprFrom(stmt, lifts)
        for i in 0 ..< lifts.len:
          let csym = lifts[i].cIdent.symbol.iSym
          if csym notin liftedSyms:
            liftedSyms.incl csym
            newStmts.add lifts[i]
      newStmts.add stmt
    pbody.statements = newStmts
    for i in 0 ..< pbody.statements.len:
      if pbody.statements[i].kind != gpuConstexpr:
        liftConstexpr(pbody.statements[i])
  of gpuIf:
    liftConstexpr(pbody.ifThen)
    for el in pbody.ifElifs.mitems:
      liftConstexpr(el.body)
    if pbody.ifElse.kind != gpuDiscard:
      liftConstexpr(pbody.ifElse)
  of gpuFor:
    liftConstexpr(pbody.fBody)
  of gpuWhile:
    liftConstexpr(pbody.wBody)
  else:
    discard

proc registerLegalizationPasses*(reg: var PassRegistry) =
  ## Register passes that make the IR well-formed.

  reg.register("ensureBlock", pkTransform, phaseEarly,
    "Wraps non-block bodies in gpuBlock",
    proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.pBody.kind != gpuBlock:
          fn.pBody = GpuAst(kind: gpuBlock, statements: @[fn.pBody])
        fn.pBody.walk(proc(n: var GpuAst): void =
          case n.kind
          of gpuIf:
            if n.ifThen.kind != gpuBlock:
              n.ifThen = GpuAst(kind: gpuBlock, statements: @[n.ifThen])
            for el in n.ifElifs.mitems:
              if el.body.kind != gpuBlock:
                el.body = GpuAst(kind: gpuBlock, statements: @[el.body])
            if n.ifElse.kind != gpuDiscard and n.ifElse.kind != gpuBlock:
              n.ifElse = GpuAst(kind: gpuBlock, statements: @[n.ifElse])
          of gpuFor:
            if n.fBody.kind != gpuBlock:
              n.fBody = GpuAst(kind: gpuBlock, statements: @[n.fBody])
          of gpuWhile:
            if n.wBody.kind != gpuBlock:
              n.wBody = GpuAst(kind: gpuBlock, statements: @[n.wBody])
          else: discard
        )
    )

  reg.register("maybeInsertResult", pkTransform, phaseMain,
    "Inserts var result and return result",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        insertResult(ctx, fn)
    )


  reg.register("liftConstexpr", pkTransform, phaseMain,
    "Extracts gpuConstexpr from expression slots into preceding statements",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        liftConstexpr(fn.pBody)
    )

  reg.register("blitBlockExprs", pkTransform, phaseMain,
    "Converts gpuBlock(isExpr:true) into scope blocks + blit temps",
    dependsOn = @["ensureBlock", "liftConstexpr"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        let fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          blitFnBody(ctx, fn.pBody, fn.pRetType)
      for fnKey in ctx.genericInsts.keys:
        let fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          blitFnBody(ctx, fn.pBody, fn.pRetType)
    )



  reg.register("unwrapBlockInDot", pkTransform, phaseMain,
    "Unwraps single-stmt gpuBlock(isExpr) inside gpuDot dParent chains",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        fn.pBody.walk(proc(n: var GpuAst): void =
          unwrapBlockInDot(n)
        )
    )
