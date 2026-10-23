# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, sets, strutils, tables]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./pass_datatypes
import ./pass_helpers
import ./passes_legal_if_expr
import ./passes_legal_block_expr

# ############################################################
#
# Legalizations
#
# ############################################################
#
# Legalizations conform the IR to the crucible IR model itself:
# every statement is a block, result is assigned, constexprs are lifted.
# Target-independent, every backend registers them identically.

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerLegalizationPasses*(reg: var PassRegistry) =
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

  reg.registerUnwrapBlockInDot()

  reg.registerRewriteIndexDeref()
  reg.registerMapIfExprToTernary()
  reg.registerBlitBlockExprs()

func registerRewriteIndexDeref(reg: var PassRegistry) =
  reg.register("rewriteIndexDeref", pkTransform, phaseMain,
    "Normalizes gpuIndex(gpuDeref(...)) to gpuIndex(dOf(...)) for pointer types",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          rewriteIndexDerefImpl(ctx, fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          rewriteIndexDerefImpl(ctx, fn.pBody)
  )

# ═══════════════════════════════════════════════════════════════════════
# Result insertion
# ═══════════════════════════════════════════════════════════════════════

proc insertResult(ctx: var GpuContext, fn: GpuAst) =
  if fn.pRetType.kind == gtVoid:
    return

  proc lastIsReturn(n: GpuAst): bool =
    doAssert n.kind == gpuBlock
    if n.statements[^1].kind == gpuReturn:
      return true

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
        if stmt.kind == gpuCall and stmt.cName in ctx.allFnTab and
            ctx.allFnTab[stmt.cName].pRetType.kind == gtVoid:
          # void calls mutate var params in place, keep scanning
          continue
        if stmt.kind == gpuBlock and stmt.isExpr:
          if stmt.statements.len == 1:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt.statements[0])
          else:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        elif stmt.kind != gpuAssign:
          fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        break


# ═══════════════════════════════════════════════════════════════════════
# Constexpr lifting
# ═══════════════════════════════════════════════════════════════════════

proc liftConstexprFrom(n: var GpuAst, lifts: var seq[GpuAst]) =
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
  ## Before (IR): gpuVar(vInit: gpuBlock(isExpr: true, statements: @[tmp]))
  ## After (IR): gpuConstexpr(cIdent: "tmp") lifts to statement position
  ## the var keeps gpuIdent(tmp) as its vInit
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


# ═══════════════════════════════════════════════════════════════════════
# Index deref normalization
# ═══════════════════════════════════════════════════════════════════════

proc rewriteIndexDerefImpl(ctx: var GpuContext, n: var GpuAst) =
  ## `(*arr)[idx]` ≡ `arr[idx]` in C, the IR keeps the simpler form.
  case n.kind
  of gpuIndex:
    if n.iArr.kind == gpuDeref:
      # ptr-to-array keeps the deref (C array decay), plain ptr rewrites
      let isPtrToArray = (block:
        let derefOf = n.iArr.dOf
        if derefOf.kind == gpuIdent and derefOf.symbol != nil:
          let symTyp = derefOf.symbol.typ
          symTyp != nil and symTyp.kind == gtPtr and
          symTyp.to != nil and symTyp.to.kind == gtArray
        else:
          false)
      if not isPtrToArray:
        n = GpuAst(kind: gpuIndex, iArr: n.iArr.dOf, iIndex: n.iIndex)
    for ch in mitems(n):
      rewriteIndexDerefImpl(ctx, ch)
  else:
    for ch in mitems(n):
      rewriteIndexDerefImpl(ctx, ch)
