## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, sets, tables]
import ../ir/gpu_types
import ./pass_datatypes

proc insertResult(ctx: var GpuContext; fn: GpuAst) =
  ## Insert `var result` and `return result` into a function body.
  if fn.pRetType.kind == gtVoid: return

  proc lastIsReturn(n: GpuAst): bool =
    doAssert n.kind == gpuBlock
    if n.statements[^1].kind == gpuReturn: return true

  if not lastIsReturn(fn.pBody):
    let resId = GpuAst(kind: gpuIdent, iName: "result",
                       iSym: "result",
                       iTyp: fn.pRetType,
                       symbolKind: gsLocal)
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
      if stmt.kind notin {gpuVar, gpuComment, gpuDiscard, gpuReturn, gpuIf, gpuFor, gpuWhile}:
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
    var liftedSyms: HashSet[string]  # dedup by cIdent.iSym
    for stmt in pbody.statements.mitems:
      if stmt.kind != gpuConstexpr:
        var lifts: seq[GpuAst]
        liftConstexprFrom(stmt, lifts)
        for i in 0 ..< lifts.len:
          let csym = lifts[i].cIdent.iSym
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
    if pbody.ifElse.kind != gpuDiscard:
      liftConstexpr(pbody.ifElse)
  of gpuFor:
    liftConstexpr(pbody.fBody)
  of gpuWhile:
    liftConstexpr(pbody.wBody)
  else:
    discard
proc getExprType(n: GpuAst; ctx: GpuContext): GpuType =
  ## Read the type of an expression from existing node fields.
  ## Errors if the node kind doesn't carry its own type (e.g. gpuDot, gpuIndex).
  ## Those cases require the caller to provide the type from context instead.
  if n == nil: error "Cannot get type of nil node"
  case n.kind
  of gpuIdent: result = n.iTyp
  of gpuLit: result = n.lType
  of gpuCall: result = ctx.getFnReturnType(n.cName)
  of gpuObjConstr: result = n.ocType
  of gpuConstexpr: result = n.cType
  of gpuMaterialize: result = n.mType
  of gpuConv: result = n.convTo
  of gpuCast: result = n.cTo
  of gpuBlock:
    if n.statements.len > 0:
      result = getExprType(n.statements[^1], ctx)
    else:
      error "Empty block expression"
  else:
    error "getExprType: unhandled node kind " & $n.kind & ". Caller must provide type from context."

proc blitExprSlot(slot: var GpuAst; ctx: var GpuContext; blitType: GpuType; fnRetType: GpuType): seq[GpuAst] =
  ## Process an expression slot. If `slot` is a gpuBlock(isExpr: true),
  ## replace it with a blit temp reference and return preamble statements.
  if slot.kind == gpuBlock and slot.isExpr:
    if slot.statements.len == 1:
      slot = slot.statements[0]
      result = blitExprSlot(slot, ctx, blitType, fnRetType)
    elif slot.statements.len > 1:
      var t = blitType
      if t.isNil or t.kind == gtVoid:
        t = getExprType(slot.statements[^1], ctx)
      if t.isNil or t.kind == gtVoid:
        t = fnRetType
      if t.isNil or t.kind == gtVoid:
        error "Cannot determine type for blit temp in block expression"
      let blitName = "_blit_" & $ctx.genSymCount
      inc ctx.genSymCount
      let blitIdent = GpuAst(kind: gpuIdent, iName: blitName, iSym: blitName,
                             iTyp: t, symbolKind: gsLocal)
      let blitDecl = GpuAst(kind: gpuVar, vName: blitIdent, vType: t,
                            vInit: GpuAst(kind: gpuDiscard), vMutable: true)
      let lastStmt = slot.statements[^1]
      slot.statements[^1] = GpuAst(kind: gpuAssign,
                                   aLeft: blitIdent.clone(),
                                   aRight: lastStmt)
      slot.isExpr = false
      slot.blockLabel = blitName
      let scopeBlock = slot
      let blitRef = GpuAst(kind: gpuIdent, iName: blitName, iSym: blitName,
                           iTyp: t, symbolKind: gsLocal)
      slot = blitRef
      result = @[blitDecl, scopeBlock]
    else:
      error "Empty block expression"
    return
  result = @[]
  case slot.kind
  of gpuVar:
    result = blitExprSlot(slot.vInit, ctx, slot.vType, fnRetType)
  of gpuCall:
    let params = ctx.getFnParams(slot.cName)
    for i in 0 ..< slot.cArgs.len:
      let pt = if i < params.len: params[i].typ else: GpuType(kind: gtVoid)
      let p = blitExprSlot(slot.cArgs[i], ctx, pt, fnRetType)
      result.add p
  of gpuTemplateCall:
    for i in 0 ..< slot.tcArgs.len:
      let p = blitExprSlot(slot.tcArgs[i], ctx, GpuType(kind: gtVoid), fnRetType)
      result.add p
  of gpuReturn:
    result = blitExprSlot(slot.rValue, ctx, fnRetType, fnRetType)
  of gpuAssign:
    # For gpuIdent LHS: use its iTyp directly
    # For gpuIndex LHS (e.g. arr[i] = block: ...): extract element type
    var lhsType = GpuType(kind: gtVoid)
    if slot.aLeft.kind == gpuIdent:
      lhsType = slot.aLeft.iTyp
    elif slot.aLeft.kind == gpuIndex and slot.aLeft.iArr.kind == gpuIdent:
      let arrTyp = slot.aLeft.iArr.iTyp
      if arrTyp != nil:
        if arrTyp != nil:
          case arrTyp.kind
          of gtPtr: lhsType = arrTyp.to
          of gtArray: lhsType = arrTyp.aTyp
          of gtUA: lhsType = arrTyp.uaTo
          else: discard
    result = blitExprSlot(slot.aRight, ctx, lhsType, fnRetType)
  of gpuBinOp:
    result.add blitExprSlot(slot.bLeft, ctx, GpuType(kind: gtVoid), fnRetType)
    result.add blitExprSlot(slot.bRight, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuDot:
    result.add blitExprSlot(slot.dParent, ctx, GpuType(kind: gtVoid), fnRetType)
    result.add blitExprSlot(slot.dField, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuIndex:
    result.add blitExprSlot(slot.iArr, ctx, GpuType(kind: gtVoid), fnRetType)
    result.add blitExprSlot(slot.iIndex, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuPrefix:
    result = blitExprSlot(slot.pVal, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuAddr:
    result = blitExprSlot(slot.aOf, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuDeref:
    result = blitExprSlot(slot.dOf, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuConv:
    result = blitExprSlot(slot.convExpr, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuCast:
    result = blitExprSlot(slot.cExpr, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuIf:
    result.add blitExprSlot(slot.ifCond, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuFor:
    result.add blitExprSlot(slot.fStart, ctx, GpuType(kind: gtVoid), fnRetType)
    result.add blitExprSlot(slot.fEnd, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuWhile:
    result = blitExprSlot(slot.wCond, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuObjConstr:
    for f in slot.ocFields.mitems:
      result.add blitExprSlot(f.value, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuArrayLit:
    for v in slot.aValues.mitems:
      result.add blitExprSlot(v, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuMaterialize:
    result = blitExprSlot(slot.mExpr, ctx, GpuType(kind: gtVoid), fnRetType)
  of gpuConstexpr:
    result = blitExprSlot(slot.cValue, ctx, GpuType(kind: gtVoid), fnRetType)
  else:
    discard

proc blitFnBody(body: var GpuAst; ctx: var GpuContext; fnRetType: GpuType) =
  ## Walk a function body tree, blitting all gpuBlock(isExpr: true) nodes.
  case body.kind
  of gpuBlock:
    # RE loop: blit standalone expression blocks, recurse into nested blocks
    var blockPreambles: seq[seq[GpuAst]]
    for i in 0 ..< body.statements.len:
      if body.statements[i].kind == gpuBlock and body.statements[i].isExpr:
        blockPreambles.add blitExprSlot(body.statements[i], ctx, GpuType(kind: gtVoid), fnRetType)
      else:
        blockPreambles.add @[]
        let sk = body.statements[i].kind
        if sk in {gpuBlock, gpuIf, gpuFor, gpuWhile}:
          blitFnBody(body.statements[i], ctx, fnRetType)
    var newStmts: seq[GpuAst]
    for i, stmt in body.statements.mpairs:
      var preamble = blockPreambles[i]
      case stmt.kind
      of gpuVar:
        if preamble.len == 0:
          preamble = blitExprSlot(stmt.vInit, ctx, stmt.vType, fnRetType)
      of gpuCall:
        let params = ctx.getFnParams(stmt.cName)
        for j in 0 ..< stmt.cArgs.len:
          let pt = if j < params.len: params[j].typ else: GpuType(kind: gtVoid)
          let p = blitExprSlot(stmt.cArgs[j], ctx, pt, fnRetType)
          preamble.add p
      of gpuTemplateCall:
        for j in 0 ..< stmt.tcArgs.len:
          let p = blitExprSlot(stmt.tcArgs[j], ctx, GpuType(kind: gtVoid), fnRetType)
          preamble.add p
      of gpuReturn:
        if preamble.len == 0:
          preamble = blitExprSlot(stmt.rValue, ctx, fnRetType, fnRetType)
      of gpuAssign:
        if preamble.len == 0:
          var lhsType = GpuType(kind: gtVoid)
          if stmt.aLeft.kind == gpuIdent:
            lhsType = stmt.aLeft.iTyp
            if lhsType.isNil or lhsType.kind == gtVoid:
              lhsType = fnRetType
          elif stmt.aLeft.kind == gpuIndex:
            var base = stmt.aLeft.iArr
            if base.kind == gpuDeref:
              base = base.dOf
            if base.kind == gpuIdent:
              let arrTyp = base.iTyp
              if arrTyp != nil:
                case arrTyp.kind
                of gtPtr: lhsType = arrTyp.to
                of gtArray: lhsType = arrTyp.aTyp
                of gtUA: lhsType = arrTyp.uaTo
                else: discard
          preamble = blitExprSlot(stmt.aRight, ctx, lhsType, fnRetType)
      of gpuBlock:
        if stmt.isExpr:
          preamble = blitExprSlot(stmt, ctx, GpuType(kind: gtVoid), fnRetType)
        elif stmt.blockLabel.len == 0:
          if stmt.statements.len == 1:
            stmt = stmt.statements[0]
      of gpuIf:
        preamble.add blitExprSlot(stmt.ifCond, ctx, GpuType(kind: gtVoid), fnRetType)
      of gpuFor:
        preamble.add blitExprSlot(stmt.fStart, ctx, GpuType(kind: gtVoid), fnRetType)
        preamble.add blitExprSlot(stmt.fEnd, ctx, GpuType(kind: gtVoid), fnRetType)
      of gpuWhile:
        preamble = blitExprSlot(stmt.wCond, ctx, GpuType(kind: gtVoid), fnRetType)
      else:
        discard
      newStmts.add preamble
      newStmts.add stmt
    body.statements = newStmts
    # Recursively process newly created statements (e.g. scope blocks from blitting)
    for i in 0 ..< body.statements.len:
      blitFnBody(body.statements[i], ctx, fnRetType)
  of gpuIf:
    blitFnBody(body.ifThen, ctx, fnRetType)
    if body.ifElse.kind != gpuDiscard:
      blitFnBody(body.ifElse, ctx, fnRetType)
  of gpuFor:
    blitFnBody(body.fBody, ctx, fnRetType)
  of gpuWhile:
    blitFnBody(body.wBody, ctx, fnRetType)
  else:
    discard

proc unwrapBlockInDot(n: var GpuAst) =
  ## Unwrap single-stmt gpuBlock(isExpr) when used as gpuDot's dParent.
  case n.kind
  of gpuDot:
    if n.dParent.kind == gpuBlock and n.dParent.isExpr and n.dParent.statements.len == 1:
      n.dParent = n.dParent.statements[0]
    else:
      unwrapBlockInDot(n.dParent)
    unwrapBlockInDot(n.dField)
  of gpuIndex:
    unwrapBlockInDot(n.iArr)
    unwrapBlockInDot(n.iIndex)
  of gpuDeref:
    unwrapBlockInDot(n.dOf)
  of gpuAddr:
    unwrapBlockInDot(n.aOf)
  of gpuCall:
    for arg in n.cArgs.mitems:
      unwrapBlockInDot(arg)
  of gpuTemplateCall:
    for arg in n.tcArgs.mitems:
      unwrapBlockInDot(arg)
  of gpuObjConstr:
    for f in n.ocFields.mitems:
      unwrapBlockInDot(f.value)
  of gpuArrayLit:
    for v in n.aValues.mitems:
      unwrapBlockInDot(v)
  of gpuReturn:
    unwrapBlockInDot(n.rValue)
  of gpuAssign:
    unwrapBlockInDot(n.aLeft)
    unwrapBlockInDot(n.aRight)
  of gpuVar:
    unwrapBlockInDot(n.vInit)
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
          blitFnBody(fn.pBody, ctx, fn.pRetType)
      for fnKey in ctx.genericInsts.keys:
        let fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          blitFnBody(fn.pBody, ctx, fn.pRetType)
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
