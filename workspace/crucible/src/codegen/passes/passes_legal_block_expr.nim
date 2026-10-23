# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
# Block-expression lowering
#
# ############################################################
#
# Dot-block unwrapping, statement if-branches and block-expression blitting,
# mutually recursive and kept together by construction.

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerUnwrapBlockInDot*(reg: var PassRegistry) =
  reg.register("unwrapBlockInDot", pkTransform, phaseMain,
    "Unwraps single-stmt gpuBlock(isExpr) used as gpuDot's dParent",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        fn.pBody.walk(proc(n: var GpuAst): void =
          unwrapBlockInDot(n)
        )
    )

import std / [macros, sequtils, sets, strutils, tables]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./pass_datatypes
import ./pass_helpers
import ./passes_legal_if_expr

func registerBlitBlockExprs*(reg: var PassRegistry) =
  reg.register("blitBlockExprs", pkTransform, phaseMain,
    "Converts gpuBlock expressions and statement-branch if-expressions into scope blocks + blit temps",
    dependsOn = @["ensureBlock", "liftConstexpr", "mapIfExprToTernary"],
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



# ═══════════════════════════════════════════════════════════════════════
# Dot block unwrapping
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# If-expression statement branches
# ═══════════════════════════════════════════════════════════════════════

proc blitExprSlot(ctx: var GpuContext, slot: var GpuAst, blitType: GpuType, fnRetType: GpuType): seq[GpuAst]

proc collectUseNames(n: GpuAst, into: var seq[string]) =
  ## Returns the names read in `n`: identifier uses.
  ## Def positions stay excluded, the gpuVar name and the gpuAssign lhs.
  case n.kind
  of gpuIdent:
    into.add n.symbol.name
  of gpuVar:
    collectUseNames(n.vInit, into)
  of gpuAssign:
    collectUseNames(n.aRight, into)
    if n.aLeft.kind != gpuIdent:
      collectUseNames(n.aLeft, into)
  else:
    for ch in n:
      collectUseNames(ch, into)

proc collectDefNames(n: GpuAst, into: var seq[string]) =
  ## Names defined in `n` at any depth: gpuVar decls, gpuAssign to idents.
  case n.kind
  of gpuVar:
    into.add n.vName.symbol.name
  of gpuAssign:
    if n.aLeft.kind == gpuIdent:
      into.add n.aLeft.symbol.name
    collectDefNames(n.aRight, into)
    if n.aLeft.kind != gpuIdent:
      collectDefNames(n.aLeft, into)
  else:
    for ch in n:
      collectDefNames(ch, into)

proc hoistSharedBranchDefs(pre: var seq[GpuAst], otherReads, branchWrites: seq[string], lifted: var seq[GpuAst]) =
  ## Moves a branch's single-target definitions that the other branch reads
  ## to the pre-if preamble.
  ##
  ## toCrucibleAst materializes a shared subexpression once, in the first
  ## branch that evaluates it.
  ## The other branch keeps a cross-branch reference to that temp.
  ##
  ## Sound only for pure definitions reading nothing any branch writes:
  ##
  ## - anything else errors, the other branch would read an undefined temp
  var kept: seq[GpuAst]
  for st in pre.mitems:
    var defs: seq[string]
    collectDefNames(st, defs)
    if defs.len != 1 or defs[0] notin otherReads:
      kept.add st
      continue
    var reads: seq[string]
    collectUseNames(st, reads)
    var hasCall = false
    st.walk(proc(m: var GpuAst): void =
      if m.kind == gpuCall:
        hasCall = true)
    for r in reads:
      if r in branchWrites:
        error "a value shared between if-expression branches depends on branch-written state: " & defs[0]
    if hasCall:
      error "a value shared between if-expression branches has a side-effecting definition: " & defs[0]
    lifted.add st
  pre = kept

proc blitFnBody(ctx: var GpuContext, body: var GpuAst, fnRetType: GpuType)

proc lowerIfExprStatementBranches(ctx: var GpuContext, slot: var GpuAst, fnRetType: GpuType): seq[GpuAst] =
  ## An if-expression whose branches carry statements lowers to a statement
  ## if that assigns a fresh temp.
  ## The unselected branch's statements must not run.
  ## Nothing hoists across branches.
  ##
  ## Before (IR):
  ##   `result = if cond: (block: (t = a; t + 1)) else: u * 2`
  ##
  ## After (IR): the fresh temp `_ifres_0` carries the branch type,
  ## both branches assign it, the slot reads it:
  ##
  ##   if cond:
  ##     t = a
  ##     _ifres_0 = t + 1
  ##   else:
  ##     _ifres_0 = u * 2
  ##   result = _ifres_0
  let condPre = ctx.blitExprSlot(slot.ifCond, GpuType(kind: gtVoid), fnRetType)
  var thenN = unwrapSingleStmtBlocks(slot.ifThen)
  var elseN = unwrapSingleStmtBlocks(slot.ifElse)

  var thenPre, elsePre: seq[GpuAst]
  var thenTail, elseTail: GpuAst
  if thenN.kind == gpuBlock:
    thenPre = thenN.statements[0 ..< thenN.statements.len - 1]
    thenTail = thenN.statements[^1]
  else:
    thenTail = thenN
  if elseN.kind == gpuBlock:
    elsePre = elseN.statements[0 ..< elseN.statements.len - 1]
    elseTail = elseN.statements[^1]
  else:
    elseTail = elseN

  # a branch tail may itself be a statement-branch if-expr: lower it here,
  # its statements belong inside this branch
  if thenTail.kind == gpuIf and thenTail.ifIsExpr:
    thenPre.add ctx.blitExprSlot(thenTail, GpuType(kind: gtVoid), fnRetType)
  if elseTail.kind == gpuIf and elseTail.ifIsExpr:
    elsePre.add ctx.blitExprSlot(elseTail, GpuType(kind: gtVoid), fnRetType)

  var t = ctx.getExprType(thenTail)
  if t.isNil or t.kind == gtVoid:
    t = ctx.getExprType(elseTail)
  if t.isNil or t.kind == gtVoid:
    error "Cannot determine type for the if-expression result temp"

  var thenReads, elseReads: seq[string]
  for st in thenPre:
    collectUseNames(st, thenReads)
  collectUseNames(thenTail, thenReads)
  for st in elsePre:
    collectUseNames(st, elseReads)
  collectUseNames(elseTail, elseReads)
  var branchWrites: seq[string]
  for st in thenPre & elsePre:
    collectDefNames(st, branchWrites)
  var lifted: seq[GpuAst]
  hoistSharedBranchDefs(thenPre, elseReads, branchWrites, lifted)
  hoistSharedBranchDefs(elsePre, thenReads, branchWrites, lifted)

  let ifresName = "_ifres_" & $ctx.genSymCount
  inc ctx.genSymCount
  let ifresSym = newSymbol(ifresName, iSym = ifresName, typ = t, symKind = gsLocal)
  let ifresIdent = GpuAst(kind: gpuIdent, symbol: ifresSym)
  let decl = GpuAst(kind: gpuVar, vName: ifresIdent.clone(), vType: t,
                    vInit: GpuAst(kind: gpuDiscard), vMutable: true)
  let thenBody = GpuAst(kind: gpuBlock, statements: thenPre & @[
    GpuAst(kind: gpuAssign, aLeft: ifresIdent.clone(), aOp: "=", aRight: thenTail)])
  let elseBody = GpuAst(kind: gpuBlock, statements: elsePre & @[
    GpuAst(kind: gpuAssign, aLeft: ifresIdent.clone(), aOp: "=", aRight: elseTail)])
  var ifStmt = GpuAst(kind: gpuIf, ifIsExpr: false,
                      ifCond: slot.ifCond, ifThen: thenBody, ifElse: elseBody)
  ctx.blitFnBody(ifStmt, fnRetType)
  result = condPre & lifted
  result.add decl
  result.add ifStmt
  slot = GpuAst(kind: gpuIdent, symbol: newSymbol(ifresName, iSym = ifresName, typ = t, symKind = gsLocal))


# ═══════════════════════════════════════════════════════════════════════
# Block-expression blitting
# ═══════════════════════════════════════════════════════════════════════

proc blitExprSlot(ctx: var GpuContext, slot: var GpuAst, blitType: GpuType, fnRetType: GpuType): seq[GpuAst] =
  ## Statements cannot live inside a C expression.  Rewrite Nim statement-expressions via a temporary
  ##
  ## Before (IR):
  ##   x = block: { t = a; t + b }
  ##
  ## After (IR):
  ##   int _blit_0                            (the temp, returned first)
  ##   { t = a
  ##     _blit_0 = t + b }                    (the body with its tail stored, returned second)
  ##   x = _blit_0                            (the slot, the block replaced by the temp reference)
  if slot.kind == gpuBlock:
    if slot.statements.len == 1:
      slot = slot.statements[0]
      result = ctx.blitExprSlot(slot, blitType, fnRetType)
    elif slot.statements.len > 1:
      var t = blitType
      if t.isNil or t.kind == gtVoid:
        t = ctx.getExprType(slot.statements[^1])
      if t.isNil or t.kind == gtVoid:
        t = fnRetType
      if t.isNil or t.kind == gtVoid:
        # bType only exists on gpuBinOp, reading it on other kinds raises FieldDefect
        var msg = "Cannot determine type for blit temp in block expression" &
          " (tail kind: " & $slot.statements[^1].kind & ")"
        if slot.statements[^1].kind == gpuBinOp:
          msg.add ", nil bType: " & $slot.statements[^1].bType.isNil
        error msg
      let blitName = "_blit_" & $ctx.genSymCount
      inc ctx.genSymCount
      let blitSym = newSymbol(blitName, iSym = blitName, typ = t, symKind = gsLocal)
      let blitIdent = GpuAst(kind: gpuIdent, symbol: blitSym)
      let blitDecl = GpuAst(kind: gpuVar, vName: blitIdent, vType: t,
                            vInit: GpuAst(kind: gpuDiscard), vMutable: true)
      let lastStmt = slot.statements[^1]
      slot.statements[^1] = GpuAst(kind: gpuAssign,
                                   aLeft: blitIdent.clone(),
                                   aRight: lastStmt)
      slot.isExpr = false
      slot.blockLabel = blitName
      let scopeBlock = slot
      let blitRef = GpuAst(kind: gpuIdent, symbol: newSymbol(blitName, iSym = blitName, typ = t, symKind = gsLocal))
      slot = blitRef
      result = @[blitDecl, scopeBlock]
    else:
      error "Empty block expression"
    return
  result = @[]
  case slot.kind
  of gpuVar:
    result = ctx.blitExprSlot(slot.vInit, slot.vType, fnRetType)
  of gpuCall:
    let params = ctx.getFnParams(slot.cName)
    for i in 0 ..< slot.cArgs.len:
      let pt = if i < params.len: params[i].typ else: GpuType(kind: gtVoid)
      let p = ctx.blitExprSlot(slot.cArgs[i], pt, fnRetType)
      result.add p
  of gpuReturn:
    result = ctx.blitExprSlot(slot.rValue, fnRetType, fnRetType)
  of gpuAssign:
    # For gpuIdent LHS: use its iTyp directly
    # For gpuIndex LHS (e.g. arr[i] = block: ...): extract element type
    var lhsType = GpuType(kind: gtVoid)
    if slot.aLeft.kind == gpuIdent:
      lhsType = slot.aLeft.symbol.typ
    elif slot.aLeft.kind == gpuIndex and slot.aLeft.iArr.kind == gpuIdent:
      let arrTyp = slot.aLeft.iArr.symbol.typ
      if arrTyp != nil:
        if arrTyp != nil:
          case arrTyp.kind
          of gtPtr: lhsType = arrTyp.to
          of gtArray: lhsType = arrTyp.aTyp
          of gtUncheckedArray: lhsType = arrTyp.uaTo
          else: discard
    # Expression block as lvalue: hoist intermediate stmts, keep last as lvalue
    if slot.aLeft.kind == gpuBlock and slot.aLeft.isExpr:
      if slot.aLeft.statements.len == 1:
        slot.aLeft = slot.aLeft.statements[0]
        result = ctx.blitExprSlot(slot.aLeft, GpuType(kind: gtVoid), fnRetType)
      elif slot.aLeft.statements.len > 1:
        let lastIdx = slot.aLeft.statements.high
        let lastStmt = slot.aLeft.statements[lastIdx]
        slot.aLeft.statements.setLen(lastIdx)
        # Blit the vInit of any gpuVar in intermediate stmts immediately
        # so blitFnBody doesn't need to re-process them
        var inlined: seq[GpuAst]
        for s in slot.aLeft.statements:
          if s.kind == gpuVar:
            let vBlit = ctx.blitExprSlot(s.vInit, s.vType, fnRetType)
            for p in vBlit:
              inlined.add p
          inlined.add s
        result = inlined
        slot.aLeft = lastStmt
        if slot.aLeft.kind == gpuIdent:
          lhsType = slot.aLeft.symbol.typ
      else:
        error "Empty block expression as lvalue"
      result.add ctx.blitExprSlot(slot.aRight, lhsType, fnRetType)
      return
    result.add ctx.blitExprSlot(slot.aLeft, GpuType(kind: gtVoid), fnRetType)
    # the RHS preamble must append, the assignment references its temps
    result.add ctx.blitExprSlot(slot.aRight, lhsType, fnRetType)
  of gpuBinOp:
    result.add ctx.blitExprSlot(slot.bLeft, GpuType(kind: gtVoid), fnRetType)
    result.add ctx.blitExprSlot(slot.bRight, GpuType(kind: gtVoid), fnRetType)
  of gpuDot:
    result.add ctx.blitExprSlot(slot.dParent, GpuType(kind: gtVoid), fnRetType)
    result.add ctx.blitExprSlot(slot.dField, GpuType(kind: gtVoid), fnRetType)
  of gpuIndex:
    result.add ctx.blitExprSlot(slot.iArr, GpuType(kind: gtVoid), fnRetType)
    result.add ctx.blitExprSlot(slot.iIndex, GpuType(kind: gtVoid), fnRetType)
  of gpuPrefix:
    result = ctx.blitExprSlot(slot.pVal, GpuType(kind: gtVoid), fnRetType)
  of gpuAddr:
    result = ctx.blitExprSlot(slot.aOf, GpuType(kind: gtVoid), fnRetType)
  of gpuDeref:
    result = ctx.blitExprSlot(slot.dOf, GpuType(kind: gtVoid), fnRetType)
  of gpuConv:
    result = ctx.blitExprSlot(slot.convExpr, GpuType(kind: gtVoid), fnRetType)
  of gpuCast:
    result = ctx.blitExprSlot(slot.cExpr, GpuType(kind: gtVoid), fnRetType)
  of gpuIf:
    if not slot.ifIsExpr:
      discard
    elif isExprOnlyBranch(slot.ifThen) and isExprOnlyBranch(slot.ifElse):
      # mapIfExprToTernary normally converted this, keep blit standalone
      slot = GpuAst(kind: gpuTernary, tCond: slot.ifCond,
                    tThen: slot.ifThen, tElse: slot.ifElse)
      result.add ctx.blitExprSlot(slot.tCond, GpuType(kind: gtVoid), fnRetType)
      result.add ctx.blitExprSlot(slot.tThen, GpuType(kind: gtVoid), fnRetType)
      result.add ctx.blitExprSlot(slot.tElse, GpuType(kind: gtVoid), fnRetType)
    else:
      result = ctx.lowerIfExprStatementBranches(slot, fnRetType)
  of gpuTernary:
    # ternary branches can carry block exprs (mapIfExprToTernary), blit every branch
    # so ensureNoExprBlocks never sees one
    result.add ctx.blitExprSlot(slot.tCond, GpuType(kind: gtVoid), fnRetType)
    result.add ctx.blitExprSlot(slot.tThen, GpuType(kind: gtVoid), fnRetType)
    result.add ctx.blitExprSlot(slot.tElse, GpuType(kind: gtVoid), fnRetType)
  of gpuFor:
    result.add ctx.blitExprSlot(slot.fStart, GpuType(kind: gtVoid), fnRetType)
    result.add ctx.blitExprSlot(slot.fEnd, GpuType(kind: gtVoid), fnRetType)
  of gpuWhile:
    result = ctx.blitExprSlot(slot.wCond, GpuType(kind: gtVoid), fnRetType)
  of gpuObjConstr:
    for f in slot.ocFields.mitems:
      result.add ctx.blitExprSlot(f.value, GpuType(kind: gtVoid), fnRetType)
  of gpuArrayLit:
    for v in slot.aValues.mitems:
      result.add ctx.blitExprSlot(v, GpuType(kind: gtVoid), fnRetType)
  of gpuMaterialize:
    result = ctx.blitExprSlot(slot.mExpr, GpuType(kind: gtVoid), fnRetType)
  of gpuConstexpr:
    result = ctx.blitExprSlot(slot.cValue, GpuType(kind: gtVoid), fnRetType)
  else:
    discard

proc blitFnBody(ctx: var GpuContext, body: var GpuAst, fnRetType: GpuType) =
  case body.kind
  of gpuBlock:
    ctx.scopeSymsStack.add(ctx.currentScopeSyms)
    ctx.currentScopeSyms = @[]
    var blockPreambles: seq[seq[GpuAst]]
    for i in 0 ..< body.statements.len:
      if body.statements[i].kind == gpuBlock and body.statements[i].isExpr:
        blockPreambles.add ctx.blitExprSlot(body.statements[i], GpuType(kind: gtVoid), fnRetType)
      else:
        blockPreambles.add @[]
        let sk = body.statements[i].kind
        if sk in {gpuBlock, gpuIf, gpuFor, gpuWhile}:
          blitFnBody(ctx, body.statements[i], fnRetType)
    var newStmts: seq[GpuAst]
    var fullPreambles: seq[seq[GpuAst]]
    fullPreambles.setLen(body.statements.len)
    for i, stmt in body.statements.mpairs:
      var preamble = blockPreambles[i]
      case stmt.kind
      of gpuVar:
        if preamble.len == 0:
          preamble = ctx.blitExprSlot(stmt.vInit, stmt.vType, fnRetType)
      of gpuCall:
        let params = ctx.getFnParams(stmt.cName)
        for j in 0 ..< stmt.cArgs.len:
          let pt = if j < params.len: params[j].typ else: GpuType(kind: gtVoid)
          let p = ctx.blitExprSlot(stmt.cArgs[j], pt, fnRetType)
          preamble.add p
      of gpuReturn:
        if preamble.len == 0:
          preamble = ctx.blitExprSlot(stmt.rValue, fnRetType, fnRetType)
      of gpuAssign:
        if preamble.len == 0:
          var lhsType = GpuType(kind: gtVoid)
          if stmt.aLeft.kind == gpuIdent:
            lhsType = stmt.aLeft.symbol.typ
            if lhsType.isNil or lhsType.kind == gtVoid:
              lhsType = fnRetType
          elif stmt.aLeft.kind == gpuIndex:
            var base = stmt.aLeft.iArr
            if base.kind == gpuDeref:
              base = base.dOf
            if base.kind == gpuIdent:
              let arrTyp = base.symbol.typ
              if arrTyp != nil:
                case arrTyp.kind
                of gtPtr: lhsType = arrTyp.to
                of gtArray: lhsType = arrTyp.aTyp
                of gtUncheckedArray: lhsType = arrTyp.uaTo
                else: discard
          preamble = ctx.blitExprSlot(stmt, lhsType, fnRetType)
      of gpuBlock:
        if stmt.isExpr:
          preamble = ctx.blitExprSlot(stmt, GpuType(kind: gtVoid), fnRetType)
        elif stmt.blockLabel.len == 0:
          if stmt.statements.len == 1:
            stmt = stmt.statements[0]
      of gpuIf:
        preamble.add ctx.blitExprSlot(stmt.ifCond, GpuType(kind: gtVoid), fnRetType)
        for el in stmt.ifElifs.mitems:
          preamble.add ctx.blitExprSlot(el.cond, GpuType(kind: gtVoid), fnRetType)
      of gpuFor:
        preamble.add ctx.blitExprSlot(stmt.fStart, GpuType(kind: gtVoid), fnRetType)
        preamble.add ctx.blitExprSlot(stmt.fEnd, GpuType(kind: gtVoid), fnRetType)
      of gpuWhile:
        preamble = ctx.blitExprSlot(stmt.wCond, GpuType(kind: gtVoid), fnRetType)
      else:
        if preamble.len == 0:
          preamble = ctx.blitExprSlot(stmt, GpuType(kind: gtVoid), fnRetType)
      newStmts.add preamble
      newStmts.add stmt
      fullPreambles[i] = preamble
    body.statements = newStmts
    for stmt in body.statements:
      if stmt.kind == gpuVar:
        let sym = stmt.vName.symbol
        scopeAdd(ctx.currentScopeSyms, sym.name, sym)
    # only the fresh preamble statements re-walk, the pass already walked the originals.
    # Re-walking the whole list costs O(N x blit-depth), a blowup on deeply nested expression blocks (ceramic evalOnceAs, 2k+ blits)
    for i in 0 ..< fullPreambles.len:
      for j in 0 ..< fullPreambles[i].len:
        var pre = fullPreambles[i][j]
        if pre.kind in {gpuBlock, gpuIf, gpuFor, gpuWhile}:
          blitFnBody(ctx, pre, fnRetType)
    ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
  of gpuIf:
    blitFnBody(ctx, body.ifThen, fnRetType)
    for el in body.ifElifs.mitems:
      blitFnBody(ctx, el.body, fnRetType)
    if body.ifElse.kind != gpuDiscard:
      blitFnBody(ctx, body.ifElse, fnRetType)
  of gpuFor:
    blitFnBody(ctx, body.fBody, fnRetType)
  of gpuWhile:
    blitFnBody(ctx, body.wBody, fnRetType)
  else:
    discard
