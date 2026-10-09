## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, sets, strutils, tables]
import ../ir/gpu_types
import ./pass_datatypes

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

proc getExprType*(ctx: GpuContext; n: GpuAst): GpuType =
  ## Read the type of an expression from existing node fields.
  ## Errors if the node kind doesn't carry its own type (e.g. gpuDot, gpuIndex).
  ## Those cases require the caller to provide the type from context instead.
  if n == nil: error "Cannot get type of nil node"
  case n.kind
  of gpuIdent:
    result = n.symbol.typ
    # Ident analogue of the gpuBinOp read-site policy (SLOP-009): a gpuIdent
    # whose symbol carries no type must not silently fall through to the
    # blitExprSlot fnRetType rung (which absorbs it in value-returning
    # procs) — the same silent wrong-type class. Error at the read site.
    if result.isNil or result.kind == gtVoid:
      error "gpuIdent with nil or void symbol type: Cannot determine type for blit temp in " &
        "block expression (nil/void symbol.typ on a gpuIdent is a defect — idents reaching getExprType must carry a type)"
  of gpuLit: result = n.lType
  of gpuCall: result = ctx.getFnReturnType(n.cName)
  of gpuObjConstr: result = n.ocType
  of gpuBinOp:
    # Presence-only check by design: verifies bType is non-nil/non-void, never its
    # value correctness — that is the construction sites' + literal oracles' job.
    # A nil/void bType must not silently fall through to the blitExprSlot fnRetType
    # rung (or raise a misleading "blit temp" error on a perfectly typed binop), so
    # it is surfaced as a defect at the read site instead of returning nil.
    result = n.bType
    if result.isNil or result.kind == gtVoid:
      error "gpuBinOp with nil or void bType: Cannot determine type for blit temp in " &
        "block expression (nil/void bType on a gpuBinOp is a defect — all construction sites must populate bType)"
  of gpuConstexpr: result = n.cType
  of gpuMaterialize: result = n.mType
  of gpuConv: result = n.convTo
  of gpuCast: result = n.cTo
  of gpuBlock:
    if n.statements.len > 0:
      result = ctx.getExprType(n.statements[^1])
    else:
      error "Empty block expression"
  of gpuIndex:
    # `iArr` may be `gpuDeref(p)` — a by-ref var-array access `(*p)[i]`, or a
    # pointer-to-scalar `p[i]`. getExprType(gpuDeref) unwraps gtPtr to the
    # POINTEE, so dispatch on the pointee: a pointer-to-array pointee is
    # gtArray and indexing yields the ELEMENT type (aTyp); a scalar pointee
    # passes through (`p[i]` is the pointee itself, pointer arithmetic).
    # Direct array-ident indexing (`gpuIndex(arr, i)`) dispatches gtArray the
    # same way. (Peeking at the deref's operand instead would return the
    # pointer — and with it the array type for ptr-to-array, not the element.)
    let arrType = ctx.getExprType(n.iArr)
    if arrType != nil:
      if n.iArr.kind == gpuDeref:
        # pointee already unwrapped — index it
        case arrType.kind
        of gtPtr: result = arrType.to      # ptr-to-ptr: (*p) is ptr T
        of gtArray: result = arrType.aTyp  # (*p)[i] → element type
        of gtUA: result = arrType.uaTo
        else: result = arrType             # scalar pointee: p[i] is the pointee
      else:
        case arrType.kind
        of gtPtr: result = arrType.to
        of gtArray: result = arrType.aTyp
        of gtUA: result = arrType.uaTo
        else:
          error "getExprType(gpuIndex): cannot get element type of " & $arrType.kind
  of gpuDeref:
    let dType = ctx.getExprType(n.dOf)
    if dType != nil and dType.kind == gtPtr:
      result = dType.to       # by-ref params: ident typed as ptr, deref is the pointee
    else:
      result = dType
  of gpuDot:
    let parentType = ctx.getExprType(n.dParent)
    if parentType != nil and parentType.kind in {gtObject, gtGenericInst}:
      let fields = if parentType.kind == gtObject: parentType.oFields else: parentType.gFields
      for f in fields:
        if f.name == n.dField.symbol.name:
          result = f.typ
          break
    if result.isNil:
      error "getExprType(gpuDot): field '" & n.dField.symbol.name & "' not found in " & $n.dParent.kind
  of gpuAddr:
    result = ctx.getExprType(n.aOf)
  of gpuTernary:
    # Presence-only check by design (same convention as gpuBinOp): the if-expr
    # construction guarantees both branches carry the same single-expression
    # type; this only verifies the then-branch is typed, never its value.
    result = ctx.getExprType(n.tThen)
    if result.isNil or result.kind == gtVoid:
      error "gpuTernary with nil or void type: Cannot determine type for blit temp in " &
        "block expression (gpuTernary with an untyped then-branch is a defect — " &
        "if-expr branches are single typed expressions)"
  else:
    error "getExprType: unhandled node kind " & $n.kind & ". Caller must provide type from context."

proc blitExprSlot(ctx: var GpuContext; slot: var GpuAst; blitType: GpuType; fnRetType: GpuType): seq[GpuAst] =
  ## Process an expression slot. If `slot` is a gpuBlock(isExpr: true),
  ## replace it with a blit temp reference and return preamble statements.
  if slot.kind == gpuBlock and slot.isExpr:
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
        # bType is a gpuBinOp-variant field — only read it when the tail IS a
        # gpuBinOp, otherwise the case-object access raises a FieldDefect.
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
  of gpuTemplateCall:
    for i in 0 ..< slot.tcArgs.len:
      let p = ctx.blitExprSlot(slot.tcArgs[i], GpuType(kind: gtVoid), fnRetType)
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
          of gtUA: lhsType = arrTyp.uaTo
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
            for p in vBlit: inlined.add p
          inlined.add s
        result = inlined
        slot.aLeft = lastStmt
        # Recompute lhsType from the actual lvalue
        if slot.aLeft.kind == gpuIdent:
          lhsType = slot.aLeft.symbol.typ
      else:
        error "Empty block expression as lvalue"
      result.add ctx.blitExprSlot(slot.aRight, lhsType, fnRetType)
      return
    result.add ctx.blitExprSlot(slot.aLeft, GpuType(kind: gtVoid), fnRetType)
    # The RHS blit appends to the LHS preamble: assigning instead would
    # drop the preamble, whose temps the assignment statement references
    # (the undeclared `_blit_N` failure).
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
    result.add ctx.blitExprSlot(slot.ifCond, GpuType(kind: gtVoid), fnRetType)
    for el in slot.ifElifs.mitems:
      result.add ctx.blitExprSlot(el.cond, GpuType(kind: gtVoid), fnRetType)
  of gpuTernary:
    # Ternary branches can carry block expressions (lowerIfExpr lowers if-exprs
    # whose branches are blocks). Blit every branch so no block survives to codegen,
    # where ensureNoExprBlocks rejects it. The ceramic gemm_cta tile-view templates
    # expand into such if-expr chains.
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

proc collectSyms(n: GpuAst; syms: var HashSet[string])
proc renameSymsInTree(n: var GpuAst; oldName, newName: string)
proc hoistLvalueVars(stmts: var seq[GpuAst])
proc dedupVarNames*(stmts: var seq[GpuAst])

proc blitFnBody(ctx: var GpuContext; body: var GpuAst; fnRetType: GpuType) =
  ## Walk a function body tree, blitting all gpuBlock(isExpr: true) nodes.
  case body.kind
  of gpuBlock:
    # RE loop: blit standalone expression blocks, recurse into nested blocks
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
      of gpuTemplateCall:
        for j in 0 ..< stmt.tcArgs.len:
          let p = ctx.blitExprSlot(stmt.tcArgs[j], GpuType(kind: gtVoid), fnRetType)
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
                of gtUA: lhsType = arrTyp.uaTo
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
    # Register all gpuVar symbols in this block's scope table
    for stmt in body.statements:
      if stmt.kind == gpuVar:
        let sym = stmt.vName.symbol
        scopeAdd(ctx.currentScopeSyms, sym.name, sym)
    # Recursively process the NEW content produced by blitting. The original
    # statements were already fully processed (nested blocks via PASS 1,
    # nested expressions via PASS 2); the preamble statements — blit scope
    # blocks and hoisted lvalue wrappers — are freshly created and must be
    # walked once. Re-walking the whole statement list here would make the
    # pass O(N x blit-depth): a blowup on deeply-nested expression blocks
    # (e.g. ceramic evalOnceAs / crd2idx, 2k+ blits => 100M+ visits).
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

proc collectSyms(n: GpuAst; syms: var HashSet[string]) =
  ## Collect all iSym values referenced via gpuIdent in an AST subtree.
  if n == nil: return
  case n.kind
  of gpuIdent:
    syms.incl n.symbol.iSym
  else:
    for child in n.items:
      collectSyms(child, syms)

proc renameSymsInTree(n: var GpuAst; oldName, newName: string) =
  ## Rename all gpuIdent references with iName == oldName to newName,
  ## including gpuVar.vName (which is not yielded by mitems).
  if n == nil: return
  case n.kind
  of gpuIdent:
    if n.symbol.name == oldName:
      n.symbol.name = newName
      n.symbol.iSym = newName
  of gpuVar:
    # vName not yielded by mitems — handle explicitly
    if n.vName.symbol.name == oldName:
      n.vName.symbol.name = newName
      n.vName.symbol.iSym = newName
    renameSymsInTree(n.vInit, oldName, newName)
  else:
    for child in n.mitems:
      renameSymsInTree(child, oldName, newName)

proc hoistLvalueVars(stmts: var seq[GpuAst]) =
  ## Hoist gpuVar declarations out of `_lvalue`-labeled gpuBlock siblings
  ## that are referenced by sibling statements. Variables declared inside
  ## the `_lvalue` block but needed outside are moved above the block.

  proc declaredVars(scope: GpuAst): OrderedTable[string, int] =
    ## Collect variable names declared at top level of a scope block.
    for j, s in scope.statements:
      if s.kind == gpuVar:
        result[s.vName.symbol.name] = j

  proc referencedBySiblings(stmts: seq[GpuAst]; scopeIdx: int;
                           varNames: seq[string]): HashSet[string] =
    ## Find vars declared in scope that are referenced by any sibling statement.
    for j in 0 ..< stmts.len:
      if j != scopeIdx:
        var refs: HashSet[string]
        collectSyms(stmts[j], refs)
        for v in varNames:
          if v in refs:
            result.incl v

  proc expandTransitives(scope: GpuAst; varIdx: OrderedTable[string, int];
                         toHoist: var HashSet[string]) =
    ## Add vars that hoisted vars depend on via their vInit.
    var changed = true
    while changed:
      changed = false
      for vname, idx in varIdx.pairs:
        if vname in toHoist:
          var refd: HashSet[string]
          if scope.statements[idx].kind == gpuVar:
            collectSyms(scope.statements[idx].vInit, refd)
          for r in refd:
            if r in varIdx and r notin toHoist:
              toHoist.incl r
              changed = true

  proc splitScope(scope: GpuAst; toHoist: HashSet[string]):
      tuple[hoisted: seq[GpuAst], remaining: seq[GpuAst]] =
    ## Partition scope statements into those being hoisted vs kept.
    for s in scope.statements:
      if s.kind == gpuVar and s.vName.symbol.name in toHoist:
        result.hoisted.add s
      else:
        result.remaining.add s

  proc spliceBefore[T](xs: var seq[T]; idx: int; prefix: seq[T]) =
    ## Insert `prefix` elements before position `idx` in `xs`.
    var outSeq: seq[T]
    for i in 0 ..< idx:
      outSeq.add xs[i]
    outSeq.add prefix
    for i in idx ..< xs.len:
      outSeq.add xs[i]
    xs = outSeq

  var i = 0
  while i < stmts.len:
    if stmts[i].kind == gpuBlock and stmts[i].blockLabel == "_lvalue":
      let scope = stmts[i]
      let varIdx = declaredVars(scope)
      if varIdx.len > 0:
        var hoistedSet = referencedBySiblings(stmts, i, toSeq(varIdx.keys))
        if hoistedSet.len > 0:
          expandTransitives(scope, varIdx, hoistedSet)
          let (hoistedStmts, remaining) = splitScope(scope, hoistedSet)
          scope.statements = remaining
          spliceBefore(stmts, i, hoistedStmts)
          i += hoistedStmts.len
    inc i

proc dedupVarNames*(stmts: var seq[GpuAst]) =
  ## Rename duplicate gpuVar declarations across sibling gpuBlock statements.
  ## Handles `{.inject.}` variables from sequential block: template expansions
  ## that would collide without C++ scope isolation.
  ## Uses iName (the short, codegen-facing name) for collision detection.
  var seenNames: Table[string, int]
  for i in 0 ..< stmts.len:
    if stmts[i].kind == gpuBlock:
      var localRenames: seq[(string, string)]
      for s in stmts[i].statements:
        if s.kind == gpuVar:
          let name = s.vName.symbol.name
          if name in seenNames:
            seenNames[name] += 1
            let newName = name & "_" & $seenNames[name]
            localRenames.add (name, newName)
          else:
            seenNames[name] = 0
      for (oldName, newName) in localRenames:
        renameSymsInTree(stmts[i], oldName, newName)
    elif stmts[i].kind == gpuVar:
      let name = stmts[i].vName.symbol.name
      if name in seenNames:
        seenNames[name] += 1
        let newName = name & "_" & $seenNames[name]
        renameSymsInTree(stmts[i], name, newName)
      else:
        seenNames[name] = 0

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

proc structFields(t: GpuType): seq[GpuTypeField] =
  ## Fields of a struct type, declaration order.
  case t.kind
  of gtObject: t.oFields
  of gtGenericInst: t.gFields
  else: @[]

proc containsPtrDeep(t: GpuType): bool =
  ## True when a gtPtr sits anywhere in the type tree, array element
  ## types and unchecked-array element types included.
  case t.kind
  of gtPtr: result = true
  of gtArray:
    result = containsPtrDeep(t.aTyp)
  of gtUA:
    result = containsPtrDeep(t.uaTo)
  of gtObject:
    for f in t.oFields:
      if containsPtrDeep(f.typ): return true
  of gtGenericInst:
    for f in t.gFields:
      if containsPtrDeep(f.typ): return true
  else: discard

proc isScalarGpuKind(k: GpuTypeKind): bool =
  k in {gtBool, gtUint8, gtUint16, gtInt16, gtUint32, gtInt32, gtUint64,
        gtInt64, gtFloat32, gtFloat64, gtFloat16, gtBf16, gtSize_t}

proc newExplodedParam(name: string, typ: GpuType, addressSpace: AddressSpace): GpuParam =
  let sym = newSymbol(name, iSym = name, typ = typ, symKind = gsDeviceKernelParam)
  GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym), typ: typ,
           addressSpace: addressSpace, passByRef: false)

proc explodeTypeInto(t: GpuType, path: seq[string], name: string,
                     params: var seq[GpuParam], usedNames: var HashSet[string],
                     leaves: var Table[seq[string], GpuAst]) =
  ## Lowers one field type into exploded kernel params, depth-first in field
  ## order:
  ## - pointer field -> ptr param (`device T*`, SSBO, storage)
  ## - scalar/enum field or fixed-size pointer-free array -> by-value
  ##   param (constant-ref, push const or read-only storage)
  ## - pointer-free value struct -> by-value struct param
  ## - struct carrying pointers -> recursion into its fields
  ## Precondition: `name` is unique among the struct's exploded leaves.
  ## `leaves` maps each field path from the param root to its leaf param ident,
  ## keyed as @[rootISym] & path.
  doAssert name notin usedNames,
    "explodePointerStructParams: exploded param name collision on '" & name & "'"
  doAssert path.len > 0
  usedNames.incl name
  if t.kind == gtPtr:
    # Type verbatim: the UA inner type survives for WGSL's runtime array,
    # MSL decays it to `device T*`.
    params.add newExplodedParam(name, t, asDevice)
    leaves[path] = params[^1].ident
  elif isScalarGpuKind(t.kind):
    params.add newExplodedParam(name, t, asRMEM)
    leaves[path] = params[^1].ident
  elif t.kind in {gtObject, gtGenericInst}:
    if containsPtrDeep(t):
      for f in structFields(t):
        explodeTypeInto(f.typ, path & f.name, name & "_" & f.name, params,
                        usedNames, leaves)
    else:
      params.add newExplodedParam(name, t, asRMEM)
      leaves[path] = params[^1].ident
  elif t.kind == gtArray:
    # Fixed-size array field: one by-value param, never per-element params.
    # Pointer-bearing elements carry no value representation: the explosion
    # fails loudly on them.
    doAssert not containsPtrDeep(t.aTyp),
      "explodePointerStructParams: array field '" & name &
      "' carries pointer-bearing elements"
    params.add newExplodedParam(name, t, asRMEM)
    leaves[path] = params[^1].ident
  else:
    doAssert false, "explodePointerStructParams: unsupported field type " &
      $t.kind & " at '" & name & "'"

proc rewriteExplodedDots(n: var GpuAst, roots: HashSet[string], leaves: Table[seq[string], GpuAst]) =
  ## Rewrites `param.field...` chains rooted at an exploded param into
  ## exploded leaf param idents:
  ## - whole chain when the leaf path covers it (ptr, scalar, value struct)
  ## - matched prefix otherwise, fields of an exploded value struct:
  ##   `dst.layout.shape` -> `dst_layout.shape`
  case n.kind
  of gpuDot:
    var fields: seq[string]
    var root = n
    while root.kind == gpuDot:
      fields.insert(root.dField.ident())
      root = root.dParent
    if root.kind == gpuIdent and root.symbol != nil and
       root.symbol.iSym in roots:
      var matched = -1
      for k in countdown(fields.len, 1):
        if leaves.hasKey(@[root.symbol.iSym] & fields[0 ..< k]):
          matched = k
          break
      doAssert matched > 0,
        "explodePointerStructParams: no exploded leaf for field chain '" &
        fields.join(".") & "' of param '" & root.symbol.iSym & "' (whole use)"
      var replacement = leaves[@[root.symbol.iSym] & fields[0 ..< matched]]
      for f in fields[matched ..< fields.len]:
        let fsym = newSymbol(f, iSym = f, typ = nil, symKind = gsNone)
        replacement = GpuAst(kind: gpuDot, dParent: replacement,
                             dField: GpuAst(kind: gpuIdent, symbol: fsym))
      n = replacement
    else:
      # the chain roots elsewhere (an indexed or called expression): descend
      # into the root, its children can still reference an exploded param
      rewriteExplodedDots(root, roots, leaves)
  else:
    for child in mitems(n):
      rewriteExplodedDots(child, roots, leaves)

proc explodePointerStructParamsImpl(ctx: var GpuContext) =
  ## Kernel params of struct types carrying pointer fields, nested anywhere,
  ## explode depth-first in field order into ptr, scalar,
  ## pointer-free-value params. Body chains `param.field...` rewrite onto
  ## the exploded leaf params:
  ## - no reconstructed local: backends without pointer-bearing struct values
  ##   (GLSL, WGSL) emit the kernel directly
  ## - the struct never crosses the launch boundary
  var done = initHashSet[string]()
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind != gpuProc or not fn.pAttributes.contains(attGlobal):
      continue
    let key = fn.pName.symbol.iSym
    if key in done:
      continue
    done.incl key
    var newParams: seq[GpuParam]
    var usedNames = initHashSet[string]()
    var leaves = initTable[seq[string], GpuAst]()
    var roots = initHashSet[string]()
    var exploded = false
    for p in fn.pParams:
      if p.typ != nil and p.typ.kind in {gtObject, gtGenericInst} and
         containsPtrDeep(p.typ):
        exploded = true
        roots.incl p.ident.symbol.iSym
        explodeTypeInto(p.typ, @[p.ident.symbol.iSym], p.ident.ident(),
                        newParams, usedNames, leaves)
      else:
        newParams.add p
    if exploded:
      fn.pParams = newParams
      rewriteExplodedDots(fn.pBody, roots, leaves)

proc registerExplodePointerStructParams*(reg: var PassRegistry) =
  ## Registers the `explodePointerStructParams` pass on `reg`.
  ## Wiring contract:
  ## - Metal, Vulkan, WebGPU only
  ## - CUDA and OpenCL take pointer-bearing structs natively, never wire
  reg.register("explodePointerStructParams", pkTransform, phaseMain,
    "Explodes kernel params of structs carrying (nested) pointer fields into ptr/scalar/value params and rewrites body accesses",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      ctx.explodePointerStructParamsImpl()
  )

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
