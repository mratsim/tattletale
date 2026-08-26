## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Canonical IR expression typing and body helpers shared across passes.
## Provides the strict `getExprType` and best-effort `exprTypeBestEffort`
## type readers, the block-expression blitter and symbol-renaming helpers,
## and the single-assignment-chain resolvers used by the Vulkan passes.

import std / [macros, sequtils, sets, tables]
import ../ir/gpu_types

# ═════════════════════════════════════════════════════════════════════════
#  Expression typing
# ═════════════════════════════════════════════════════════════════════════

proc getExprType*(ctx: GpuContext; n: GpuAst): GpuType

proc getExprType*(ctx: GpuContext; n: GpuAst; fns: Table[string, GpuAst]): GpuType =
  ## Read the type of an expression from existing node fields.
  ## Errors if the node kind doesn't carry its own type (e.g. gpuDot, gpuIndex).
  ## Those cases require the caller to provide the type from context instead.
  ##
  ## `fns` resolves gpuCall to a device-fn return type (Vulkan ptr-index fold):
  ## only integer return kinds resolve, and a callee missing from the table
  ## leaves the type nil. An empty table resolves through the context fn tables instead.
  if n == nil: error "Cannot get type of nil node"
  case n.kind
  of gpuIdent:
    result = n.symbol.typ
    # Ident analogue of the gpuBinOp read-site policy: a gpuIdent whose symbol
    # carries no type must not silently fall through to the blitExprSlot fnRetType rung
    # (which absorbs it in value-returning procs), the same silent wrong-type class.
    # Error at the read site.
    if result.isNil or result.kind == gtVoid:
      error "gpuIdent with nil or void symbol type: Cannot determine type for blit temp in " &
        "block expression (nil/void symbol.typ on a gpuIdent is a defect — idents reaching getExprType must carry a type)"
  of gpuLit: result = n.lType
  of gpuCall:
    if fns.len > 0:
      # device-fn table resolution for the Vulkan ptr-index fold: only integer
      # return kinds participate in the offset coercion, and a callee missing
      # from the table leaves the type unresolved (nil) so the coercion is skipped
      if n.cName.symbol != nil and fns.hasKey(n.cName.symbol.iSym):
        let callee = fns[n.cName.symbol.iSym]
        if not callee.pRetType.isNil and
           callee.pRetType.kind in {gtUint8, gtInt16, gtUint16, gtInt32,
                                    gtUint32, gtInt64, gtUint64}:
          result = callee.pRetType
    else:
      result = ctx.getFnReturnType(n.cName)
  of gpuObjConstr: result = n.ocType
  of gpuBinOp:
    # Presence-only check by design: verifies bType is non-nil/non-void,
    # never its value correctness, which the construction sites and literal typing establish.
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
    # `iArr` may be `gpuDeref(p)`, a by-ref var-array access `(*p)[i]`, or a
    # pointer-to-scalar `p[i]`. getExprType(gpuDeref) unwraps gtPtr to the
    # POINTEE, so dispatch on the pointee: a pointer-to-array pointee is
    # gtArray and indexing yields the ELEMENT type (aTyp). A scalar pointee
    # passes through (`p[i]` is the pointee itself, pointer arithmetic).
    # Direct array-ident indexing (`gpuIndex(arr, i)`) dispatches gtArray the
    # same way. (Peeking at the deref's operand instead would return the
    # pointer, and with it the array type for ptr-to-array, not the element.)
    let arrType = ctx.getExprType(n.iArr)
    if arrType != nil:
      if n.iArr.kind == gpuDeref:
        # pointee already unwrapped, so index it
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
    # type. This only verifies the then-branch is typed, never its value.
    result = ctx.getExprType(n.tThen)
    if result.isNil or result.kind == gtVoid:
      error "gpuTernary with nil or void type: Cannot determine type for blit temp in " &
        "block expression (gpuTernary with an untyped then-branch is a defect — " &
        "if-expr branches are single typed expressions)"
  else:
    error "getExprType: unhandled node kind " & $n.kind & ". Caller must provide type from context."

proc getExprType*(ctx: GpuContext; n: GpuAst): GpuType =
  ## Read the type of an expression from existing node fields.
  ## gpuCall resolves through the context fn tables. The `fns`-taking
  ## overload resolves device-fn return types for the Vulkan ptr-index fold.
  ctx.getExprType(n, initTable[string, GpuAst]())

proc exprTypeBestEffort*(ctx: GpuContext; n: GpuAst): GpuType =
  ## Best-effort type of an expression node (nil when unknown). The printers
  ## use it to detect array-typed operands of `addr`, which need pointer
  ## decay rather than `&`.
  case n.kind
  of gpuIdent: n.symbol.typ
  of gpuLit: n.lType
  of gpuBinOp: n.bType
  of gpuPrefix: ctx.exprTypeBestEffort(n.pVal)
  of gpuCall: ctx.getFnReturnType(n.cName)
  of gpuAddr: ctx.exprTypeBestEffort(n.aOf)
  of gpuDeref: ctx.exprTypeBestEffort(n.dOf)
  of gpuIndex:
    let arrT = ctx.exprTypeBestEffort(n.iArr)
    if arrT.isNil:
      nil
    elif arrT.kind == gtArray: arrT.aTyp
    elif arrT.kind == gtPtr: arrT.to
    elif arrT.kind == gtUA: arrT.uaTo
    else: nil
  of gpuObjConstr: n.ocType
  of gpuConv: n.convTo
  of gpuCast: n.cTo
  of gpuMaterialize: ctx.exprTypeBestEffort(n.mExpr)
  of gpuDot:
    # The field's type on the parent's struct type (ptr/UA layers
    # stripped): generic field-type inference for field-access chains.
    # The value address-space resolution uses it for nested
    # pointer-field chains.
    block:
      var pT = ctx.exprTypeBestEffort(n.dParent)
      if pT != nil and pT.kind == gtPtr: pT = pT.to
      if pT != nil and pT.kind == gtUA: pT = pT.uaTo
      if pT == nil or n.dField.kind != gpuIdent or n.dField.symbol == nil:
        nil
      else:
        let fields =
          case pT.kind
          of gtObject: pT.oFields
          of gtGenericInst: pT.gFields
          else: @[]
        var found: GpuType = nil
        for f in fields:
          if f.name == n.dField.symbol.name:
            found = f.typ
            break
        found
  else: nil

# ═════════════════════════════════════════════════════════════════════════
#  Shared body helpers
# ═════════════════════════════════════════════════════════════════════════

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
        # bType is a gpuBinOp-variant field. Only read it when the tail IS a
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

proc blitFnBody*(ctx: var GpuContext; body: var GpuAst; fnRetType: GpuType) =
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
    # nested expressions via PASS 2). The preamble statements (blit scope
    # blocks and hoisted lvalue wrappers) are freshly created and must be
    # walked once. Re-walking the whole statement list here would make the
    # pass O(N x blit-depth), a blowup on deeply-nested expression blocks
    # (e.g. ceramic evalOnceAs / crd2idx).
    for i in 0 ..< fullPreambles.len:
      for j in 0 ..< fullPreambles[i].len:
        var pre = fullPreambles[i][j]
        if pre.kind in {gpuBlock, gpuIf, gpuFor, gpuWhile}:
          blitFnBody(ctx, pre, fnRetType)
    ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
  of gpuIf:
    blitFnBody(ctx, body.ifThen, fnRetType)
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
    # vName not yielded by mitems, so handle it explicitly
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

proc unwrapBlockInDot*(n: var GpuAst) =
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

# ═════════════════════════════════════════════════════════════════════════
#  Vulkan single-assignment-chain helpers
# ═════════════════════════════════════════════════════════════════════════

proc collectAssigns*(n: GpuAst; assigns, consts: var Table[string, GpuAst]) =
  ## Collects single-assignment chains: gpuAssign(ident ← rhs), gpuVar
  ## (vName ← vInit) and gpuConstexpr (cIdent ← cValue), keyed by iSym.
  case n.kind
  of gpuAssign:
    if n.aLeft.kind == gpuIdent and n.aLeft.symbol != nil:
      assigns[n.aLeft.symbol.iSym] = n.aRight
      # Nim IR quirk: the blit pass assigns the fn result through a symbol
      # whose iSym differs from the `result` slot the Return references
      # (same display name, e.g. `result` vs `result___69c5…`). The codegen
      # resolves by name, so alias the name to keep return-value resolution
      # (tainted-return fns) working.
      if n.aLeft.symbol.name == "result" and n.aLeft.symbol.iSym != "result":
        assigns["result"] = n.aRight
  of gpuVar:
    if n.vInit.kind != gpuDiscard and n.vName.symbol != nil:
      assigns[n.vName.symbol.iSym] = n.vInit
  of gpuConstexpr:
    if n.cIdent.kind == gpuIdent and n.cIdent.symbol != nil:
      consts[n.cIdent.symbol.iSym] = n.cValue
  else:
    discard
  for ch in n:
    collectAssigns(ch, assigns, consts)

proc resolveValue*(n: GpuAst; assigns, consts: Table[string, GpuAst];
                   visited: var HashSet[string]; depth: int): GpuAst =
  ## Resolves an expression through single-assignment chains (blit temps,
  ## constexprs) and folds dots over object constructions, so a leaf value
  ## becomes a pure expression over params/literals. `visited` guards
  ## against cycles. `depth` bounds pathological chains.
  if n.isNil: return n
  if depth > 512:
    raiseAssert "Vulkan: resolveValue exceeded depth (assignment cycle?)"
  case n.kind
  of gpuIdent:
    if n.symbol == nil: return n
    let i = n.symbol.iSym
    if i in assigns:
      if i in visited:
        raiseAssert "Vulkan: assignment cycle involving '" & i & "'"
      visited.incl i
      result = resolveValue(assigns[i], assigns, consts, visited, depth + 1)
      visited.excl i
    elif i in consts:
      if i in visited:
        raiseAssert "Vulkan: constexpr cycle involving '" & i & "'"
      visited.incl i
      result = resolveValue(consts[i], assigns, consts, visited, depth + 1)
      visited.excl i
    else:
      result = n
  of gpuDot:
    let parent = resolveValue(n.dParent, assigns, consts, visited, depth + 1)
    if parent.kind == gpuObjConstr and n.dField.kind == gpuIdent:
      let fname = n.dField.ident()
      for f in parent.ocFields:
        if f.name == fname:
          return resolveValue(f.value, assigns, consts, visited, depth + 1)
      raiseAssert "Vulkan: field '" & fname & "' not found in object construction"
    elif parent.kind == gpuObjConstr:
      # parent resolved to a construction but the field is not an ident
      result = n
    else:
      # parent resolved through let-chains to a param/other expr: rebuild the
      # dot on the RESOLVED parent (else leaf exprs keep stale local names
      # like `sh`/`st` that are out of scope at the call site)
      result = GpuAst(kind: gpuDot, dParent: parent, dField: n.dField)
  of gpuObjConstr:
    result = GpuAst(kind: gpuObjConstr, ocType: n.ocType)
    for f in n.ocFields:
      result.ocFields.add GpuFieldInit(name: f.name, typ: f.typ,
        value: resolveValue(f.value, assigns, consts, visited, depth + 1))
  of gpuCast:
    result = GpuAst(kind: gpuCast, cTo: n.cTo,
                    cExpr: resolveValue(n.cExpr, assigns, consts, visited, depth + 1))
  of gpuConv:
    result = GpuAst(kind: gpuConv, convTo: n.convTo,
                    convExpr: resolveValue(n.convExpr, assigns, consts, visited, depth + 1))
  of gpuBinOp:
    result = GpuAst(kind: gpuBinOp, bOp: n.bOp,
                    bLeft: resolveValue(n.bLeft, assigns, consts, visited, depth + 1),
                    bRight: resolveValue(n.bRight, assigns, consts, visited, depth + 1),
                    bIsOverloaded: n.bIsOverloaded, bType: n.bType)
  of gpuIndex:
    result = GpuAst(kind: gpuIndex,
                    iArr: resolveValue(n.iArr, assigns, consts, visited, depth + 1),
                    iIndex: resolveValue(n.iIndex, assigns, consts, visited, depth + 1))
  of gpuPrefix:
    result = GpuAst(kind: gpuPrefix, pOp: n.pOp,
                    pVal: resolveValue(n.pVal, assigns, consts, visited, depth + 1))
  of gpuArrayLit:
    result = GpuAst(kind: gpuArrayLit, aLitType: n.aLitType)
    for v in n.aValues:
      result.aValues.add resolveValue(v, assigns, consts, visited, depth + 1)
  of gpuCall:
    result = GpuAst(kind: gpuCall, cIsExpr: n.cIsExpr, cName: n.cName)
    for a in n.cArgs:
      result.cArgs.add resolveValue(a, assigns, consts, visited, depth + 1)
  of gpuLit:
    result = n
  else:
    result = n

proc substIdents*(n: GpuAst; subst: Table[string, GpuAst]): GpuAst =
  ## Replaces ident refs (by iSym) with deep-copied expressions. A deref of a
  ## substituted pointer ident collapses to the substituted expression.
  if n.isNil: return nil
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in subst:
      result = subst[n.symbol.iSym].clone()
    else:
      result = n
  of gpuDeref:
    if n.dOf.kind == gpuIdent and n.dOf.symbol != nil and n.dOf.symbol.iSym in subst:
      result = subst[n.dOf.symbol.iSym].clone()
    else:
      result = GpuAst(kind: gpuDeref, dOf: substIdents(n.dOf, subst))
  else:
    result = n.clone()
    for ch in result.mitems:
      ch = substIdents(ch, subst)
