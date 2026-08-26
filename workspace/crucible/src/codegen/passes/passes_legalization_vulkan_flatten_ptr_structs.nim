## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan IR legalization pass 2: flatten struct values that carry pointer
## fields into leaf scalars and ptr-leaf expressions (GLSL structs cannot
## hold pointer members).
##
## Tainted vars split into leaf vars, value params split into leaf params,
## struct-returning fns resolve to per-leaf return expressions over their
## params, and dot-access chains are rewritten onto the leaves. Tainted
## struct type defs are removed. Runs after pass 1.

import std/[sets, tables]
import ../ir/gpu_types
import ./passes_legalization_vulkan_helpers
import ./passes_utils

# ═════════════════════════════════════════════════════════════════════════
#  Pass 2: vulkanFlattenStructPtrValues
# ═════════════════════════════════════════════════════════════════════════

type
  ## Kind of a flattened leaf: a value backed by a var/param ident, or a
  ## pointer expression.
  LeafKind* = enum
    lkValue      ## scalar or plain-struct leaf, backed by a var or param ident
    lkPtr        ## pointer leaf, backed by an expression (SSBO ref / ptr arith)

  ## One leaf of a flattened tainted struct value. `path` and `typ` locate
  ## the leaf in the struct tree. `name` is the ident for lkValue leaves,
  ## and `expr` the expression for lkPtr leaves.
  FlattenedLeaf* = object
    path*: seq[string]
    typ*: GpuType
    kind*: LeafKind
    name*: string        ## ident name for lkValue leaves
    expr*: GpuAst        ## expression for lkPtr leaves

  LeafMap* = Table[string, seq[FlattenedLeaf]]   ## var/param iSym → leaves

proc mkValueLeaf(path: seq[string]; typ: GpuType; name: string): FlattenedLeaf =
  FlattenedLeaf(path: path, typ: typ, kind: lkValue, name: name)

proc mkPtrLeaf(path: seq[string]; typ: GpuType; expr: GpuAst): FlattenedLeaf =
  FlattenedLeaf(path: path, typ: typ, kind: lkPtr, expr: expr)

proc flattenedParamName(base: string; path: seq[string]): string =
  leafName(base, path)

proc flattenStructPtrValues*(ctx: var GpuContext) =
  ## Pass 2 (vulkanFlattenStructPtrValues): flatten struct-with-ptr-field
  ## values (see module header).

  let ctxL = ctx  # closures cannot capture the var param, so alias it
  let reachable = reachableFns(ctx)
  var byISym = initTable[string, GpuAst]()
  for fn in reachable:
    byISym[fn.pName.symbol.iSym] = fn

  # ── Phase 1: per-fn flattened param lists + return-leaf maps ───────────
  # newParams[iSym] = seq[(origParamIdx, leafPath, leafTyp)] per new param
  var newParams = initTable[string, seq[tuple[origIdx: int, leaf: FlattenedLeaf]]]()
  # returnLeaves[iSym] = seq[(path, typ, expr-over-params)] for tainted returns
  var returnLeaves = initTable[string, seq[FlattenedLeaf]]()
  var taintedReturnFns = initHashSet[string]()

  for fn in reachable:
    let fnISym = fn.pName.symbol.iSym
    var nps: seq[tuple[origIdx: int, leaf: FlattenedLeaf]]
    for i, p in fn.pParams:
      if isTaintedStruct(p.typ):
        let leaves = taintedLeaves(p.typ)
        for lf in leaves:
          let lname = flattenedParamName(p.ident.ident(), lf.path)
          if isPtrType(lf.typ):
            nps.add (i, mkPtrLeaf(lf.path, lf.typ, nil))  # ptr param, bound by pass 3
            nps[^1].leaf.name = lname
          else:
            nps.add (i, mkValueLeaf(lf.path, lf.typ, lname))
      else:
        # plain param: keep as-is (mark origIdx = i, leaf = identity)
        nps.add (i, mkValueLeaf(@[], p.typ, p.ident.ident()))
    newParams[fnISym] = nps
    # tainted return?
    if isTaintedStruct(fn.pRetType):
      taintedReturnFns.incl fnISym
      var assigns, consts = initTable[string, GpuAst]()
      if not fn.pBody.isNil:
        collectAssigns(fn.pBody, assigns, consts)
      var retVal: GpuAst
      proc findReturn(n: GpuAst) =
        if not retVal.isNil: return
        case n.kind
        of gpuReturn:
          retVal = n.rValue
        else:
          for ch in n:
            if retVal.isNil: findReturn(ch)
      if not fn.pBody.isNil:
        findReturn(fn.pBody)
      if retVal.isNil:
        raiseAssert "Vulkan: tainted-returning fn '" & fn.pName.ident() &
          "' has no return statement"
      var visited = initHashSet[string]()
      let constr = resolveValue(retVal, assigns, consts, visited, 0)
      if constr.kind != gpuObjConstr:
        raiseAssert "Vulkan: tainted-returning fn '" & fn.pName.ident() &
          "' return does not resolve to an object construction"
      let leaves = taintedLeaves(fn.pRetType)
      for lf in leaves:
        # extract the leaf expression from the construction
        var e = constr
        var ok = true
        for fname in lf.path:
          if e.kind != gpuObjConstr:
            ok = false
            break
          var found = false
          for f in e.ocFields:
            if f.name == fname:
              e = f.value
              found = true
              break
          if not found:
            ok = false
            break
        if not ok:
          raiseAssert "Vulkan: cannot extract return leaf " & $lf.path &
            " from fn '" & fn.pName.ident() & "'"
        visited = initHashSet[string]()
        let leafExpr = resolveValue(e, assigns, consts, visited, 0)
        returnLeaves.mgetOrPut(fnISym, @[]).add mkPtrLeaf(lf.path, lf.typ, leafExpr)

  # ── Phase 2: rewrite each reachable fn body ────────────────────────────
  # per-fn state: leaf maps for its flattened vars and params
  var
    varMaps = initTable[string, LeafMap]()       # fn iSym → var iSym → leaves
    paramMaps = initTable[string, LeafMap]()     # fn iSym → param iSym → leaves
    assignTables = initTable[string, Table[string, GpuAst]]()
    constTables = initTable[string, Table[string, GpuAst]]()

  for fn in reachable:
    let fnISym = fn.pName.symbol.iSym
    var assigns, consts = initTable[string, GpuAst]()
    if not fn.pBody.isNil:
      collectAssigns(fn.pBody, assigns, consts)
    assignTables[fnISym] = assigns
    constTables[fnISym] = consts

  # rewriteDot: rewrite a dot chain rooted at a flattened var/param
  proc rewriteDot(n: GpuAst; vmap, pmap: LeafMap; assigns, consts: Table[string, GpuAst]): GpuAst =
    ## Returns the rewritten node, or nil when the chain is not tainted-rooted.
    if n.kind != gpuDot: return nil
    # walk to the base ident, collecting the path
    var path: seq[string]
    var base = n
    while base.kind == gpuDot:
      if base.dField.kind == gpuIdent:
        path.insert(base.dField.ident(), 0)
      else:
        return nil
      base = base.dParent
    if base.kind == gpuIdent and base.symbol != nil:
      let i = base.symbol.iSym
      var leaves: seq[FlattenedLeaf]
      if i in vmap: leaves = vmap[i]
      elif i in pmap: leaves = pmap[i]
      else: return nil
      # find the longest leaf-path prefix of `path`
      var best: FlattenedLeaf
      var bestLen = -1
      for lf in leaves:
        if lf.path.len > bestLen and lf.path.len <= path.len:
          var match = true
          for j in 0 ..< lf.path.len:
            if lf.path[j] != path[j]:
              match = false
              break
          if match:
            best = lf
            bestLen = lf.path.len
      if bestLen < 0:
        return nil
      if bestLen == path.len:
        # full leaf: value → ident, ptr → expr (or the leaf-param ident when
        # the ptr is a device-fn param bound later by pass 3)
        if best.kind == lkValue:
          result = newGpuIdent(best.name)
          result.symbol.typ = best.typ
        elif not best.expr.isNil:
          result = best.expr.clone()
        else:
          result = newGpuIdent(best.name)
          result.symbol.typ = best.typ
      else:
        # plain-struct leaf + remaining dots on the leaf ident
        if best.kind != lkValue:
          return nil
        var acc: GpuAst = newGpuIdent(best.name)
        acc.symbol.typ = best.typ
        for j in bestLen ..< path.len:
          var d = GpuAst(kind: gpuDot, dParent: acc)
          d.dField = newGpuIdent(path[j])
          acc = d
        result = acc
    else:
      result = nil

  # leafValueOf: the value of a tainted value expression at a leaf path
  proc leafValueOf(e0: GpuAst; path: seq[string]; fnISym: string;
                   vmap, pmap: LeafMap): GpuAst =
    var e = e0
    let assigns = assignTables.getOrDefault(fnISym)
    let consts = constTables.getOrDefault(fnISym)
    # ident with a leaf map → leaf value directly
    if e.kind == gpuIdent and e.symbol != nil:
      let i = e.symbol.iSym
      if i in vmap or i in pmap:
        let leaves = if i in vmap: vmap[i] else: pmap[i]
        for lf in leaves:
          if lf.path == path:
            if lf.kind == lkValue:
              result = newGpuIdent(lf.name)
              result.symbol.typ = lf.typ
            elif not lf.expr.isNil:
              result = lf.expr.clone()
            else:
              # ptr-leaf param bound by pass 3 → the leaf-param ident
              result = newGpuIdent(lf.name)
              result.symbol.typ = lf.typ
            return
        raiseAssert "Vulkan: leaf " & $path & " not found for '" & i & "'"
    # call to a tainted-returning fn → return-leaf map, params substituted
    if e.kind == gpuCall and e.cName.symbol != nil and
       e.cName.symbol.iSym in returnLeaves:
      let fnISym2 = e.cName.symbol.iSym
      let callee = byISym[fnISym2]
      var subst = initTable[string, GpuAst]()
      for i, p in callee.pParams:
        if i < e.cArgs.len:
          subst[p.ident.symbol.iSym] = e.cArgs[i]
      for lf in returnLeaves[fnISym2]:
        if lf.path == path:
          var r = substIdents(lf.expr, subst)
          # fold the substituted call-site args (tuple constructions) into
          # the leaf, e.g. Dot((1,1,N,K), F0) → 1, using the CALLER's
          # assigns/consts so remaining idents resolve in caller scope
          var vis = initHashSet[string]()
          r = resolveValue(r, assigns, consts, vis, 0)
          result = r
          return
      raiseAssert "Vulkan: return leaf " & $path & " not found for fn '" &
        e.cName.ident() & "'"
    # resolve through assigns/consts to an object construction, then drill
    var visited = initHashSet[string]()
    e = resolveValue(e, assigns, consts, visited, 0)
    for fname in path:
      if e.kind == gpuObjConstr:
        var found = false
        for f in e.ocFields:
          if f.name == fname:
            e = f.value
            found = true
            break
        if not found:
          raiseAssert "Vulkan: field '" & fname & "' missing in construction while extracting " & $path
        visited = initHashSet[string]()
        e = resolveValue(e, assigns, consts, visited, 0)
      elif e.kind == gpuIdent and e.symbol != nil:
        let i = e.symbol.iSym
        let leaves = if i in vmap: vmap[i] else: pmap[i]
        for lf in leaves:
          if lf.path == path:
            if lf.kind == lkValue:
              result = newGpuIdent(lf.name)
              result.symbol.typ = lf.typ
            elif not lf.expr.isNil:
              result = lf.expr.clone()
            else:
              # ptr-leaf param bound by pass 3 → the leaf-param ident
              result = newGpuIdent(lf.name)
              result.symbol.typ = lf.typ
            return
        raiseAssert "Vulkan: leaf " & $path & " not found for '" & i & "'"
      else:
        raiseAssert "Vulkan: cannot extract leaf " & $path & " (got " & $e.kind & ")"
    result = e

  # lowerPtrIndex: Index(cast[ptr](uint64(base) + uint64(off)*sizeof), i) → base[off + i]
  proc lowerPtrIndex(arr: GpuAst; idx: GpuAst): GpuAst =
    ## Recognizes the `+%` pointer-arithmetic shape (ptr = cast[ptr T](
    ## uint64(base) + uint64(off) * sizeof(T))) and lowers element indexing
    ## over it to SSBO indexing with the offset folded into the index.
    result = foldPtrIndexToElement(arr, idx, ctxL, byISym)
    if result.isNil:
      result = GpuAst(kind: gpuIndex, iArr: arr, iIndex: idx)

  # The body rewriter
  proc rewriteBody(fnISym: string; body: var GpuAst; vmap, pmap: LeafMap) =
    ## In-place rewrite of one fn body. Handles:
    ## - tainted var flattening (vInit = construction / tainted-returning call)
    ## - tainted assign elimination (blit temps)
    ## - dot-chain rewrite on flattened vars/params
    ## - call-arg expansion to the callee's flattened signature
    ## - ptr-index folding over pointer-arithmetic chains
    # Nested closures may capture locals, not var params: rebind to locals.
    # The caller never reads the maps back, so value params are sufficient.
    var vmapL = vmap
    var pmapL = pmap
    let assigns = assignTables[fnISym]
    let consts = constTables[fnISym]

    proc rewriteExpr(n: var GpuAst) =
      ## Rewrite an expression node in place (dots, indexes, nested calls).
      case n.kind
      of gpuDot:
        let r = rewriteDot(n, vmapL, pmapL, assigns, consts)
        if not r.isNil:
          n = r
          # The replacement (a ptr-leaf expr / leaf ident / plain-struct dot)
          # may itself contain unrewritten dots (e.g. the inner Dot(gl, data)
          # of a pointer-arith chain). ALWAYS re-rewrite the replacement.
          rewriteExpr(n)
        else:
          for ch in n.mitems:
            rewriteExpr(ch)
      of gpuIndex:
        # rewrite the index expr first, then the array expr, and fold
        # ptr-arith bases
        rewriteExpr(n.iIndex)
        var arrWasDot = n.iArr.kind == gpuDot
        rewriteExpr(n.iArr)
        if arrWasDot or n.iArr.kind == gpuCast:
          # ptr-leaf expression or cast chain as array base → fold to SSBO index
          let folded = lowerPtrIndex(n.iArr, n.iIndex)
          if not folded.isNil:
            n = folded
      of gpuCall:
        # expand args per the callee's flattened signature
        if n.cName.symbol != nil and n.cName.symbol.iSym in newParams:
          let calleeISym = n.cName.symbol.iSym
          let calleeParams = newParams[calleeISym]
          var newArgs: seq[GpuAst]
          var origIdx = 0
          var pi = 0
          while pi < calleeParams.len:
            let (oi, leaf) = calleeParams[pi]
            # advance origIdx to oi (original args between are unchanged)
            while origIdx < oi:
              if origIdx < n.cArgs.len:
                var a = n.cArgs[origIdx]
                rewriteExpr(a)
                newArgs.add a
              inc origIdx
            if leaf.path.len == 0 and leaf.kind == lkValue and
               leaf.name != "":
              # plain param passthrough marker (created with name = param name)
              if origIdx < n.cArgs.len:
                var a = n.cArgs[origIdx]
                rewriteExpr(a)
                newArgs.add a
              inc origIdx
              inc pi
            else:
              # a leaf of a flattened tainted param: the group shares one
              # original arg (same oi). Consume it once, emit all leaves
              let groupOi = oi
              while pi < calleeParams.len and calleeParams[pi].origIdx == groupOi:
                if origIdx >= n.cArgs.len:
                  raiseAssert "Vulkan: too few args calling '" & calleeISym & "'"
                var lv = leafValueOf(n.cArgs[origIdx], calleeParams[pi].leaf.path,
                                     fnISym, vmapL, pmapL)
                rewriteExpr(lv)
                newArgs.add lv
                inc pi
              inc origIdx
          # trailing original args (shouldn't happen, but be safe)
          while origIdx < n.cArgs.len:
            var a = n.cArgs[origIdx]
            rewriteExpr(a)
            newArgs.add a
            inc origIdx
          n.cArgs = newArgs
        else:
          for a in n.cArgs.mitems:
            rewriteExpr(a)
      of gpuObjConstr:
        for f in n.ocFields.mitems:
          rewriteExpr(f.value)
      of gpuCast:
        rewriteExpr(n.cExpr)
      of gpuConv:
        rewriteExpr(n.convExpr)
      of gpuBinOp:
        rewriteExpr(n.bLeft)
        rewriteExpr(n.bRight)
      of gpuAddr:
        rewriteExpr(n.aOf)
      of gpuDeref:
        rewriteExpr(n.dOf)
      of gpuArrayLit:
        for v in n.aValues.mitems:
          rewriteExpr(v)
      of gpuPrefix:
        rewriteExpr(n.pVal)
      else:
        discard

    # statement-level rewrite, returns true when the node should be dropped
    proc rewriteStmt(n: var GpuAst): bool =
      case n.kind
      of gpuVar:
        if isTaintedStruct(n.vType):
          # flatten the tainted var into leaf vars
          let vISym = n.vName.symbol.iSym
          let leaves = taintedLeaves(n.vType)
          var leafList: seq[FlattenedLeaf]
          var replacements: seq[GpuAst]
          if n.vInit.kind == gpuDiscard:
            # blit temp: value comes from a later assign, so drop the decl
            vmapL[vISym] = @[]
            return true
          for lf in leaves:
            var lv: GpuAst
            if n.vInit.kind == gpuCall:
              lv = leafValueOf(n.vInit, lf.path, fnISym, vmapL, pmapL)
            elif n.vInit.kind == gpuObjConstr:
              lv = leafValueOf(n.vInit, lf.path, fnISym, vmapL, pmapL)
            else:
              raiseAssert "Vulkan: tainted var '" & n.vName.ident() &
                "' has an unsupported init (" & $n.vInit.kind & ")"
            # the leaf expr may contain unrewritten dots, so always re-rewrite
            rewriteExpr(lv)
            if isPtrType(lf.typ):
              # ptr leaf: no local var, just an expression mapping
              leafList.add mkPtrLeaf(lf.path, lf.typ, lv)
            else:
              let lname = leafName(n.vName.ident(), lf.path)
              leafList.add mkValueLeaf(lf.path, lf.typ, lname)
              var lvNode = GpuAst(kind: gpuVar, vName: newGpuIdent(lname),
                                  vType: lf.typ, vInit: lv,
                                  vMutable: false, addressSpace: asRMEM)
              lvNode.vName.symbol.typ = lf.typ
              replacements.add lvNode
          vmapL[vISym] = leafList
          # replace the var node with the leaf var declarations
          if replacements.len == 0:
            return true
          var blk = GpuAst(kind: gpuBlock, statements: replacements)
          n = blk
          return false
        else:
          rewriteExpr(n.vInit)
          return false
      of gpuAssign:
        if n.aLeft.kind == gpuIdent and isTaintedStruct(n.aLeft.symbol.typ):
          # assignment to a tainted var: update its leaves. Value leaves are
          # ASSIGNED when already declared (re-assignment must use the new
          # value, HIDN-A-001: the old code dropped the assign, leaving the
          # stale first value) or DECLARED for the blit-temp pattern (gpuVar
          # with discard init). Ptr leaves update the expression map.
          let vISym = n.aLeft.symbol.iSym
          let leaves = taintedLeaves(n.aLeft.symbol.typ)
          var leafList: seq[FlattenedLeaf]
          var replacements: seq[GpuAst]
          let prevLeaves = vmapL.getOrDefault(vISym)
          var prevNames = initHashSet[string]()
          for pl in prevLeaves:
            if pl.kind == lkValue:
              prevNames.incl pl.name
          for lf in leaves:
            var lv = leafValueOf(n.aRight, lf.path, fnISym, vmapL, pmapL)
            rewriteExpr(lv)
            if isPtrType(lf.typ):
              leafList.add mkPtrLeaf(lf.path, lf.typ, lv)
            else:
              let lname = leafName(n.aLeft.ident(), lf.path)
              leafList.add mkValueLeaf(lf.path, lf.typ, lname)
              if lname in prevNames:
                # already declared at the declaration site, assign the new value
                var upd = GpuAst(kind: gpuAssign, aLeft: newGpuIdent(lname),
                                 aRight: lv)
                upd.aLeft.symbol.typ = lf.typ
                replacements.add upd
              else:
                var lvNode = GpuAst(kind: gpuVar, vName: newGpuIdent(lname),
                                    vType: lf.typ, vInit: lv,
                                    vMutable: false, addressSpace: asRMEM)
                lvNode.vName.symbol.typ = lf.typ
                replacements.add lvNode
          vmapL[vISym] = leafList
          if replacements.len == 0:
            return true
          var blk = GpuAst(kind: gpuBlock, statements: replacements)
          n = blk
          return false
        else:
          rewriteExpr(n.aLeft)
          rewriteExpr(n.aRight)
          return false
      of gpuCall:
        rewriteExpr(n)
        return false
      of gpuBlock:
        var outStmts: seq[GpuAst]
        for st in n.statements.mitems:
          var s = st
          if not rewriteStmt(s):
            outStmts.add s
        n.statements = outStmts
        return false
      of gpuIf:
        rewriteExpr(n.ifCond)
        discard rewriteStmt(n.ifThen)
        if n.ifElse.kind != gpuDiscard:
          discard rewriteStmt(n.ifElse)
        return false
      of gpuFor:
        rewriteExpr(n.fStart)
        rewriteExpr(n.fEnd)
        rewriteExpr(n.fStep)
        discard rewriteStmt(n.fBody)
        return false
      of gpuWhile:
        rewriteExpr(n.wCond)
        discard rewriteStmt(n.wBody)
        return false
      of gpuReturn:
        rewriteExpr(n.rValue)
        return false
      else:
        # constexpr / comment / emit: rewrite nested exprs, keep
        for ch in n.mitems:
          rewriteExpr(ch)
        return false

    discard rewriteStmt(body)

  # apply the body rewrite per fn
  for fn in reachable:
    let fnISym = fn.pName.symbol.iSym
    var vmap = initTable[string, seq[FlattenedLeaf]]()
    var pmap = initTable[string, seq[FlattenedLeaf]]()
    # own flattened params: register the param leaf map
    for i, p in fn.pParams:
      if isTaintedStruct(p.typ):
        let leaves = taintedLeaves(p.typ)
        var leafList: seq[FlattenedLeaf]
        for lf in leaves:
          if isPtrType(lf.typ):
            leafList.add mkPtrLeaf(lf.path, lf.typ, nil)  # ptr param, bound by pass 3
            leafList[^1].name = flattenedParamName(p.ident.ident(), lf.path)
          else:
            leafList.add mkValueLeaf(lf.path, lf.typ,
                                     flattenedParamName(p.ident.ident(), lf.path))
        pmap[p.ident.symbol.iSym] = leafList
    if not fn.pBody.isNil:
      rewriteBody(fnISym, fn.pBody, vmap, pmap)
    # now apply the flattened param list to the signature
    if fnISym in newParams:
      var newPS: seq[GpuParam]
      for (oi, leaf) in newParams[fnISym]:
        if leaf.path.len == 0 and leaf.name != "":
          # plain param passthrough: keep the original
          newPS.add fn.pParams[oi]
        else:
          var np = GpuParam(ident: newGpuIdent(leaf.name),
                            typ: leaf.typ,
                            addressSpace: asRMEM,
                            passByRef: false)
          np.ident.symbol.typ = leaf.typ
          if leaf.kind == lkPtr:
            # ptr leaf param: keep a ptr param (bound by pass 3). The name
            # must be unique, so reuse the leaf name
            np.typ = leaf.typ
          newPS.add np
      fn.pParams = newPS

  # ── remove tainted-returning fns (all call sites were expanded) ────────
  for iSym in taintedReturnFns:
    removeFn(ctx, iSym)

  # ── remove tainted struct type defs (no legal GLSL representation) ─────
  var taintedTypes: seq[GpuType]
  for t in ctx.types.keys:
    if isTaintedStruct(t):
      taintedTypes.add t
  for t in taintedTypes:
    ctx.types.del t
