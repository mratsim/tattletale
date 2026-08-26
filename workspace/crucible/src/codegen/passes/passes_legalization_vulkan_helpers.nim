## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan-specific shared helpers for the Vulkan IR legalization passes:
## taint analysis (struct types containing pointer fields), expression
## resolution over single-assignment chains, fn-table and reachability
## helpers, and the ptr-index fold shared by passes 2 and 3.

import std/[sets, strutils, tables]
import ../ir/gpu_types

# ═════════════════════════════════════════════════════════════════════════
#  Taint analysis: struct types that contain pointer fields
# ═════════════════════════════════════════════════════════════════════════

proc structFieldsOf*(t: GpuType): seq[GpuTypeField] =
  ## The fields of a struct type (gtObject or gtGenericInst).
  case t.kind
  of gtObject: result = t.oFields
  of gtGenericInst: result = t.gFields
  else: discard

proc isStructType(t: GpuType): bool =
  ## True when `t` is a struct-like type with fields.
  not t.isNil and t.kind in {gtObject, gtGenericInst}

proc containsPtrField*(t: GpuType): bool =
  ## True when the type tree contains a gtPtr/gtUA/gtVoidPtr anywhere.
  if t.isNil: return false
  case t.kind
  of gtPtr, gtUA, gtVoidPtr: result = true
  of gtObject, gtGenericInst:
    for f in structFieldsOf(t):
      if containsPtrField(f.typ): return true
  of gtArray:
    result = containsPtrField(t.aTyp)
  else:
    result = false

proc isTaintedStruct*(t: GpuType): bool =
  ## A struct type whose field tree contains a pointer: GLSL structs cannot
  ## hold pointer members, so values of this type must be flattened away.
  isStructType(t) and containsPtrField(t)

proc isPtrType*(t: GpuType): bool =
  not t.isNil and t.kind in {gtPtr, gtUA, gtVoidPtr}

proc taintedLeaves*(t: GpuType, path: seq[string] = @[]): seq[tuple[path: seq[string], typ: GpuType]] =
  ## Decomposes a tainted struct type into leaves: each maximal non-tainted
  ## subtree (scalar, plain struct, array) and each pointer field becomes one
  ## leaf. Depth-first in field order: the SAME order is used for param
  ## flattening (callee signature) and call-site arg expansion (caller), so
  ## positions always line up.
  if t.isNil: return
  case t.kind
  of gtObject, gtGenericInst:
    for f in structFieldsOf(t):
      if isTaintedStruct(f.typ):
        result.add taintedLeaves(f.typ, path & f.name)
      elif containsPtrField(f.typ):
        # pointer (or ptr-carrying non-struct) leaf, kept whole
        result.add (path & f.name, f.typ)
      else:
        # plain subtree: one leaf of its own type
        result.add (path & f.name, f.typ)
  else:
    # not a struct: should not be called with non-structs
    result.add (path, t)

# ═════════════════════════════════════════════════════════════════════════
#  IR helpers
# ═════════════════════════════════════════════════════════════════════════

# moved to passes_utils in the dedup mission
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

proc exprType(n: GpuAst; fns: Table[string, GpuAst]): GpuType =
  ## Best-effort static type of an expression, for the ptr-index fold's
  ## offset coercion (COMP-B-003: GLSL forbids mixed-type arithmetic).
  ## `fns` maps callee iSym → device fn so a gpuCall operand resolves to
  ## the callee's return type. Without it the coercion is silently skipped
  ## and mixed-type GLSL surfaces only at glslang ingest.
  if n.isNil: return nil
  case n.kind
  of gpuIdent: result = n.symbol.typ
  of gpuLit: result = n.lType
  of gpuBinOp: result = n.bType
  of gpuCast: result = n.cTo
  of gpuConv: result = n.convTo
  of gpuCall:
    if n.cName.symbol != nil and fns.hasKey(n.cName.symbol.iSym):
      let callee = fns[n.cName.symbol.iSym]
      if not callee.pRetType.isNil and
         callee.pRetType.kind in {gtUint8, gtInt16, gtUint16, gtInt32,
                                  gtUint32, gtInt64, gtUint64}:
        result = callee.pRetType
  else: result = nil

proc leafName*(base: string; path: seq[string]): string =
  ## `epi` + ["C", "rsc"] → "epi_C_rsc". Single-underscore separator:
  ## GLSL §3.7 reserves identifiers containing two consecutive underscores
  ## (BUG-A-006).
  result = base
  for p in path:
    result.add "_" & p
  if result.startsWith("gl_"):
    # identifiers starting with `gl_` are reserved in GLSL (the TensorView
    # param in loadTile/storeTile is named `gl`), so escape the prefix
    result = "lv_" & result

# ═════════════════════════════════════════════════════════════════════════
#  fn-table helpers
# ═════════════════════════════════════════════════════════════════════════

proc isGlobalFn*(fn: GpuAst): bool =
  fn.kind == gpuProc and attGlobal in fn.pAttributes

proc allFnIdentifiers*(ctx: GpuContext): seq[tuple[key: GpuAst, fn: GpuAst]] =
  ## Snapshot of every fn in allFnTab + genericInsts, deduped by iSym
  ## (genericInsts entries are merged into allFnTab by preprocess, so a
  ## later clone/removal must touch both consistently).
  var seen = initHashSet[string]()
  for k, v in ctx.allFnTab:
    if k.symbol != nil and k.symbol.iSym notin seen:
      seen.incl k.symbol.iSym
      result.add (k, v)
  for k, v in ctx.genericInsts:
    if k.symbol != nil and k.symbol.iSym notin seen:
      seen.incl k.symbol.iSym
      result.add (k, v)

proc removeFn*(ctx: var GpuContext; iSym: string) =
  ## Removes a fn from allFnTab, genericInsts and fnTab (all are searched by
  ## preprocess's scanFunctions / farmTopLevel. fnTab is what the backends'
  ## codegen iterates, so dead fns must not linger there).
  var toDel: seq[GpuAst]
  for k in ctx.allFnTab.keys:
    if k.symbol != nil and k.symbol.iSym == iSym:
      toDel.add k
  for k in toDel:
    ctx.allFnTab.del k
  toDel.setLen(0)
  for k in ctx.genericInsts.keys:
    if k.symbol != nil and k.symbol.iSym == iSym:
      toDel.add k
  for k in toDel:
    ctx.genericInsts.del k
  toDel.setLen(0)
  for k in ctx.fnTab.keys:
    if k.symbol != nil and k.symbol.iSym == iSym:
      toDel.add k
  for k in toDel:
    ctx.fnTab.del k

proc addFn*(ctx: var GpuContext; fn: GpuAst) =
  ## Adds a (possibly cloned) fn to all tables, including fnTab (the table
  ## the backends' codegen iterates).
  ctx.allFnTab[fn.pName] = fn
  ctx.genericInsts[fn.pName] = fn
  ctx.fnTab[fn.pName] = fn

# ═════════════════════════════════════════════════════════════════════════
#  Reachability
# ═════════════════════════════════════════════════════════════════════════

proc collectCalls*(n: GpuAst; outCalls: var seq[GpuAst]) =
  ## Collects every gpuCall node in `n` (the fn body).
  case n.kind
  of gpuCall:
    outCalls.add n
    for a in n.cArgs:
      collectCalls(a, outCalls)
  else:
    for ch in n:
      collectCalls(ch, outCalls)

proc reachableFns*(ctx: GpuContext): seq[GpuAst] =
  ## Kernels + device fns reachable from kernels (transitive), in the order
  ## they appear in allFnTab/genericInsts.
  var byISym = initTable[string, GpuAst]()
  for (k, v) in allFnIdentifiers(ctx):
    byISym[k.symbol.iSym] = v
  var reachable = initOrderedTable[string, GpuAst]()
  var queue: seq[GpuAst]
  for (k, v) in allFnIdentifiers(ctx):
    if v.isGlobalFn():
      reachable[k.symbol.iSym] = v
      queue.add v
  var qi = 0
  while qi < queue.len:
    let fn = queue[qi]
    inc qi
    if fn.pBody.isNil: continue
    var calls: seq[GpuAst]
    collectCalls(fn.pBody, calls)
    for c in calls:
      if c.cName.symbol != nil and c.cName.symbol.iSym in byISym:
        let callee = byISym[c.cName.symbol.iSym]
        if callee.kind == gpuProc and not callee.isGlobalFn() and
           not contains(reachable, callee.pName.symbol.iSym):
          reachable[callee.pName.symbol.iSym] = callee
          queue.add callee
  for k, v in reachable.pairs:
    result.add v

# ═════════════════════════════════════════════════════════════════════════
#  Ptr-index fold (shared by passes 2 and 3)
# ═════════════════════════════════════════════════════════════════════════

proc foldPtrIndexToElement*(arr, idx: GpuAst; ctx: GpuContext;
                            fns: Table[string, GpuAst]): GpuAst =
  ## Recognizes the `+%` pointer-arithmetic shape (ptr = cast[ptr T](
  ## uint64(base) + uint64(off) * sizeof(T))) and lowers element indexing
  ## over it to SSBO indexing with the offset folded into the index.
  ## Returns nil when the shape does not match, so callers keep their
  ## original Index node.
  if arr.kind == gpuCast and arr.cTo.kind == gtPtr and arr.cExpr.kind == gpuBinOp:
    let bop = arr.cExpr
    if bop.bOp.kind == gpuIdent and bop.bOp.ident() == "+":
      var base: GpuAst = nil
      var off: GpuAst = nil
      # operand 1: base (possibly uint64-cast of an ident)
      proc unwrapU64(x: GpuAst): GpuAst =
        if x.kind == gpuCast and x.cTo.kind == gtUint64 and x.cExpr.kind == gpuIdent:
          x.cExpr
        elif x.kind == gpuIdent:
          x
        else:
          nil
      base = unwrapU64(bop.bLeft)
      # operand 2: off * sizeof(T) or off (bytes already?)
      if bop.bRight.kind == gpuBinOp and bop.bRight.bOp.kind == gpuIdent and
         bop.bRight.bOp.ident() == "*":
        let sizeOp = bop.bRight.bRight
        if sizeOp.kind == gpuConv and sizeOp.convTo.kind == gtUint64 and
           sizeOp.convExpr.kind == gpuLit:
          # off is in elements: the sizeof(T) literal folds away
          off = bop.bRight.bLeft
        else:
          off = nil
      elif bop.bRight.kind == gpuLit:
        off = nil  # literal byte offset: cannot recover the element count
      else:
        off = nil
      if not base.isNil and not off.isNil:
        # fold: base[off + idx]. The whole index is preserved, so a `+`-binop
        # index keeps both operands.
        var offC = off
        if offC.kind == gpuCast and offC.cTo.kind == gtUint64:
          offC = offC.cExpr
        let idxT = exprType(idx, fns)
        if not idxT.isNil:
          let offT = exprType(offC, fns)
          if not offT.isNil and offT.kind != idxT.kind and
             idxT.kind in {gtInt32, gtUint32}:
            # coerce the offset to the index's type: GLSL forbids mixed-type
            # arithmetic operands
            offC = GpuAst(kind: gpuConv, convTo: idxT, convExpr: offC)
        let newIdx = GpuAst(kind: gpuBinOp,
                           bOp: newGpuIdent("+"),
                           bLeft: offC.clone(), bRight: idx,
                           bIsOverloaded: false, bType: nil)
        result = GpuAst(kind: gpuIndex, iArr: base, iIndex: newIdx)
