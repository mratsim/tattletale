## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan-specific shared helpers for the Vulkan IR legalization passes:
## taint analysis (struct types containing pointer fields), fn-table and reachability helpers,
## plus the ptr-index fold shared by passes 2 and 3.

import std/[sets, strutils, tables]
import ../ir/gpu_types
import ./passes_utils

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
  ## True when `t` is a non-nil pointer-like type (gtPtr / gtUA / gtVoidPtr).
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
  ## True when the fn carries the global (kernel) attribute.
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
  ## Returns nil when the shape does not match, so callers keep their original Index node.
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
        let idxT = getExprType(ctx, idx, fns)
        if not idxT.isNil:
          let offT = getExprType(ctx, offC, fns)
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
