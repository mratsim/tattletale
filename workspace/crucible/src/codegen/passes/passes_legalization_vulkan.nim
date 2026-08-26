## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan-only IR legalizations (registered from the `vulkan:` codegen path).
##
## GLSL (Vulkan) has no raw pointers, no references and no array-valued
## returns. The ceramic tile layer (gemm_with_epilogue + tile_epilogues +
## tile_mma) uses all three: device-fn `ptr UncheckedArray` params,
## `var T` (pass-by-reference) params, and struct VALUES that carry pointer
## fields (`StridedOperand.data`, `TensorView.data`).
## These three passes lower those IR shapes to legal GLSL:
##
## 1. `vulkanVarParamsToValue` — device-fn `var T` params become value params.
##    Mutated struct/scalar params are returned by value (call sites become
##    `x = f(x, …)`); array-typed var params (no GLSL array returns) inline
##    the fn at its call sites, each copy wrapped in its own block.
## 2. `vulkanFlattenStructPtrValues` — struct values that (transitively)
##    contain pointer fields are eliminated: vars split into leaf scalars +
##    ptr-leaf expressions (GLSL has no pointer locals), value params split
##    into leaf params, struct-returning fns (gd,
##    local_tile_dyn) resolve to per-leaf return expressions over their
##    params, dot-access chains are rewritten, and the tainted struct type
##    defs are removed (GLSL structs cannot hold pointer members).
## 3. `vulkanBindDeviceFnPtrParams` — per-call-site device-fn binding:
##    device fns with `ptr` params are cloned per agreeing call-site arg
##    tuple, the ptr args are substituted ident→expression into the body
##    (`buf +% baseOff` shapes), and ptr-arg indexing over
##    pointer-arithmetic chains folds to SSBO element indexing.
##
## All three are gated on `crucibleCompileTarget == ctVulkan` so the other
## backends (Metal/CUDA/OpenCL/WGSL) never see them. They run after the
## common passes (blit/constexpr normalization) and before vulkan_lang's
## codegen, which then only asserts that the IR is legal.

import std/[algorithm, sets, strformat, strutils, tables]
import ../ir/gpu_types
import ../builtins/builtins_compilermagic
import ./pass_datatypes

# ═════════════════════════════════════════════════════════════════════════
#  Taint analysis: struct types that contain pointer fields
# ═════════════════════════════════════════════════════════════════════════

proc structFieldsOf(t: GpuType): seq[GpuTypeField] =
  ## The fields of a struct type (gtObject or gtGenericInst).
  case t.kind
  of gtObject: result = t.oFields
  of gtGenericInst: result = t.gFields
  else: discard

proc isStructType(t: GpuType): bool =
  ## True when `t` is a struct-like type with fields.
  not t.isNil and t.kind in {gtObject, gtGenericInst}

proc containsPtrField(t: GpuType): bool =
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

proc isTaintedStruct(t: GpuType): bool =
  ## A struct type whose field tree contains a pointer: GLSL structs cannot
  ## hold pointer members, so values of this type must be flattened away.
  isStructType(t) and containsPtrField(t)

proc isPtrType(t: GpuType): bool =
  not t.isNil and t.kind in {gtPtr, gtUA, gtVoidPtr}

proc taintedLeaves(t: GpuType, path: seq[string] = @[]): seq[tuple[path: seq[string], typ: GpuType]] =
  ## Decomposes a tainted struct type into leaves: each maximal non-tainted
  ## subtree (scalar, plain struct, array) and each pointer field becomes one
  ## leaf. Depth-first in field order — the SAME order is used for param
  ## flattening (callee signature) and call-site arg expansion (caller), so
  ## positions always line up.
  if t.isNil: return
  case t.kind
  of gtObject, gtGenericInst:
    for f in structFieldsOf(t):
      if isTaintedStruct(f.typ):
        result.add taintedLeaves(f.typ, path & f.name)
      elif containsPtrField(f.typ):
        # pointer (or ptr-carrying non-struct) leaf — kept whole
        result.add (path & f.name, f.typ)
      else:
        # plain subtree: one leaf of its own type
        result.add (path & f.name, f.typ)
  else:
    # not a struct — should not be called with non-structs
    result.add (path, t)

# ═════════════════════════════════════════════════════════════════════════
#  IR helpers
# ═════════════════════════════════════════════════════════════════════════

proc collectAssigns(n: GpuAst; assigns, consts: var Table[string, GpuAst]) =
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

proc resolveValue(n: GpuAst; assigns, consts: Table[string, GpuAst];
                  visited: var HashSet[string]; depth: int): GpuAst =
  ## Resolves an expression through single-assignment chains (blit temps,
  ## constexprs) and folds dots over object constructions, so a leaf value
  ## becomes a pure expression over params/literals. `visited` guards
  ## against cycles; `depth` bounds pathological chains.
  if n.isNil: return n
  if depth > 512:
    raiseAssert "Vulkan: resolveValue exceeded depth — assignment cycle?"
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

proc substIdents(n: GpuAst; subst: Table[string, GpuAst]): GpuAst =
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

proc exprType(n: GpuAst): GpuType =
  ## Best-effort static type of an expression, for the ptr-index fold's
  ## offset coercion (COMP-B-003: GLSL forbids mixed-type arithmetic).
  if n.isNil: return nil
  case n.kind
  of gpuIdent: result = n.symbol.typ
  of gpuLit: result = n.lType
  of gpuBinOp: result = n.bType
  of gpuCast: result = n.cTo
  of gpuConv: result = n.convTo
  else: result = nil

proc leafName(base: string; path: seq[string]): string =
  ## `epi` + ["C", "rsc"] → "epi_C_rsc". Single-underscore separator:
  ## GLSL §3.7 reserves identifiers containing two consecutive underscores
  ## (BUG-A-006).
  result = base
  for p in path:
    result.add "_" & p
  if result.startsWith("gl_"):
    # identifiers starting with `gl_` are reserved in GLSL (the TensorView
    # param in loadTile/storeTile is named `gl`) — escape the prefix
    result = "lv_" & result

# ═════════════════════════════════════════════════════════════════════════
#  fn-table helpers
# ═════════════════════════════════════════════════════════════════════════

proc isGlobalFn(fn: GpuAst): bool =
  fn.kind == gpuProc and attGlobal in fn.pAttributes

proc allFnIdentifiers(ctx: GpuContext): seq[tuple[key: GpuAst, fn: GpuAst]] =
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

proc removeFn(ctx: var GpuContext; iSym: string) =
  ## Removes a fn from allFnTab, genericInsts and fnTab (all are searched by
  ## preprocess's scanFunctions / farmTopLevel; fnTab is what the backends'
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

proc addFn(ctx: var GpuContext; fn: GpuAst) =
  ## Adds a (possibly cloned) fn to all tables, including fnTab (the table
  ## the backends' codegen iterates).
  ctx.allFnTab[fn.pName] = fn
  ctx.genericInsts[fn.pName] = fn
  ctx.fnTab[fn.pName] = fn

# ═════════════════════════════════════════════════════════════════════════
#  Reachability
# ═════════════════════════════════════════════════════════════════════════

proc collectCalls(n: GpuAst; outCalls: var seq[GpuAst]) =
  ## Collects every gpuCall node in `n` (the fn body).
  case n.kind
  of gpuCall:
    outCalls.add n
    for a in n.cArgs:
      collectCalls(a, outCalls)
  else:
    for ch in n:
      collectCalls(ch, outCalls)

proc reachableFns(ctx: GpuContext): seq[GpuAst] =
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
#  Pass 1: vulkanVarParamsToValue
# ═════════════════════════════════════════════════════════════════════════

proc varParamWritten(fn: GpuAst; pISym: string): bool =
  ## True when the var param is written somewhere in the body (assign target
  ## or address-taken as a var arg). Struct var params are written through
  ## field/index chains (`tmp.frags[n][m].frag[v] = …`), so the assign
  ## target's BASE (after unwrapping Index/Dot/Deref) is checked for the
  ## param ident.
  var found = false
  proc assignBase(l0: GpuAst): GpuAst =
    var l = l0
    while true:
      case l.kind
      of gpuIndex: l = l.iArr
      of gpuDot: l = l.dParent
      of gpuDeref: l = l.dOf
      else: break
    l
  proc walk(n: GpuAst) =
    if found: return
    case n.kind
    of gpuAssign:
      let b = assignBase(n.aLeft)
      if b.kind == gpuIdent and b.symbol != nil and b.symbol.iSym == pISym:
        found = true
        return
    of gpuAddr:
      let b = assignBase(n.aOf)
      if b.kind == gpuIdent and b.symbol != nil and b.symbol.iSym == pISym:
        found = true
        return
    else:
      discard
    for ch in n:
      if not found: walk(ch)
  if not fn.pBody.isNil:
    walk(fn.pBody)
  result = found

proc rewriteBodyDerefs(n: var GpuAst; paramISyms: HashSet[string]) =
  ## `Deref(Ident p)` → `Ident p` for converted value params.
  case n.kind
  of gpuDeref:
    if n.dOf.kind == gpuIdent and n.dOf.symbol != nil and
       n.dOf.symbol.iSym in paramISyms:
      n = n.dOf
    else:
      for ch in n.mitems:
        rewriteBodyDerefs(ch, paramISyms)
  else:
    for ch in n.mitems:
      rewriteBodyDerefs(ch, paramISyms)

proc convertVarParams(ctx: var GpuContext) =
  ## Device-fn `var T` params → GLSL-legal form:
  ## - array-typed `var` params: inline the fn at each call site (GLSL has
  ##   no array returns), each copy wrapped in its own block.
  ## - struct/scalar `var` params: value param + return-by-value when written
  ##   (call sites become `x = f(x, …)`); `Deref(p)` in bodies collapses.
  let reachable = reachableFns(ctx)
  var byISym = initTable[string, GpuAst]()
  for fn in reachable:
    byISym[fn.pName.symbol.iSym] = fn

  # ── Pass 1a: inline array-typed var-param fns ──────────────────────────
  proc paramIndexedAsArray(fn: GpuAst; pISym: string): bool =
    ## True when the body indexes the param ident directly (array-style,
    ## `d[0]`). The IR models `var array[N, T]` params as `var T` (element
    ## ptr) with array indexing on the ident — the array-ness only shows up
    ## in the body, not in the param type.
    if fn.pBody.isNil: return false
    var found = false
    proc scan(n: GpuAst) =
      if found: return
      case n.kind
      of gpuIndex:
        if n.iArr.kind == gpuIdent and n.iArr.symbol != nil and
           n.iArr.symbol.iSym == pISym:
          found = true
          return
        scan(n.iArr)
        scan(n.iIndex)
      else:
        for ch in n:
          if not found: scan(ch)
    scan(fn.pBody)
    result = found

  var arrayVarFns: seq[GpuAst]
  for fn in reachable:
    if fn.isGlobalFn(): continue
    for p in fn.pParams:
      if p.typ.kind == gtPtr and p.typ.implicit:
        if (not p.typ.to.isNil and p.typ.to.kind == gtArray) or
           paramIndexedAsArray(fn, p.ident.symbol.iSym):
          arrayVarFns.add fn
          break

  for fnItem in arrayVarFns:
    let fn = fnItem  # ref copy — lent loop vars cannot be captured
    let fnISym = fn.pName.symbol.iSym
    # locate hosts that call it
    var hosts: seq[GpuAst]
    for host in reachable:
      if host.pBody.isNil: continue
      var calls: seq[GpuAst]
      collectCalls(host.pBody, calls)
      for c in calls:
        if c.cName.symbol != nil and c.cName.symbol.iSym == fnISym:
          hosts.add host
          break
    if hosts.len == 0:
      # dead — removed by (b) if tainted-returning; otherwise leave
      continue
    for host in hosts:
      # inline every call to this fn in one walk, matching by callee iSym
      # (GpuAst `==` only supports idents; ref identity is unavailable in
      # the compile-time VM). Only STATEMENT-position calls are inlined
      # (GLSL has no expression blocks); an expression-position call to an
      # array-var-param fn is rejected loudly. Any `return` in the callee
      # body is also rejected: an inlined `return` would return from the
      # HOST kernel, silently truncating it (BUG-A-003).
      proc checkNoReturn(n: GpuAst) =
        case n.kind
        of gpuReturn:
          raiseAssert "Vulkan: cannot inline array-var-param device fn '" &
            fn.pName.ident() & "' — body contains a `return` (an inlined " &
            "return would return from the host kernel)"
        else:
          for ch in n:
            checkNoReturn(ch)
      proc inlineWalk(n: var GpuAst)   # forward — inlineBlock calls it
      proc inlineBlock(stmts: var seq[GpuAst]) =
        var outS: seq[GpuAst]
        for st in stmts.mitems:
          var s = st
          if s.kind == gpuCall and s.cName.symbol != nil and
             s.cName.symbol.iSym == fnISym:
            if s.cArgs.len != fn.pParams.len:
              raiseAssert "Vulkan: arity mismatch inlining device fn '" &
                fn.pName.ident() & "' (" & $s.cArgs.len & " args vs " &
                $fn.pParams.len & " params)"
            var subst = initTable[string, GpuAst]()
            for i, p in fn.pParams:
              subst[p.ident.symbol.iSym] = s.cArgs[i]
            var inlined = fn.pBody.clone()
            inlined = substIdents(inlined, subst)
            checkNoReturn(inlined)
            # replace the call statement with the inlined body in a fresh block
            outS.add GpuAst(kind: gpuBlock, statements: @[inlined])
          else:
            inlineWalk(s)
            outS.add s
        stmts = outS
      proc inlineWalk(n: var GpuAst) =
        case n.kind
        of gpuBlock:
          inlineBlock(n.statements)
        of gpuCall:
          if n.cName.symbol != nil and n.cName.symbol.iSym == fnISym:
            raiseAssert "Vulkan: cannot inline array-var-param device fn '" &
              fn.pName.ident() &
              "' in expression position — GLSL has no expression blocks"
          for a in n.cArgs.mitems:
            inlineWalk(a)
        else:
          for ch in n.mitems:
            inlineWalk(ch)
      inlineWalk(host.pBody)
    removeFn(ctx, fnISym)

  # ── Pass 1b: struct/scalar var params → value + return ─────────────────
  for fnItem in reachable:
    let fn = fnItem  # ref copy — lent loop vars cannot be captured
    if fn.isGlobalFn(): continue
    var varPos = newSeq[int]()
    for i, p in fn.pParams:
      if p.typ.kind == gtPtr and p.typ.implicit and
         (p.typ.to.isNil or p.typ.to.kind != gtArray):
        if i notin varPos:  # dedup — genericInsts may share param objects
          varPos.add i
    if varPos.len == 0:
      continue
    let fnISym = fn.pName.symbol.iSym
    # map param iSym → leaf param name (same name, value type)
    var valueParamISyms = initHashSet[string]()
    var writtenISyms = newSeq[string]()
    for pos in varPos:
      let p = fn.pParams[pos]
      if p.typ.kind != gtPtr:
        # already a value param (dup varPos or shared param object from a
        # generic-instance clone) — nothing left to convert
        continue
      valueParamISyms.incl p.ident.symbol.iSym
      if varParamWritten(fn, p.ident.symbol.iSym):
        writtenISyms.add p.ident.symbol.iSym
      # change the param type: var T → T (value)
      fn.pParams[pos].typ = p.typ.to
      if fn.pParams[pos].typ.isNil:
        raiseAssert "Vulkan: var param '" & p.ident.ident() & "' of fn '" &
          fn.pName.ident() & "' has nil pointee type"
      fn.pParams[pos].passByRef = false
    # collapse Deref(p) in the body
    if not fn.pBody.isNil:
      rewriteBodyDerefs(fn.pBody, valueParamISyms)
    # if any written: return the mutated value(s). GLSL returns one value, so
    # a fn with >1 written var param cannot be lowered this way.
    if writtenISyms.len > 1:
      raiseAssert "Vulkan: device fn '" & fn.pName.ident() &
        "' has " & $writtenISyms.len & " written var params — GLSL fns return one value"
    if writtenISyms.len == 1:
      let retISym = writtenISyms[0]
      var retIdent: GpuAst
      var retPos = -1
      for i, p in fn.pParams:
        if p.ident.symbol.iSym == retISym:
          retPos = i
          retIdent = p.ident
          break
      if not fn.pRetType.isNil and fn.pRetType.kind != gtVoid:
        raiseAssert "Vulkan: device fn '" & fn.pName.ident() &
          "' already returns a value and mutates a var param — cannot lower"
      # return the param's final value. Use the CONVERTED param type (the
      # pointee), not the param symbol's type — the symbol still carries the
      # original `var T` (gtPtr) type and would leak a raw pointer into the
      # fn signature (codegen rejects gtPtr return types).
      if retPos < 0 or fn.pParams[retPos].typ.isNil:
        raiseAssert "Vulkan: written var param '" & retISym & "' of fn '" &
          fn.pName.ident() & "' has no converted value type"
      let newRet = fn.pParams[retPos].typ
      fn.pRetType = newRet
      if not fn.pBody.isNil and fn.pBody.kind == gpuBlock:
        # Every `return` in the body must carry the mutated value: the old
        # code only converted the trailing return, leaving bare `return;` in
        # early-exit branches — invalid GLSL in a non-void fn (BUG-A-002).
        var body = fn.pBody
        proc fixReturn(n: var GpuAst) =
          case n.kind
          of gpuReturn:
            n.rValue = retIdent.clone()
          of gpuProc:
            # nested device fn — its returns belong to its own scope and the
            # outer fn's retIdent is not in scope there (SLOP-003: the old
            # full-tree recursion rewrote them, emitting `return x;` inside
            # the nested fn where x is undeclared)
            discard
          else:
            for ch in n.mitems:
              fixReturn(ch)
        fixReturn(body)
        # GLSL requires every path of a non-void fn to return a value —
        # append a trailing `return x` for the fall-through path (dead code
        # when the body already ends with a return).
        body.statements.add GpuAst(kind: gpuReturn, rValue: retIdent.clone())
    # rewrite call sites: `x = f(x, …)` — one walk per host, matching calls
    # by callee iSym (GpuAst `==` only supports idents; ref identity is
    # unavailable in the compile-time VM). Each call node computes its own
    # lvalue from its own args.
    var hosts: seq[GpuAst]
    for host in reachable:
      if host.pBody.isNil: continue
      var calls: seq[GpuAst]
      collectCalls(host.pBody, calls)
      for c in calls:
        if c.cName.symbol != nil and c.cName.symbol.iSym == fnISym:
          hosts.add host
          break
    for host in hosts:
      proc rewriteCalls(n: var GpuAst) =
        case n.kind
        of gpuCall:
          if n.cName.symbol != nil and n.cName.symbol.iSym == fnISym:
            if n.cArgs.len != fn.pParams.len:
              raiseAssert "Vulkan: arity mismatch calling device fn '" & fn.pName.ident() &
                "' (" & $n.cArgs.len & " args vs " & $fn.pParams.len & " params)"
            # find the var arg(s): the position(s) that were var params.
            # Only WRITTEN var args conflict with GLSL's one-return-value
            # rule; unwritten ones become plain value params (BUG-A-005).
            # The old guard counted ALL var args and rejected valid calls
            # with a factually wrong message (SLOP-002).
            var writtenArgPos: seq[int]
            for pos in varPos:
              if pos < n.cArgs.len and writtenISyms.len == 1 and
                 fn.pParams[pos].ident.symbol.iSym == writtenISyms[0]:
                writtenArgPos.add pos
            if writtenArgPos.len > 1:
              raiseAssert "Vulkan: call to '" & fn.pName.ident() &
                "' passes " & $writtenArgPos.len &
                " written var args — GLSL fns return one value"
            # the callee now takes VALUE params — strip addr/deref from every
            # var arg (written ones additionally come back as the return value)
            var newArgs = n.cArgs
            for pos in varPos:
              if pos >= newArgs.len:
                continue
              var a = newArgs[pos]
              if a.kind == gpuAddr:
                a = a.aOf
              if a.kind == gpuDeref:
                a = a.dOf
              newArgs[pos] = a
            var newCall = GpuAst(kind: gpuCall, cIsExpr: true, cName: n.cName)
            for a in newArgs:
              newCall.cArgs.add a
            if writtenArgPos.len == 1:
              # the callee returns the mutated value → `lvalue = f(…lvalue…)`
              let lvalue = newArgs[writtenArgPos[0]]
              let assign = GpuAst(kind: gpuAssign, aLeft: lvalue.clone(), aRight: newCall)
              n = assign
            else:
              # no written var param: plain value call
              n = newCall
        of gpuBlock:
          for i, st in n.statements.mpairs:
            rewriteCalls(st)
        else:
          for ch in n.mitems:
            rewriteCalls(ch)
      rewriteCalls(host.pBody)

# ═════════════════════════════════════════════════════════════════════════
#  Pass 2: vulkanFlattenStructPtrValues
# ═════════════════════════════════════════════════════════════════════════

type
  LeafKind* = enum
    lkValue      ## scalar or plain-struct leaf — backed by a var or param ident
    lkPtr        ## pointer leaf — backed by an expression (SSBO ref / ptr arith)

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

proc flattenStructPtrValues(ctx: var GpuContext) =
  ## (b) — see module header.

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
            nps.add (i, mkPtrLeaf(lf.path, lf.typ, nil))  # ptr param — bound by pass A
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
        # full leaf: value → ident; ptr → expr (or the leaf-param ident when
        # the ptr is a device-fn param bound later by pass A)
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
              # ptr-leaf param bound by pass A → the leaf-param ident
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
          # the leaf — e.g. Dot((1,1,N,K), F0) → 1 — using the CALLER's
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
              # ptr-leaf param bound by pass A → the leaf-param ident
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
            # off is in elements; the lit is sizeof(T) — fold it away
            off = bop.bRight.bLeft
          else:
            off = nil
        elif bop.bRight.kind == gpuLit:
          off = nil  # literal byte offset — cannot recover element count
        else:
          off = nil
        if not base.isNil and not off.isNil:
          # fold: base[off + idx]. The WHOLE index is preserved — a `+`-binop
          # index keeps both operands (BUG-A-001: the old special branch
          # dropped idx.bLeft, silently mis-addressing `ptr[a + b]` reads).
          var offC = off
          if offC.kind == gpuCast and offC.cTo.kind == gtUint64:
            offC = offC.cExpr
          let idxT = exprType(idx)
          if not idxT.isNil:
            let offT = exprType(offC)
            if not offT.isNil and offT.kind != idxT.kind and
               idxT.kind in {gtInt32, gtUint32}:
              # coerce the offset to the index's type (COMP-B-003: GLSL
              # forbids mixed-type arithmetic operands)
              offC = GpuAst(kind: gpuConv, convTo: idxT, convExpr: offC)
          let newIdx = GpuAst(kind: gpuBinOp,
                             bOp: newGpuIdent("+"),
                             bLeft: offC.clone(), bRight: idx,
                             bIsOverloaded: false, bType: nil)
          result = GpuAst(kind: gpuIndex, iArr: base, iIndex: newIdx)
          return
    # plain ident base (SSBO member) or anything else: keep arr as-is
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
        # rewrite the index expr first, then the array expr; fold ptr-arith bases
        rewriteExpr(n.iIndex)
        var arrWasDot = n.iArr.kind == gpuDot
        rewriteExpr(n.iArr)
        if n.iArr.kind != gpuIdent and n.iArr.kind != gpuCast and n.iArr.kind != gpuDot:
          discard
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
              # original arg (same oi); consume it once, emit all leaves
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

    # statement-level rewrite; returns true when the node should be dropped
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
            # blit temp — value comes from a later assign; drop the decl
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
            # the leaf expr may contain unrewritten dots — always re-rewrite
            rewriteExpr(lv)
            if isPtrType(lf.typ):
              # ptr leaf: no local var — expression mapping
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
          # assignment to a tainted var — update its leaves. Value leaves are
          # ASSIGNED when already declared (re-assignment must use the new
          # value — HIDN-A-001: the old code dropped the assign, leaving the
          # stale first value) or DECLARED for the blit-temp pattern (gpuVar
          # with discard init); ptr leaves update the expression map.
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
                # already declared at the declaration site — assign the new value
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
            leafList.add mkPtrLeaf(lf.path, lf.typ, nil)  # ptr param — bound by pass A
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
            # ptr leaf param: keep a ptr param (bound by pass A); the name
            # must be unique — reuse the leaf name
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

# ═════════════════════════════════════════════════════════════════════════
#  Pass 3: vulkanBindDeviceFnPtrParams
# ═════════════════════════════════════════════════════════════════════════

proc structuralKey(n: GpuAst): string =
  ## Structural pretty-print with ident display names replaced by iSyms, so
  ## two expressions over different symbols never produce the same grouping
  ## key (BUG-B-002: name-keyed grouping merged distinct buffers).
  case n.kind
  of gpuIdent:
    if n.symbol != nil:
      result = "id(" & n.symbol.iSym & ")"
    else:
      result = "id(?)"
  of gpuCast: result = "cast[" & $n.cTo.kind & "](" & structuralKey(n.cExpr) & ")"
  of gpuConv: result = "conv[" & $n.convTo.kind & "](" & structuralKey(n.convExpr) & ")"
  of gpuBinOp:
    result = "(" & structuralKey(n.bLeft) & " " & structuralKey(n.bOp) &
             " " & structuralKey(n.bRight) & ")"
  of gpuIndex: result = structuralKey(n.iArr) & "[" & structuralKey(n.iIndex) & "]"
  of gpuDot: result = structuralKey(n.dParent) & "." & structuralKey(n.dField)
  of gpuDeref: result = "*" & structuralKey(n.dOf)
  of gpuAddr: result = "&" & structuralKey(n.aOf)
  of gpuLit: result = "lit(" & n.lValue & ")"
  of gpuPrefix: result = "prefix(" & n.pOp & "," & structuralKey(n.pVal) & ")"
  of gpuCall:
    result = "call(" & structuralKey(n.cName)
    for a in n.cArgs:
      result.add "," & structuralKey(a)
    result.add ")"
  of gpuArrayLit:
    result = "arr("
    for v in n.aValues:
      result.add structuralKey(v) & ","
    result.add ")"
  of gpuObjConstr:
    result = "constr("
    for f in n.ocFields:
      result.add f.name & ":" & structuralKey(f.value) & ","
    result.add ")"
  else: result = $n.kind

proc callArgKey(a: GpuAst): string =
  ## Canonical grouping key for a ptr call-site arg: idents by symbol
  ## identity (iSym), others by their structure with base-ident iSyms
  ## substituted for display names.
  if a.kind == gpuIdent and a.symbol != nil:
    result = "ident:" & a.symbol.iSym
  else:
    result = "expr:" & structuralKey(a)

proc bindDeviceFnPtrParams(ctx: var GpuContext) =
  ## (a) — see module header.

  let reachable = reachableFns(ctx)
  var byISym = initTable[string, GpuAst]()
  for fn in reachable:
    byISym[fn.pName.symbol.iSym] = fn

  # call depth: kernels 0, device fns = 1 + max caller depth
  var depthMemo = initTable[string, int]()
  proc fnDepth(iSym: string): int =
    if iSym in depthMemo: return depthMemo[iSym]
    let fn = byISym[iSym]
    if fn.isGlobalFn(): return 0
    var maxCaller = 0
    for host in reachable:
      if host.pBody.isNil: continue
      var calls: seq[GpuAst]
      collectCalls(host.pBody, calls)
      for c in calls:
        if c.cName.symbol != nil and c.cName.symbol.iSym == iSym:
          maxCaller = max(maxCaller, fnDepth(host.pName.symbol.iSym))
    result = 1 + maxCaller
    depthMemo[iSym] = result

  var deviceFns: seq[tuple[depth: int, fn: GpuAst]]
  for fn in reachable:
    if not fn.isGlobalFn():
      var hasPtr = false
      for p in fn.pParams:
        if p.typ.kind == gtPtr:
          hasPtr = true
          break
      if hasPtr:
        deviceFns.add (fnDepth(fn.pName.symbol.iSym), fn)
  deviceFns.sort(proc(a, b: tuple[depth: int, fn: GpuAst]): int = cmp(a.depth, b.depth))

  for (depth, fn) in deviceFns:
    let fnISym = fn.pName.symbol.iSym
    var ptrPos: seq[int]
    for i, p in fn.pParams:
      if p.typ.kind == gtPtr:
        ptrPos.add i
    if ptrPos.len == 0:
      continue
    # collect call sites across reachable fns
    var sites: seq[tuple[host: GpuAst, call: GpuAst]]
    for host in reachable:
      if host.pBody.isNil: continue
      var calls: seq[GpuAst]
      collectCalls(host.pBody, calls)
      for c in calls:
        if c.cName.symbol != nil and c.cName.symbol.iSym == fnISym:
          sites.add (host, c)
    if sites.len == 0:
      # unreachable from a kernel (or dead) — leave untouched (never emitted)
      continue
    for (host, call) in sites:
      if call.cArgs.len != fn.pParams.len:
        raiseAssert "Vulkan: arity mismatch calling device fn '" & fn.pName.ident() &
          "' from '" & host.pName.ident() & "' (" & $call.cArgs.len & " args vs " &
          $fn.pParams.len & " params)"
    # group by ptr-arg tuple
    var groups = initTable[string, seq[tuple[host: GpuAst, call: GpuAst]]]()
    var groupOrder: seq[string]
    for (host, call) in sites:
      var key = ""
      for pos in ptrPos:
        key.add callArgKey(call.cArgs[pos]) & "|"
      if key notin groups:
        groupOrder.add key
      groups.mgetOrPut(key, @[]).add (host, call)
    if groups.len == 1:
      # single agreement: bind in place
      let sites2 = groups[groupOrder[0]]
      var renames = initTable[string, GpuAst]()
      for pos in ptrPos:
        let arg = sites2[0].call.cArgs[pos]
        renames[fn.pParams[pos].ident.symbol.iSym] = arg
      if not fn.pBody.isNil:
        fn.pBody = substIdents(fn.pBody, renames)
      # drop the ptr params (highest position first) and the matching args
      for i in countdown(ptrPos.len - 1, 0):
        fn.pParams.delete(ptrPos[i])
        for (host, call) in sites2:
          call.cArgs.delete(ptrPos[i])
    else:
      # clone per group
      var clones: seq[tuple[key: string, fn: GpuAst]]
      for gi, key in groupOrder:
        let gs = groups[key]
        let cl = fn.clone()
        # fresh pName symbol
        let newName = fn.pName.ident() & "_vk" & $gi
        cl.pName = newGpuIdent(newName)
        cl.pName.symbol.iSym = newName
        cl.pName.symbol.symKind = gsProc
        var renames = initTable[string, GpuAst]()
        for pos in ptrPos:
          let arg = gs[0].call.cArgs[pos]
          renames[cl.pParams[pos].ident.symbol.iSym] = arg
        if not cl.pBody.isNil:
          cl.pBody = substIdents(cl.pBody, renames)
        for i in countdown(ptrPos.len - 1, 0):
          cl.pParams.delete(ptrPos[i])
        clones.add (key, cl)
      # point each group's call sites at its clone
      for gi, key in groupOrder:
        let cl = clones[gi].fn
        addFn(ctx, cl)
        for (host, call) in groups[key]:
          call.cName = cl.pName.clone()
          for i in countdown(ptrPos.len - 1, 0):
            call.cArgs.delete(ptrPos[i])
      # original is superseded (unless a group keeps using it — it doesn't)
      removeFn(ctx, fnISym)

  # ── post: fold ptr-index bases introduced by the substitutions ─────────
  # (Index over `cast[ptr](uint64(base) + off*sizeof)` → base[off + idx])
  proc foldPtrIndexes(n: var GpuAst) =
    case n.kind
    of gpuIndex:
      foldPtrIndexes(n.iArr)
      foldPtrIndexes(n.iIndex)
      if n.iArr.kind == gpuCast and n.iArr.cTo.kind == gtPtr:
        var idx = n.iIndex
        # reuse the same lowering as pass 2
        let bop = n.iArr.cExpr
        if bop.kind == gpuBinOp and bop.bOp.kind == gpuIdent and bop.bOp.ident() == "+":
          var base: GpuAst = nil
          var off: GpuAst = nil
          proc unwrapU64(x: GpuAst): GpuAst =
            if x.kind == gpuCast and x.cTo.kind == gtUint64 and x.cExpr.kind == gpuIdent:
              x.cExpr
            elif x.kind == gpuIdent:
              x
            else:
              nil
          base = unwrapU64(bop.bLeft)
          if bop.bRight.kind == gpuBinOp and bop.bRight.bOp.kind == gpuIdent and
             bop.bRight.bOp.ident() == "*" and
             bop.bRight.bRight.kind == gpuConv and
             bop.bRight.bRight.convTo.kind == gtUint64 and
             bop.bRight.bRight.convExpr.kind == gpuLit:
            off = bop.bRight.bLeft
          if not base.isNil and not off.isNil:
            var offC = off
            if offC.kind == gpuCast and offC.cTo.kind == gtUint64:
              offC = offC.cExpr
            let idxT = exprType(idx)
            if not idxT.isNil:
              let offT = exprType(offC)
              if not offT.isNil and offT.kind != idxT.kind and
                 idxT.kind in {gtInt32, gtUint32}:
                # coerce the offset to the index's type (COMP-B-003)
                offC = GpuAst(kind: gpuConv, convTo: idxT, convExpr: offC)
            let newIdx = GpuAst(kind: gpuBinOp, bOp: newGpuIdent("+"),
                               bLeft: offC.clone(), bRight: idx,
                               bIsOverloaded: false, bType: nil)
            n = GpuAst(kind: gpuIndex, iArr: base, iIndex: newIdx)
    else:
      for ch in n.mitems:
        foldPtrIndexes(ch)
  # Iterate ctx.fnTab (not the stale pre-clone `reachable` snapshot): the
  # per-call-site clones were added mid-pass, and their bodies are exactly
  # the ones that received substituted ptr-arith expressions (BUG-A-004).
  for fnIdent, fn in ctx.fnTab.mpairs:
    if not fn.pBody.isNil:
      foldPtrIndexes(fn.pBody)

# ═════════════════════════════════════════════════════════════════════════
#  Pass 4: vulkanSubgroupGuard32 (GPU-B-001)
# ═════════════════════════════════════════════════════════════════════════

proc usesReductionBuiltin(n: GpuAst): bool =
  ## True when the body contains a subgroup-shuffle (reduction) builtin call.
  if n.isNil: return false
  if n.kind == gpuCall and n.cName.symbol != nil and
     n.cName.symbol.reductionBuiltin != gbkNone:
    return true
  for ch in n:
    if usesReductionBuiltin(ch):
      return true

proc subgroupGuard32(ctx: var GpuContext) =
  ## GPU-B-001: the fp16-subgroup shuffle path (tileKMax reduction trees,
  ## universalMma8x8x8) assumes 32-lane subgroups — the shuffle trees use
  ## deltas up to 16 and a 32-lane bit decomposition, both undefined on
  ## 8/16-lane-subgroup devices (Intel Gen9+, some Mali/Adreno). Fail
  ## loudly instead of silently computing wrong results:
  ##  - every kernel whose transitive call graph reaches a reduction builtin
  ##    gets `if (gl_SubgroupSize < 32u) { return; }` as its first statement
  ##    (the kernel returns without writing its outputs, so a host-side
  ##    value check fails loudly);
  ##  - in those fns, the lane id comes from `gl_SubgroupInvocationID` (the
  ##    true subgroup lane) instead of `gl_LocalInvocationIndex` (the
  ##    workgroup lane — equal only when workgroup == subgroup, which the
  ##    guard fixes at 32 alongside the kernels' baked 32-wide workgroups).
  ## The engine-level VkPhysicalDeviceSubgroupProperties ingest query is
  ## tracked debt (no engine edits in this op).
  let reachable = reachableFns(ctx)
  # transitive closure over the call graph: a fn is shuffle-reachable when
  # its body contains a reduction builtin or calls a shuffle-reachable fn
  var shuffleReachable = initHashSet[string]()
  var changed = true
  while changed:
    changed = false
    for fn in reachable:
      if fn.pName.symbol.iSym in shuffleReachable: continue
      var hits = not fn.pBody.isNil and usesReductionBuiltin(fn.pBody)
      if not hits and not fn.pBody.isNil:
        var calls: seq[GpuAst]
        collectCalls(fn.pBody, calls)
        for c in calls:
          if c.cName.symbol != nil and c.cName.symbol.iSym in shuffleReachable:
            hits = true
            break
      if hits:
        shuffleReachable.incl fn.pName.symbol.iSym
        changed = true
  # lane id: thread_index_in_threadgroup → gl_SubgroupInvocationID in
  # shuffle-reachable bodies. Replace the node rather than mutating it: the
  # catalog ident node is sigTab-shared across the module, so an in-place
  # symbol swap would leak the subgroup lane into every non-shuffle fn that
  # still references the shared node (GOAL-001/SLOP-001) — non-shuffle fns
  # must keep gl_LocalInvocationIndex (gl_SubgroupInvocationID is only valid
  # where the subgroup extensions are enabled).
  proc rewriteLaneId(n: var GpuAst) =
    if n.kind == gpuIdent and n.symbol != nil and
       n.symbol.coordBuiltin == gbkThreadIndexInThreadgroup:
      n = GpuAst(kind: gpuIdent, symbol: newSymbol("gl_SubgroupInvocationID",
                 typ = n.symbol.typ, symKind = gsBuiltin))
    else:
      for ch in n.mitems:
        rewriteLaneId(ch)
  for fn in reachable:
    if fn.pName.symbol.iSym in shuffleReachable and not fn.pBody.isNil:
      rewriteLaneId(fn.pBody)
  # guard: first statement of every subgroup-using kernel
  for fn in reachable:
    if fn.isGlobalFn() and fn.pName.symbol.iSym in shuffleReachable and
       not fn.pBody.isNil:
      let guard = GpuAst(kind: gpuEmit, parts: @[GpuEmitPart(
        kind: peLiteral, literal: "if (gl_SubgroupSize < 32u) { return; }")])
      if fn.pBody.kind == gpuBlock:
        fn.pBody.statements.insert(guard, 0)
      else:
        fn.pBody = GpuAst(kind: gpuBlock, statements: @[guard, fn.pBody])

# ═════════════════════════════════════════════════════════════════════════
#  Registration
# ═════════════════════════════════════════════════════════════════════════

proc registerLegalizationVulkanPasses*(reg: var PassRegistry) =
  ## Registers the Vulkan-only legalizations. Called from the `vulkan:`
  ## codegen path AFTER registerCommonPasses, so the blit/constexpr
  ## normalization is already in place; runs phaseMain (transform) and is
  ## gated on crucibleCompileTarget == ctVulkan so the other backends are
  ## untouched.
  reg.register("vulkanVarParamsToValue", pkTransform, phaseMain,
    "Device-fn var params → value params (+return by value; array-typed var-param fns inlined)",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      if crucibleCompileTarget == ctVulkan:
        convertVarParams(ctx)
  )
  reg.register("vulkanFlattenStructPtrValues", pkTransform, phaseMain,
    "Flatten struct-with-ptr-field values into leaf scalars + SSBO ptr expressions",
    dependsOn = @["vulkanVarParamsToValue"],
    run = proc(ctx: var GpuContext): void =
      if crucibleCompileTarget == ctVulkan:
        flattenStructPtrValues(ctx)
  )
  reg.register("vulkanBindDeviceFnPtrParams", pkTransform, phaseMain,
    "Per-call-site device-fn ptr-param binding with ident→expression substitution",
    dependsOn = @["vulkanFlattenStructPtrValues"],
    run = proc(ctx: var GpuContext): void =
      if crucibleCompileTarget == ctVulkan:
        bindDeviceFnPtrParams(ctx)
  )
  reg.register("vulkanSubgroupGuard32", pkTransform, phaseMain,
    "Fail-loudly gl_SubgroupSize<32 guard + gl_SubgroupInvocationID lane id on the fp16-subgroup shuffle path (GPU-B-001)",
    dependsOn = @["vulkanBindDeviceFnPtrParams"],
    run = proc(ctx: var GpuContext): void =
      if crucibleCompileTarget == ctVulkan:
        subgroupGuard32(ctx)
  )
