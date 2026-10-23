# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
# Span and view lowering
#
# ############################################################
#
# openArray views split into ptr + len pairs,
# spans materialize as locals with rewritten uses,
# array and span parameters normalize.

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerLowerArraysAndViews*(reg: var PassRegistry, decayArrays: static bool) =
  reg.register("lowerArraysAndViews", pkTransform, phaseMain,
    "Arrays decay to element pointers per decayArrays, openArray views split into ptr + len",
    run = proc(ctx: var GpuContext): void =
      ctx.lowerArraysAndViews(decayArrays)
  )

import std / [hashes, macros, sequtils, sets, strutils, tables]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ../ir/ast_type_constructors
import ./pass_datatypes
import ./pass_registry
import ./pass_helpers

type
  SpanParam = object
    paramIdx: int        # index in the param list
    ptrName: string      # name for the ptr param
    lenName: string      # name for the len param
    ptrTyp: GpuType
    lenTyp: GpuType      # int32

  SpanSigMap = TableRef[string, seq[SpanParam]]

proc collectSpanSigs(ctx: var GpuContext): SpanSigMap =
  ## Build a map of function name → span params to split
  result = newTable[string, seq[SpanParam]]()

  template scanFn(fn: GpuAst) =
    if fn.isNil or fn.kind != gpuProc:
      continue
    if fn.pName.isNil or fn.pName.kind != gpuIdent:
      continue
    let fnName = fn.pName.symbol.name
    if fnName.len == 0:
      continue
    var spans: seq[SpanParam]
    for idx, p in fn.pParams:
      if p.flavor == kVar:
        # kVar params are references: they bind whole, never split into ptr + len
        continue
      if p.typ.kind == gtSpan:
        spans.add SpanParam(
          paramIdx: idx,
          ptrName: p.ident.symbol.name,
          lenName: p.ident.symbol.name & "_len",
          ptrTyp: initGpuPtrType(p.typ.sElemTyp, implicitPtr = false, addressSpace = p.typ.addressSpace),
          lenTyp: initGpuType(gtInt32)
        )
    if spans.len > 0:
      result[fnName] = spans

  var seen = initHashSet[string]()
  template scan(tab: untyped) =
    for fn in tab.values:
      if fn.isNil or fn.kind != gpuProc:
        continue
      if fn.pName.isNil or fn.pName.kind != gpuIdent:
        continue
      let nm = fn.pName.symbol.name
      if nm notin seen:
        seen.incl nm
        scanFn(fn)
  scan(ctx.allFnTab)
  scan(ctx.genericInsts)
  scan(ctx.fnTab)

proc rewriteFnSig(fn: var GpuAst, sigMap: SpanSigMap) =
  ## Split gtSpan params in a function definition
  if fn.isNil or fn.kind != gpuProc:
    return
  if fn.pName.isNil or fn.pName.kind != gpuIdent:
    return
  let fnName = fn.pName.symbol.name
  if fnName.len == 0:
    return
  let spans = sigMap.getOrDefault(fnName)
  if spans.len == 0:
    return

  var newParams: seq[GpuParam]
  var skipMask: seq[bool]
  skipMask.setLen(fn.pParams.len)

  for s in spans:
    skipMask[s.paramIdx] = true

  for idx, p in fn.pParams:
    if idx < skipMask.len and skipMask[idx]:
      for s in spans:
        if s.paramIdx == idx:
          # Find the span info for this param
          newParams.add GpuParam(
            ident: GpuAst(kind: gpuIdent, symbol: newSymbol(s.ptrName, typ = s.ptrTyp, symKind = p.ident.symbol.symKind)),
            typ: s.ptrTyp,
            addressSpace: p.addressSpace,
            passByRef: false
          )
          newParams.add GpuParam(
            ident: GpuAst(kind: gpuIdent, symbol: newSymbol(s.lenName, typ = s.lenTyp, symKind = p.ident.symbol.symKind)),
            typ: s.lenTyp,
            addressSpace: p.addressSpace,
            passByRef: false
          )
    else:
      newParams.add p
  fn.pParams = newParams

proc spanPtrLenParts(c: GpuAst): tuple[ptrPlus, lenExpr: GpuAst] =
  ## toOpenArray(ptr, first, last) -> (ptr + first, last - first + 1)
  let idxTyp = c.cArgs[1].getLoweringExprType()
  # getLoweringExprType never returns nil so idxTyp is taken above,
  # the gtInt32 variant guards genLit's nil-lType FieldDefect
  let lenTyp = (if idxTyp.isNil: initGpuType(gtInt32) else: idxTyp)
  let ptrPlus = GpuAst(kind: gpuBinOp,
                       bType: c.cArgs[0].getLoweringExprType(),
                       bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+")),
                       bLeft: c.cArgs[0],
                       bRight: c.cArgs[1])
  let lenSub = GpuAst(kind: gpuBinOp,
                      bType: idxTyp,
                      bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("-")),
                      bLeft: c.cArgs[2],
                      bRight: c.cArgs[1])
  let lenExpr = GpuAst(kind: gpuBinOp,
                       bType: idxTyp,
                       bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+")),
                       bLeft: lenSub,
                       bRight: GpuAst(kind: gpuLit, lValue: "1", lType: lenTyp))
  result = (ptrPlus, lenExpr)

proc rewriteViewUses(n: var GpuAst, sigMap: SpanSigMap) =
  ## Span-typed idents become pointer idents, `s.len` and `len(s)`
  ## become `s_len`, view arguments at call sites split into ptr + len.
  ## An unconsumed `toOpenArray` call is an error.
  case n.kind
  of gpuIdent:
    if n.symbol.typ != nil and n.symbol.typ.kind == gtSpan:
      # Span-typed ident -> ptr ident (same name, ptr type)
      n = GpuAst(kind: gpuIdent, symbol: newSymbol(n.symbol.name, typ = initGpuPtrType(n.symbol.typ.sElemTyp, implicitPtr = false, addressSpace = n.symbol.typ.addressSpace), symKind = n.symbol.symKind))
  of gpuDot:
    if n.dField.kind == gpuIdent and n.dField.symbol.name == "len" and
       n.dParent.symbol.typ != nil and n.dParent.symbol.typ.kind == gtSpan:
      # span.len -> len ident
      n = GpuAst(kind: gpuIdent,
                 symbol: newSymbol(n.dParent.symbol.name & "_len", typ = initGpuType(gtInt32)))
      return
    else:
      for child in mitems(n):
        child.rewriteViewUses(sigMap)
  of gpuCall:
    var fnName: string
    if n.cName.kind == gpuIdent:
      fnName = n.cName.symbol.name

    if fnName == "len" or fnName.startsWith("len_"):
      if n.cArgs.len >= 1:
        if n.cArgs[0].kind == gpuIdent and
           n.cArgs[0].symbol.typ != nil and n.cArgs[0].symbol.typ.kind == gtSpan:
          # len(span) -> span_len
          n = GpuAst(kind: gpuIdent,
                     symbol: newSymbol(n.cArgs[0].symbol.name & "_len", typ = initGpuType(gtInt32)))
          return
        if n.cArgs[0].kind == gpuCall and
           n.cArgs[0].cName.kind == gpuIdent and
           n.cArgs[0].cName.symbol.name == "toOpenArray" and
           n.cArgs[0].cArgs.len >= 3:
          # len(toOpenArray(ptr, first, last)) -> last - first + 1
          let (_, lenExpr) = spanPtrLenParts(n.cArgs[0])
          n = lenExpr
          return
      return

    if fnName == "toOpenArray" and n.cArgs.len >= 3:
      error "toOpenArray result used outside a span argument, a len call, or a view binding"

    # Split view arguments BEFORE recursing, the raw toOpenArray call stays visible
    var calleeSpans = sigMap.getOrDefault(fnName)
    if calleeSpans.len == 0:
      for k, v in sigMap:
        if fnName.startsWith(k):
          # Try matching by prefix (mangled generic names)
          calleeSpans = v
          break
    if calleeSpans.len > 0:
      var newArgs: seq[GpuAst]
      var argIdx = 0
      var spanIdx = 0
      while argIdx < n.cArgs.len:
        let isSpan = spanIdx < calleeSpans.len and calleeSpans[spanIdx].paramIdx == argIdx
        if isSpan:
          let spanArg = n.cArgs[argIdx]
          if spanArg.kind == gpuCall and spanArg.cName.kind == gpuIdent and
             spanArg.cName.symbol.name == "toOpenArray" and spanArg.cArgs.len >= 3:
            let (ptrPlus, lenExpr) = spanPtrLenParts(spanArg)
            newArgs.add ptrPlus
            newArgs.add lenExpr
          elif spanArg.kind == gpuIdent and spanArg.symbol.typ != nil and spanArg.symbol.typ.kind == gtSpan:
            newArgs.add GpuAst(kind: gpuIdent, symbol: newSymbol(spanArg.symbol.name, typ = calleeSpans[spanIdx].ptrTyp))
            newArgs.add GpuAst(kind: gpuIdent, symbol: newSymbol(spanArg.symbol.name & "_len", typ = calleeSpans[spanIdx].lenTyp))
          else:
            newArgs.add spanArg
          spanIdx += 1
        else:
          newArgs.add n.cArgs[argIdx]
        argIdx += 1
      n.cArgs = newArgs

    for child in mitems(n):
      child.rewriteViewUses(sigMap)
  else:
    for child in mitems(n):
      child.rewriteViewUses(sigMap)



# ═══════════════════════════════════════════════════════════════════════
# Span materialization
# ═══════════════════════════════════════════════════════════════════════

proc decomposePtrLen(n: var GpuAst): GpuAst =
  ## Decomposes `let s = toOpenArray(p, a, b)` in place:
  ## `s` holds `p + a`, the returned statement is `let s_len = b - a + 1`.
  if n.kind != gpuVar or n.vInit.kind != gpuCall:
    return nil
  # A `var` view that is never rebound is a `let`.
  let c = n.vInit
  if c.cName.kind != gpuIdent or c.cName.symbol.name != "toOpenArray" or c.cArgs.len < 3:
    return nil
  let (ptrPlus, lenExpr) = spanPtrLenParts(c)
  let ptrTyp = initGpuPtrType(n.vType.sElemTyp, implicitPtr = false, addressSpace = n.vType.addressSpace)
  n.vType = ptrTyp
  n.vInit = ptrPlus
  n.vName.symbol.typ = ptrTyp
  n.vMutable = false
  let lenSym = newSymbol(n.vName.symbol.name & "_len", iSym = n.vName.symbol.iSym & "_len",
                         typ = initGpuType(gtInt32), symKind = n.vName.symbol.symKind)
  result = GpuAst(kind: gpuVar,
                  vName: GpuAst(kind: gpuIdent, symbol: lenSym),
                  vType: initGpuType(gtInt32),
                  vInit: lenExpr,
                  vMutable: n.vMutable)

proc rewriteStmts(stmts: var seq[GpuAst], sigMap: SpanSigMap) =
  ## View bindings decompose at their own statement scope.
  ## Nested statement lists recurse.
  var outStmts: seq[GpuAst]
  for stmt in stmts.mitems:
    case stmt.kind
    of gpuBlock:
      rewriteStmts(stmt.statements, sigMap)
    of gpuIf:
      rewriteStmts(stmt.ifThen.statements, sigMap)
      for el in stmt.ifElifs.mitems:
        rewriteStmts(el.body.statements, sigMap)
      if stmt.ifElse.kind != gpuDiscard:
        rewriteStmts(stmt.ifElse.statements, sigMap)
    of gpuFor:
      rewriteStmts(stmt.fBody.statements, sigMap)
    of gpuWhile:
      rewriteStmts(stmt.wBody.statements, sigMap)
    else:
      discard
    let lenDecl = decomposePtrLen(stmt)
    if lenDecl.isNil:
      outStmts.add stmt
    else:
      outStmts.add stmt
      outStmts.add lenDecl
  stmts = outStmts

proc lowerArraysAndViews(ctx: var GpuContext, decayArrays: bool) =
  ## Arrays decay and openArray views split into ptr + len pairs.
  ## The pass emits no block expressions.
  ##
  ## | Before (IR)                    | After (IR)                                |
  ## | ------------------------------ | ----------------------------------------- |
  ## | `f(s: openArray[T])`           | `f(s: ptr T, s_len: int32)`               |
  ## | `f(toOpenArray(p, a, b))`      | `f(p + a, b - a + 1)`                     |
  ## | `let v = toOpenArray(p, a, b)` | `let v = p + a` + `let v_len = b - a + 1` |
  ## | `s.len` / `len(s)`             | `s_len`                                   |
  ## | `f(dst: var array[N, T])`      | `f(dst: ptr T)` (decayArrays = true)      |
  ##
  ## `var array[N, T]` / `var openArray[T]` params are element pointers (T*),
  ## matching Nim's C backend. The IR is gtPtr(gtArray) / gtPtr(gtSpan),
  ## HiddenDeref on access and HiddenAddr on call args, normalized to:
  ## - gtPtr(gtArray[N, T]) -> gtPtr(T) when decayArrays, gtPtr(gtSpan) -> gtSpan
  ## - index-context derefs fold into the index (`(*p)[i]` ≡ `p[i]`),
  ##   the pass keeps derefs under gpuAddr, `&(*p)` is the array address not `T**`
  ## - gpuAddr(array) call args become the bare array (C decay) for array-ptr params
  ## - gpuAddr(array) on a span param becomes toOpenArray(array, 0, len-1), inlined here
  ##
  ## The gpuAddr(gpuDot) case keys on the TYPE (gtArray), not the field name:
  ## any array-typed field passed by addr rewrites identically
  # Phase 1+2: param shape prep, index deref folding, view argument creation,
  # signature type normalization
  var calleeKinds = initTable[string, seq[ArraySpanParamKind]]()
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      calleeKinds[fn.pName.symbol.name] = collectArraySpanParamKinds(fn)
  for fnKey in ctx.genericInsts.keys:
    let fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      calleeKinds[fn.pName.symbol.name] = collectArraySpanParamKinds(fn)

  template normalizeFn(fn: var GpuAst) =
    normalizeArraySpanBody(fn.pBody, calleeKinds, decayArrays)
    normalizeArraySpanFn(fn, decayArrays)
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      normalizeFn(fn)
      ctx.allFnTab[fnKey] = fn
  for fnKey in ctx.genericInsts.keys:
    var fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      normalizeFn(fn)
      ctx.genericInsts[fnKey] = fn

  # Phase 3+4: split view signatures, rewrite bodies at statement scope.
  # An empty sigMap only skips the signature splits.
  # The local-view decomposition and span-use rewriting still run.
  let sigMap = collectSpanSigs(ctx)
  var rewritten = initHashSet[string]()
  template rewriteTab(tab: untyped) =
    var keys: seq[GpuAst]
    for k in tab.keys:
      keys.add k
    for k in keys:
      var fn = tab[k]
      if fn.isNil or fn.kind != gpuProc:
        continue
      if fn.pName.isNil or fn.pName.kind != gpuIdent:
        continue
      let nm = fn.pName.symbol.name
      if nm notin rewritten:
        rewritten.incl nm
        rewriteFnSig(fn, sigMap)
        rewriteStmts(fn.pBody.statements, sigMap)
        for i in 0 ..< fn.pBody.statements.len:
          fn.pBody.statements[i].rewriteViewUses(sigMap)
        tab[k] = fn
  rewriteTab(ctx.allFnTab)
  rewriteTab(ctx.genericInsts)
  rewriteTab(ctx.fnTab)

# ═══════════════════════════════════════════════════════════════════════
# Array/span parameter normalization
# ═══════════════════════════════════════════════════════════════════════

type
  ArraySpanParamKind = enum
    aspNone       ## not an array/span param
    aspArrayPtr   ## `var array[N, T]`, element pointer after normalization
    aspSpan       ## `var openArray[T]`, span (ptr + len) after normalization

proc collectArraySpanParamKinds(fn: GpuAst): seq[ArraySpanParamKind] =
  ## Original param kinds (before type normalization), by param position.
  result = newSeq[ArraySpanParamKind](fn.pParams.len)
  for i, p in fn.pParams:
    if p.typ.kind == gtSpan or (p.typ.kind == gtPtr and p.typ.to.kind == gtSpan):
      result[i] = aspSpan
    elif p.typ.kind == gtPtr and p.typ.to.kind == gtArray:
      result[i] = aspArrayPtr

proc isArrayOrSpanPtr(t: GpuType): bool =
  if t == nil:
    return false
  t.kind == gtSpan or
    (t.kind == gtPtr and (t.to.kind == gtArray or t.to.kind == gtSpan))

proc makeToOpenArray(arr: GpuAst, aLen: int): GpuAst =
  ## LowerSpans inlines this to ptr + len.
  let idxTyp = initGpuType(gtInt32)
  GpuAst(
    kind: gpuCall,
    cName: GpuAst(kind: gpuIdent, symbol: newSymbol("toOpenArray")),
    cArgs: @[
      arr,
      GpuAst(kind: gpuLit, lValue: "0", lType: idxTyp),
      GpuAst(kind: gpuLit, lValue: $(aLen - 1), lType: idxTyp)
    ]
  )

proc normalizeArraySpanBody(body: var GpuAst, calleeKinds: Table[string, seq[ArraySpanParamKind]], decayArrays = true) =
  ## Body rewrites for calls whose callee takes array or span pointers:
  ## - an index through a pointer folds into the index, `(*p)[i]` is `p[i]`
  ## - an `addr(arr)` argument rewrites to the callee's parameter shape:
  ##   pointer-to-array takes the bare array, openArray takes
  ##   `toOpenArray(arr, 0, len-1)`
  body.walk(proc(n: var GpuAst): void =
    case n.kind
    of gpuIndex:
      # index context: (*p)[i] ≡ p[i] when p is an array/span pointer
      if n.iArr.kind == gpuDeref and
         n.iArr.dOf.kind == gpuIdent and
         isArrayOrSpanPtr(n.iArr.dOf.symbol.typ):
        n.iArr = n.iArr.dOf
    of gpuCall:
      if n.cName.kind == gpuIdent:
        let kinds = calleeKinds.getOrDefault(n.cName.symbol.name)
        if kinds.len > 0:
          for i, arg in n.cArgs.mpairs:
            if i >= kinds.len:
              break
            if kinds[i] == aspNone:
              continue
            if kinds[i] == aspArrayPtr and not decayArrays:
              continue
            if arg.kind == gpuAddr:
              if arg.aOf.kind == gpuIdent:
                let t = arg.aOf.symbol.typ
                if t != nil and t.kind == gtArray:
                  if kinds[i] == aspArrayPtr:
                    arg = arg.aOf                # bare array, C decays to T*
                  elif kinds[i] == aspSpan:
                    arg = makeToOpenArray(arg.aOf, t.aLen)
              elif arg.aOf.kind == gpuDot:
                # var-array param passed a tensor's data field, same rewrite,
                # the bare field access C-decays to T*
                let t = arg.aOf.dField.symbol.typ
                if t != nil and t.kind == gtArray:
                  if kinds[i] == aspArrayPtr:
                    arg = arg.aOf
                  elif kinds[i] == aspSpan:
                    arg = makeToOpenArray(arg.aOf, t.aLen)
    else:
      discard
  )

func normalizeArraySpanFn(fn: var GpuAst, decayArrays: bool) =
  for p in fn.pParams.mitems:
    if p.flavor == kVar:
      # kVar params are references: by-value array/span normalization does not apply
      continue
    if p.typ.kind == gtPtr and p.typ.to.kind == gtArray:
      if decayArrays:
        p.typ = initGpuPtrType(p.typ.to.aTyp, implicitPtr = p.typ.implicit, addressSpace = p.typ.addressSpace)
        p.ident.symbol.typ = p.typ
    elif p.typ.kind == gtPtr and p.typ.to.kind == gtSpan:
      p.typ = p.typ.to
      p.ident.symbol.typ = p.typ
