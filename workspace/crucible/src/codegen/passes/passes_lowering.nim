## Lowering pass: materialize gtSpan into ptr + len
##
## 1. Collect functions with gtSpan params → build replacement map
## 2. Rewrite function signatures: split gtSpan params into ptr + len
## 3. Rewrite all function bodies:
##    - gpuIdent with gtSpan type → ptr ident
##    - gpuDot(span, "len") → len ident
##    - gpuCall("len", [span]) → len ident
##    - Call sites: split gtSpan args into (ptr_arg, len_arg)

import std / [tables, sequtils, sets, strutils]
import std/macros # for the `warning` builtin (compile-time diagnostics)
import ../ir/gpu_types
import ../ir/gpu_type_constructors
import ./pass_datatypes

type
  SpanParam = object
    paramIdx: int        # index in the param list
    ptrName: string      # name for the ptr param
    lenName: string      # name for the len param
    ptrTyp: GpuType      # ptr to element type
    lenTyp: GpuType      # int32

  SpanSigMap = TableRef[string, seq[SpanParam]]

proc exprGpuType(n: GpuAst): GpuType =
  ## Best-effort GpuType of a lowering expression (span ptr/len args).
  ## The lowering sites build fresh gpuBinOps whose bType must be non-nil:
  ## the tail of the gpuBlock(isExpr: true) built in rewriteNode below is
  ## read by getExprType (the lenExpr binop), so it must carry a real type.
  ## A nil/void derivation is as unusable as an unknown shape: the constructed
  ## `+ 1` literal's lType is this type (genLit derefs it at emission — a nil
  ## GpuType there raises a FieldDefect) and the lenExpr binop's bType is read
  ## by getExprType. Route everything through the one loud gtInt32 fallback so
  ## exprGpuType can never hand back nil to the lowering sites.
  case n.kind
  of gpuIdent: result = n.symbol.typ
  of gpuLit: result = n.lType
  of gpuBinOp: result = n.bType
  of gpuConv: result = n.convTo
  else:
    discard
  if result.isNil or result.kind == gtVoid:
    # Span ptr/len args are ints in practice, but a silent wrong-width
    # assumption must not pass unnoticed — surface it when this fires.
    result = initGpuType(gtInt32)
    warning "[lowerSpans] WARNING: exprGpuType could not derive a type for " & $n.kind &
      "; assuming gtInt32 for span index/pointer arithmetic"

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
      if p.typ.kind == gtSpan:
        spans.add SpanParam(
          paramIdx: idx,
          ptrName: p.ident.symbol.name,
          lenName: p.ident.symbol.name & "_len",
          ptrTyp: initGpuPtrType(p.typ.sElemTyp, implicitPtr = false),
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

proc rewriteFnSig(fn: var GpuAst; sigMap: SpanSigMap) =
  ## Split gtSpan params in a function definition
  if fn.isNil or fn.kind != gpuProc:
    return
  if fn.pName.isNil or fn.pName.kind != gpuIdent:
    return
  let fnName = fn.pName.symbol.name
  if fnName.len == 0: return
  let spans = sigMap.getOrDefault(fnName)
  if spans.len == 0: return

  var newParams: seq[GpuParam]
  var skipMask: seq[bool]
  skipMask.setLen(fn.pParams.len)

  for s in spans:
    skipMask[s.paramIdx] = true

  for idx, p in fn.pParams:
    if idx < skipMask.len and skipMask[idx]:
      # Find the span info for this param
      for s in spans:
        if s.paramIdx == idx:
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

proc rewriteNode(n: var GpuAst; sigMap: SpanSigMap) =
  ## Walk AST, rewriting span idents and adjusting call sites

  case n.kind
  of gpuIdent:
    # Span-typed ident → ptr ident (same name, ptr type)
    if n.symbol.typ != nil and n.symbol.typ.kind == gtSpan:
      n = GpuAst(kind: gpuIdent, symbol: newSymbol(n.symbol.name, typ = initGpuPtrType(n.symbol.typ.sElemTyp, implicitPtr = false), symKind = n.symbol.symKind))
  of gpuDot:
    # span.len → len ident
    if n.dField.kind == gpuIdent and n.dField.symbol.name == "len" and
       n.dParent.symbol.typ != nil and n.dParent.symbol.typ.kind == gtSpan:
      n = GpuAst(kind: gpuIdent,
                 symbol: newSymbol(n.dParent.symbol.name & "_len", typ = initGpuType(gtInt32)))
      return
    else:
      for child in mitems(n):
        child.rewriteNode(sigMap)
  of gpuCall:
    var fnName: string
    if n.cName.kind == gpuIdent:
      fnName = n.cName.symbol.name

    # Handle builtin calls: len(span) → span_len
    if fnName == "len" or fnName.startsWith("len_"):
      if n.cArgs.len >= 1:
        if n.cArgs[0].kind == gpuIdent and
           n.cArgs[0].symbol.typ != nil and n.cArgs[0].symbol.typ.kind == gtSpan:
          n = GpuAst(kind: gpuIdent,
                     symbol: newSymbol(n.cArgs[0].symbol.name & "_len", typ = initGpuType(gtInt32)))
          return
        # len(toOpenArray_lowered_block) → statements[1] (the len expr)
        if n.cArgs[0].kind == gpuBlock and n.cArgs[0].isExpr and
           n.cArgs[0].statements.len >= 2:
          n = n.cArgs[0].statements[1].clone()
          return
        # len(toOpenArray(ptr, first, last)) → last - first + 1
        if n.cArgs[0].kind == gpuCall and
           n.cArgs[0].cName.kind == gpuIdent and
           (n.cArgs[0].cName.symbol.name == "toOpenArray" or n.cArgs[0].cName.symbol.name.startsWith("toOpenArray_")) and
           n.cArgs[0].cArgs.len >= 3:
          let tc = n.cArgs[0]
          let idxTyp = tc.cArgs[1].exprGpuType()
          let lenSub = GpuAst(kind: gpuBinOp,
                              bType: idxTyp,
                              bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("-")),
                              bLeft: tc.cArgs[2],
                              bRight: tc.cArgs[1])
          n = GpuAst(kind: gpuBinOp,
                     bType: idxTyp,
                     bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+")),
                     bLeft: lenSub,
                     bRight: GpuAst(kind: gpuLit, lValue: "1",
                                    lType: (if idxTyp.isNil: initGpuType(gtInt32) else: idxTyp)))
          # exprGpuType never returns nil, so idxTyp is taken above; the gtInt32
          # arm is a defensive guard against genLit's nil-lType FieldDefect
          return
      return

    # Recurse into children before handling toOpenArray
    for child in mitems(n):
      child.rewriteNode(sigMap)

    # Handle toOpenArray: inline as ptr + first
    if (fnName == "toOpenArray" or fnName.startsWith("toOpenArray_")) and n.cArgs.len >= 3:
      # toOpenArray(ptr, first, last) →
      #   { ptr + first, last - first + 1 } as two separate values
      # Inline: create a gpuBinOp for ptr+first
      let dataExpr = if n.cArgs[0].kind == gpuBlock and n.cArgs[0].isExpr and n.cArgs[0].statements.len >= 2:
                       n.cArgs[0].statements[0]
                     else:
                       n.cArgs[0]
      let ptrPlus = GpuAst(kind: gpuBinOp,
                           bType: dataExpr.exprGpuType(),
                           bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+")),
                           bLeft: dataExpr,
                           bRight: n.cArgs[1])
      let idxTyp = n.cArgs[1].exprGpuType()
      let lenSub = GpuAst(kind: gpuBinOp,
                          bType: idxTyp,
                          bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("-")),
                          bLeft: n.cArgs[2],
                          bRight: n.cArgs[1])
      let lenExpr = GpuAst(kind: gpuBinOp,
                           bType: idxTyp,
                           bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+")),
                           bLeft: lenSub,
                           bRight: GpuAst(kind: gpuLit, lValue: "1",
                                          lType: (if idxTyp.isNil: initGpuType(gtInt32) else: idxTyp)))
      # exprGpuType never returns nil, so idxTyp is taken above; the gtInt32
      # arm is a defensive guard against genLit's nil-lType FieldDefect
      # Return as block containing both expressions
      n = GpuAst(kind: gpuBlock, isExpr: true, statements: @[ptrPlus, lenExpr])
      return

    # Recurse into children first (so toOpenArray in args is transformed)
    for child in mitems(n):
      child.rewriteNode(sigMap)

    # Then handle callee span params
    var calleeSpans = sigMap.getOrDefault(fnName)
    if calleeSpans.len == 0:
      # Try matching by prefix (mangled generic names)
      for k, v in sigMap:
        if fnName.startsWith(k):
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
          if spanArg.kind == gpuBlock and spanArg.isExpr and spanArg.statements.len >= 2:
            newArgs.add spanArg.statements[0]
            newArgs.add spanArg.statements[1]
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

    # Recurse into args
    for child in mitems(n):
      child.rewriteNode(sigMap)
  else:
    for child in mitems(n):
      child.rewriteNode(sigMap)

proc lowerSpans*(ctx: var GpuContext) =
  let sigMap = collectSpanSigs(ctx)
  if sigMap.len == 0: return

  # Rewrite function signatures (dedup by name)
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
        fn.pBody.rewriteNode(sigMap)
        tab[k] = fn

  rewriteTab(ctx.allFnTab)
  rewriteTab(ctx.genericInsts)
  rewriteTab(ctx.fnTab)


# ═══════════════════════════════════════════════════════════════════
# Compile-time parameter pruning
# ═══════════════════════════════════════════════════════════════════

proc eraseCallArgs(n: var GpuAst, callee: string, dropped: seq[int]) =
  ## Removes the arguments at `dropped` positions from every call to `callee`
  if n.kind == gpuCall and n.cName.symbol.name == callee:
    var kept: seq[GpuAst]
    for i, arg in n.cArgs:
      if i notin dropped:
        kept.add arg
    n.cArgs = kept
  for child in n.mitems:
    child.eraseCallArgs(callee, dropped)

proc pruneCompileTimeParams*(ctx: var GpuContext) =
  ## Drops parameters whose type has no runtime bytes: compile-time
  ## arguments and phantoms, the value lives in the type
  ## Erases the matching call arguments.
  var drops: Table[string, seq[int]]
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind != gpuProc: continue
    var dropped: seq[int]
    for i, p in fn.pParams:
      if size(p.typ) == 0:
        # no runtime representation: compile-time args and phantoms
        dropped.add i
    if dropped.len > 0:
      drops[fn.pName.symbol.name] = dropped
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    for callee, dropped in drops:
      fn.pBody.eraseCallArgs(callee, dropped)
    if fn.pName.symbol.name in drops:
      let dropped = drops[fn.pName.symbol.name]
      var kept: seq[GpuParam]
      for i, p in fn.pParams:
        if i notin dropped:
          kept.add p
      fn.pParams = kept

proc registerTypeSystemPasses*(reg: var PassRegistry) =
  ## Passes that lower the Nim type system onto the target: compile-time
  ## parameters have no representation there. Run before the legalizations.
  reg.register(
    name = "pruneCompileTimeParams",
    kind = pkTransform,
    phase = phaseMain,
    description = "Drop compile-time-only parameters (static records, dead scalar statics, typedescs, phantom values) and erase their call arguments",
    run = pruneCompileTimeParams
  )

proc getOperandType(n: GpuAst): GpuType =
  ## Type of an expression for deref insertion, nil when undeterminable:
  ## synthesized idents and blit temps may carry no type, those are never pointers.
  case n.kind
  of gpuIdent:
    if n.symbol != nil: n.symbol.typ else: nil
  of gpuLit: n.lType
  of gpuCast: n.cTo
  of gpuConv: n.convTo
  of gpuObjConstr: n.ocType
  of gpuDeref:
    let d = getOperandType(n.dOf)
    if not d.isNil and d.kind == gtPtr: d.to else: d
  of gpuIndex:
    let t = getOperandType(n.iArr)
    if t.isNil: nil
    else:
      case t.kind
      of gtArray: t.aTyp
      of gtUA: t.uaTo
      of gtPtr: t.to
      else: nil
  of gpuDot:
    # The dot's type is the type of the field: the parser records each
    # identifier's type on its symbol, so the field ident already knows it
    if n.dField.kind == gpuIdent and n.dField.symbol != nil and
        n.dField.symbol.typ != nil:
      n.dField.symbol.typ
    else:
      nil
  else:
    nil

proc insertPointerDerefImpl(ctx: var GpuContext, n: var GpuAst) =
  ## Wraps pointer-typed expressions in explicit `gpuDeref` at value positions:
  ##
  ## - array indexing over a pointer to an array
  ## - field access over a pointer
  ## - deref over a pointer
  ##
  ## Address-of subtrees are untouched, the reference itself is the value there.
  ## Backends that cannot index pointers render the deref as a block-member access, C-family backends collapse it.
  case n.kind
  of gpuAddr:
    discard
  of gpuIndex:
    insertPointerDerefImpl(ctx, n.iArr)
    insertPointerDerefImpl(ctx, n.iIndex)
    if n.iArr.kind != gpuDeref:
      let t = getOperandType(n.iArr)
      if not t.isNil and t.kind == gtPtr and not t.to.isNil:
        n.iArr = GpuAst(kind: gpuDeref, dOf: n.iArr)
  of gpuDot:
    insertPointerDerefImpl(ctx, n.dParent)
    insertPointerDerefImpl(ctx, n.dField)
    if n.dParent.kind != gpuDeref:
      let t = getOperandType(n.dParent)
      if t != nil and t.kind == gtPtr:
        n.dParent = GpuAst(kind: gpuDeref, dOf: n.dParent)
  of gpuDeref:
    # Recurse into the deref'd subtree only: this node is already the wrap
    # for its operand. A re-wrap chains derefs and adds one extra `.x[0]`
    # per visit in the backends that render them
    insertPointerDerefImpl(ctx, n.dOf)
  else:
    for ch in mitems(n):
      insertPointerDerefImpl(ctx, ch)

proc insertPointerDeref*(ctx: var GpuContext) =
  for _, fn in ctx.allFnTab.mpairs:
    insertPointerDerefImpl(ctx, fn.pBody)

proc registerLoweringPasses*(reg: var PassRegistry) =
  reg.register(
    name = "lowerSpans",
    kind = pkTransform,
    phase = phaseMain,
    description = "Materialize openArray/varargs spans into ptr + len parameters",
    run = lowerSpans
  )
