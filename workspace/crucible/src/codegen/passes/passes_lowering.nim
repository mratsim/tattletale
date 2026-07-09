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
          n = GpuAst(kind: gpuBinOp,
                     bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+")),
                     bLeft: GpuAst(kind: gpuBinOp,
                                   bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("-")),
                                   bLeft: tc.cArgs[2],
                                   bRight: tc.cArgs[1]),
                     bRight: GpuAst(kind: gpuLit, lValue: "1",
                                    lType: initGpuType(gtInt32)))
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
                           bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+")),
                           bLeft: dataExpr,
                           bRight: n.cArgs[1])
      let lenExpr = GpuAst(kind: gpuBinOp,
                           bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+")),
                           bLeft: GpuAst(kind: gpuBinOp,
                                         bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("-")),
                                         bLeft: n.cArgs[2],
                                         bRight: n.cArgs[1]),
                           bRight: GpuAst(kind: gpuLit, lValue: "1",
                                          lType: initGpuType(gtInt32)))
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

proc registerLoweringPasses*(reg: var PassRegistry) =
  reg.register(
    name = "lowerSpans",
    kind = pkTransform,
    phase = phaseMain,
    description = "Materialize openArray/varargs spans into ptr + len parameters",
    run = lowerSpans
  )
