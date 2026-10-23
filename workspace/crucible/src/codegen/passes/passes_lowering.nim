## Shared lowering impls.
##
## - registerSharedLoweringPasses registers the ones every backend wants
## - per-backend passes register in passes_lowering_<backend>.nim
## - each replaces an IR construct with its composition at the target
##   calling convention
## - a policy differing per target takes a flag

import std / [tables, sequtils, sets, strutils]
import std/macros # for the `warning` builtin (compile-time diagnostics)
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ../ir/ast_type_constructors
import ./pass_datatypes
import ./pass_registry
import ./passes_legalizations

# ############################################################
#
# Lowering
#
# ############################################################
#
# Lowering replaces IR constructs with their composition at the target
# calling convention: spans become ptr + len, compile-time params are erased.

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSharedLoweringPasses*(reg: var PassRegistry) =
  reg.registerPruneCompileTimeParams()
  reg.registerDecomposeMemcpyVars()

func registerRenameReservedIdentifiers*(reg: var PassRegistry, name: static string, reserved: static HashSet[string]) =
  reg.register(name, pkTransform, phaseMain,
    "Renames identifiers colliding with backend reserved words: fn names, call sites, params, globals, struct fields, field accesses",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.renameReservedIdentifiers(reserved)
  )

func registerInsertPointerDeref*(reg: var PassRegistry) =
  reg.register("insertPointerDeref", pkTransform, phaseMain,
    "Wraps pointer-typed index and field parents in explicit gpuDeref",
    dependsOn = @["rewriteIndexDeref"],
    run = insertPointerDeref
  )

func registerCollectValueAddressSpaces*(reg: var PassRegistry) =
  reg.register("collectValueAddressSpaces", pkTransform, phaseMain,
    "Records var and objconstr pointer-field address spaces for resolution",
    dependsOn = @[],
    run = collectValueAddressSpaces
  )

func registerLowerArraysAndViews*(reg: var PassRegistry, decayArrays: static bool) =
  reg.register("lowerArraysAndViews", pkTransform, phaseMain,
    "Arrays decay to element pointers per decayArrays, openArray views split into ptr + len",
    run = proc(ctx: var GpuContext): void =
      ctx.lowerArraysAndViews(decayArrays)
  )

func registerPruneCompileTimeParams(reg: var PassRegistry) =
  reg.register("pruneCompileTimeParams", pkTransform, phaseMain,
    "Drop compile-time-only parameters (static records, dead scalar statics, typedescs, phantom values) and erase their call arguments",
    run = pruneCompileTimeParams
  )

func registerDecomposeMemcpyVars(reg: var PassRegistry) =
  reg.register("decomposeMemcpyVars", pkTransform, phaseMain,
    "Splits gpuVar/gpuAssign with RequiresMemcpy into decl + memcpy call",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          decomposeMemcpyVarsImpl(ctx, fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          decomposeMemcpyVarsImpl(ctx, fn.pBody)
  )

func registerExpandStructKernelParams*(reg: var PassRegistry) =
  ## Metal/WebGPU/MSL expand, CUDA/OpenCL/GLSL cross structs natively
  ## (GLSL pointer fields cross as 64-bit buffer device addresses)
  reg.register(
    name = "explodeGlobalStructWithPtrFields",
    kind = pkTransform,
    phase = phaseMain,
    description = "Expands kernel params of structs carrying (nested) pointer fields into ptr/scalar leaves and rebuilds the struct in the body",
    dependsOn = @["ensureBlock", "pruneCompileTimeParams"],
    run = explodeGlobalStructWithPtrFields
  )

# ═══════════════════════════════════════════════════════════════════════
# Reserved keyword renaming
# ═══════════════════════════════════════════════════════════════════════

proc renameReservedKeywords(name: string, reserved: HashSet[string]): string =
  if name in reserved: name & "_nim"
  else: name

proc renameIdents(n: var GpuAst, symToRename: Table[string, string]) =
  ## Substitutes the names of identifiers whose iSym is in the table
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in symToRename:
      n.symbol.name = symToRename[n.symbol.iSym]
  of gpuCall:
    # the child iterator yields cArgs only, the callee ident is walked here
    renameIdents(n.cName, symToRename)
    for ch in mitems(n):
      renameIdents(ch, symToRename)
  else:
    for ch in mitems(n):
      renameIdents(ch, symToRename)

proc renameFieldAccesses(n: var GpuAst, reserved: HashSet[string]) =
  ## Renames field accesses whose name is a target keyword, to match `renameReservedIdentifiers`
  case n.kind
  of gpuDot:
    if n.dField.kind == gpuIdent:
      n.dField.symbol.name = renameReservedKeywords(n.dField.symbol.name, reserved)
    renameFieldAccesses(n.dParent, reserved)
  else:
    for ch in mitems(n):
      renameFieldAccesses(ch, reserved)

proc renameReservedIdentifiers(ctx: var GpuContext, reserved: HashSet[string]) =
  ## Renames identifiers colliding with target keywords at every emitted site:
  ## - fn names, call sites and fn params
  ## - globals, where WGSL lifts kernel params
  ## - struct fields, including block-wrapped type defs
  ## - field accesses
  var fnRenames = initTable[string, string]()
  for fn in ctx.allFnTab.values:
    let oldName = fn.pName.ident()
    let safeName = renameReservedKeywords(oldName, reserved)
    if oldName != safeName:
      fnRenames[fn.pName.symbol.iSym] = safeName
      fn.pName.symbol.name = safeName
  for fn in ctx.genericInsts.values:
    let oldName = fn.pName.ident()
    let safeName = renameReservedKeywords(oldName, reserved)
    if oldName != safeName:
      fnRenames[fn.pName.symbol.iSym] = safeName
      fn.pName.symbol.name = safeName
  for g in mvalues(ctx.globals):
    g.ident.symbol.name = renameReservedKeywords(g.ident.ident(), reserved)
  for tab in [ctx.allFnTab, ctx.genericInsts, ctx.fnTab]:
    for fn in tab.values:
      var renames = initTable[string, string]()
      for p in mitems(fn.pParams):
        let oldName = p.ident.ident()
        let safeName = renameReservedKeywords(oldName, reserved)
        if oldName != safeName:
          renames[p.ident.symbol.iSym] = safeName
          p.ident.symbol.name = safeName
      if renames.len > 0:
        renameIdents(fn.pBody, renames)
  for fn in ctx.allFnTab.values:
    renameIdents(fn.pBody, fnRenames)
  for fn in ctx.genericInsts.values:
    renameIdents(fn.pBody, fnRenames)
  for fn in ctx.fnTab.values:
    renameIdents(fn.pBody, fnRenames)
  proc renameFieldDecls(n: var GpuAst) =
    case n.kind
    of gpuTypeDef:
      for f in mitems(n.tFields):
        f.name = renameReservedKeywords(f.name, reserved)
    else:
      for ch in mitems(n):
        renameFieldDecls(ch)
  for blk in mitems(ctx.globalBlocks):
    renameFieldDecls(blk)
  for _, typ in mpairs(ctx.types):
    # field decls live in the type registry before the target preprocess
    # farms them into globalBlocks
    renameFieldDecls(typ)
  for tab in [ctx.allFnTab, ctx.genericInsts, ctx.fnTab]:
    for fn in tab.values:
      renameFieldAccesses(fn.pBody, reserved)
  for blk in mitems(ctx.globalBlocks):
    renameFieldAccesses(blk, reserved)

# ═══════════════════════════════════════════════════════════════════════
# Pointer deref insertion
# ═══════════════════════════════════════════════════════════════════════

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
      let t = ctx.getOperandType(n.iArr)
      if not t.isNil and t.kind == gtPtr and not t.to.isNil:
        n.iArr = GpuAst(kind: gpuDeref, dOf: n.iArr)
  of gpuDot:
    insertPointerDerefImpl(ctx, n.dParent)
    insertPointerDerefImpl(ctx, n.dField)
    if n.dParent.kind != gpuDeref:
      let t = ctx.getOperandType(n.dParent)
      if not t.isNil and t.kind == gtPtr:
        n.dParent = GpuAst(kind: gpuDeref, dOf: n.dParent)
  of gpuDeref:
    # Recurse into the deref'd subtree only: this node is already the wrap
    # for its operand. A re-wrap chains derefs and adds one extra `.x[0]`
    # per visit in the backends that render them
    insertPointerDerefImpl(ctx, n.dOf)
  else:
    for ch in mitems(n):
      insertPointerDerefImpl(ctx, ch)

proc insertPointerDeref(ctx: var GpuContext) =
  for _, fn in ctx.allFnTab.mpairs:
    insertPointerDerefImpl(ctx, fn.pBody)

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
# Memcpy decomposition
# ═══════════════════════════════════════════════════════════════════════

proc renameSymsInTree(n: var GpuAst, oldName, newName: string) =
  if n == nil:
    return
  case n.kind
  of gpuIdent:
    if n.symbol.name == oldName:
      n.symbol.name = newName
      n.symbol.iSym = newName
  of gpuVar:
    # mitems does not yield vName
    if n.vName.symbol.name == oldName:
      n.vName.symbol.name = newName
      n.vName.symbol.iSym = newName
    renameSymsInTree(n.vInit, oldName, newName)
  else:
    for child in n.mitems:
      renameSymsInTree(child, oldName, newName)

proc dedupVarNames(stmts: var seq[GpuAst]) =
  ## Renames duplicate gpuVar declarations across sibling blocks,
  ## template expansions collide without C++ scope isolation
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

proc genMemcpyCall(lhs, rhs: GpuAst, sizeExpr: GpuAst): GpuAst =
  let memcpySym = newSymbol("__builtin_memcpy", iSym = "__builtin_memcpy", symKind = gsProc)
  let memcpyIdent = GpuAst(kind: gpuIdent, symbol: memcpySym)
  var call = GpuAst(kind: gpuCall)
  call.cName = memcpyIdent
  call.cArgs = @[lhs, rhs, sizeExpr]
  result = call

func genSizeofCall(ctx: var GpuContext, expr: GpuAst): GpuAst =
  let sizeSym = newSymbol("__builtin_sizeof", iSym = "__builtin_sizeof", symKind = gsProc)
  let sizeIdent = GpuAst(kind: gpuIdent, symbol: sizeSym)
  var call = GpuAst(kind: gpuCall)
  call.cName = sizeIdent
  call.cArgs = @[expr]
  result = call

proc decomposeMemcpyVarsImpl(ctx: var GpuContext, n: var GpuAst) =
  ## After this pass, backends only emit `=` for simple assignments
  case n.kind
  of gpuVar:
    if n.vRequiresMemcpy and n.vInit.kind != gpuDiscard:
      let addrOfVar = GpuAst(kind: gpuAddr, aOf: n.vName)
      let addrOfInit = GpuAst(kind: gpuAddr, aOf: n.vInit)
      let sizeCall = genSizeofCall(ctx, n.vName)
      let memcpyNode = genMemcpyCall(addrOfVar, addrOfInit, sizeCall)

      n.vRequiresMemcpy = false
      n.vInit = GpuAst(kind: gpuDiscard)
      n = GpuAst(kind: gpuBlock, blockLabel: "_memcpy", statements: @[n, memcpyNode])
    else:
      for ch in mitems(n):
        decomposeMemcpyVarsImpl(ctx, ch)

  of gpuAssign:
    if n.aRequiresMemcpy:
      let addrOfLeft = GpuAst(kind: gpuAddr, aOf: n.aLeft)
      let addrOfRight = GpuAst(kind: gpuAddr, aOf: n.aRight)
      let sizeCall = genSizeofCall(ctx, n.aLeft)
      n = genMemcpyCall(addrOfLeft, addrOfRight, sizeCall)
    else:
      for ch in mitems(n):
        decomposeMemcpyVarsImpl(ctx, ch)

  of gpuIf:
    decomposeMemcpyVarsImpl(ctx, n.ifCond)
    decomposeMemcpyVarsImpl(ctx, n.ifThen)
    for el in n.ifElifs.mitems:
      decomposeMemcpyVarsImpl(ctx, el.cond)
      decomposeMemcpyVarsImpl(ctx, el.body)
    if n.ifElse.kind != gpuDiscard:
      decomposeMemcpyVarsImpl(ctx, n.ifElse)

  of gpuFor:
    decomposeMemcpyVarsImpl(ctx, n.fBody)

  of gpuWhile:
    decomposeMemcpyVarsImpl(ctx, n.wBody)

  of gpuBlock:
    var newStmts: seq[GpuAst]
    for i in 0 ..< n.statements.len:
      var stmt = n.statements[i]
      decomposeMemcpyVarsImpl(ctx, stmt)
      if stmt.kind == gpuBlock and stmt.blockLabel == "_memcpy":
        # only anonymous _memcpy blocks inline, labeled blocks keep C++ scope isolation
        for s in stmt.statements:
          newStmts.add s
      else:
        newStmts.add stmt
    n.statements = newStmts
    # template expansions duplicate var names across blocks flattened by codegen
    dedupVarNames(n.statements)

  of gpuProc:
    decomposeMemcpyVarsImpl(ctx, n.pBody)

  else:
    for ch in mitems(n):
      decomposeMemcpyVarsImpl(ctx, ch)

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
# Compile-time parameter pruning
# ═══════════════════════════════════════════════════════════════════════

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

proc pruneCompileTimeParams(ctx: var GpuContext) =
  ## Drops parameters whose type has no runtime bytes: compile-time
  ## arguments and phantoms, the value lives in the type
  ## Erases the matching call arguments.
  var drops: Table[string, seq[int]]
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind != gpuProc:
      continue
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

# ═══════════════════════════════════════════════════════════════════════
# Struct kernel-param expansion
# ═══════════════════════════════════════════════════════════════════════

func containsPtrDeep(t: GpuType): bool =
  ## True when a pointer is reachable through the type's fields
  case t.kind
  of gtPtr: true
  of gtObject, gtGenericInst:
    for f in t.fields:
      if f.typ.containsPtrDeep:
        return true
    false
  of gtArray: t.aTyp.containsPtrDeep
  else: false

type KernelParamLeaf = object
  ## One signature entry of an expanded struct kernel param
  name: string
  typ: GpuType
  isBuffer: bool

proc kernelParamLeaves(t: GpuType, name: string): seq[KernelParamLeaf] =
  ## Metal/WebGPU cannot take struct-with-pointer kernel args, the signature
  ## expands depth-first in field order, matching the serializer binding order
  result = newSeq[KernelParamLeaf]()
  proc walk(t: GpuType, name: string, leaves: var seq[KernelParamLeaf]) =
    case t.kind
    of gtPtr:
      leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: true)
    of gtObject, gtGenericInst:
      if t.containsPtrDeep:
        for f in t.fields:
          if size(f.typ) > 0:
            walk(f.typ, name & "_" & f.name, leaves)
      else:
        leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
    of gtArray:
      doAssert not t.containsPtrDeep,
        "kernelParamLeaves: array field '" & name & "' carries pointers, not supported as a kernel param leaf"
      leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
    else:
      leaves.add KernelParamLeaf(name: name, typ: t, isBuffer: false)
  walk(t, name, result)

proc buildLeafConstr(t: GpuType, name: string): GpuAst =
  result = GpuAst(kind: gpuObjConstr, ocType: t)
  for f in t.fields:
    if size(f.typ) == 0:
      result.ocFields.add GpuFieldInit(name: f.name, value: GpuAst(kind: gpuDiscard), typ: f.typ)
    elif f.typ.kind in {gtObject, gtGenericInst} and f.typ.containsPtrDeep:
      result.ocFields.add GpuFieldInit(name: f.name, value: buildLeafConstr(f.typ, name & "_" & f.name), typ: f.typ)
    else:
      result.ocFields.add GpuFieldInit(name: f.name, value: newGpuIdent(name & "_" & f.name), typ: f.typ)

proc explodeGlobalStructWithPtrFields(ctx: var GpuContext) =
  for _, fn in ctx.allFnTab.mpairs:
    if fn.kind != gpuProc or attGlobal notin fn.pAttributes:
      continue
    var taken: seq[string]
    for p in fn.pParams:
      taken.add p.ident.ident()
    var params: seq[GpuParam]
    for p in fn.pParams:
      if p.typ.kind in {gtObject, gtGenericInst} and p.typ.containsPtrDeep:
        let leaves = p.typ.kernelParamLeaves(p.ident.ident())
        for leaf in leaves:
          doAssert leaf.name notin taken,
            "explodeGlobalStructWithPtrFields: leaf '" & leaf.name & "' collides with an existing kernel parameter"
          let sym = newSymbol(leaf.name, typ = leaf.typ, symKind = gsGlobalKernelParam)
          let space = if leaf.isBuffer: asDevice else: asRMEM
          params.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym),
                              typ: leaf.typ, addressSpace: space)
        let constr = buildLeafConstr(p.typ, p.ident.ident())
        let local = GpuAst(kind: gpuVar, vName: newGpuIdent(p.ident.ident()),
                           vType: p.typ, vInit: constr, vMutable: false,
                           addressSpace: asRMEM)
        doAssert fn.pBody.kind == gpuBlock, "explodeGlobalStructWithPtrFields: kernel body is not a block"
        fn.pBody.statements.insert(local, 0)
      else:
        params.add p
    fn.pParams = params

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

