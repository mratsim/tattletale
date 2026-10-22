## Shared lowering impls.
##
## - registerSharedLoweringPasses registers the ones every backend wants
## - backends cherry-pick the rest from passes_lowering_<backend>.nim
## - each replaces an IR construct with its composition at the target
##   calling convention
## - a policy differing per target takes a flag

import std / [tables, sequtils, sets, strutils]
import std/macros # for the `warning` builtin (compile-time diagnostics)
import ../ir/gpu_ast_datatypes
import ../ir/gpu_ast_type_resolvers
import ../ir/gpu_type_constructors
import ./pass_datatypes
import ./pass_registry

# ############################################################
#
# Lowering
#
# ############################################################
#
# Lowering replaces IR constructs with their composition at the target
# calling convention: spans become ptr + len, compile-time params are erased.
# The universal ones register via registerSharedLoweringPasses, backends
# cherry-pick the rest. A policy that differs per target takes a flag.

{.experimental: "codeReordering".}

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

func registerPruneCompileTimeParams*(reg: var PassRegistry) =
  reg.register("pruneCompileTimeParams", pkTransform, phaseMain,
    "Drop compile-time-only parameters (static records, dead scalar statics, typedescs, phantom values) and erase their call arguments",
    run = pruneCompileTimeParams
  )

func registerLowerSpans*(reg: var PassRegistry) =
  reg.register("lowerSpans", pkTransform, phaseMain,
    "Materialize openArray/varargs spans into ptr + len parameters",
    run = lowerSpans
  )

func registerSharedLoweringPasses*(reg: var PassRegistry) =
  ## Registers the lowering passes common to every backend
  reg.registerPruneCompileTimeParams()
  reg.registerLowerSpans()

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

proc rewriteNode(n: var GpuAst, sigMap: SpanSigMap) =
  ## Walk AST, rewriting span idents and adjusting call sites

  case n.kind
  of gpuIdent:
    if n.symbol.typ != nil and n.symbol.typ.kind == gtSpan:
      # Span-typed ident → ptr ident (same name, ptr type)
      n = GpuAst(kind: gpuIdent, symbol: newSymbol(n.symbol.name, typ = initGpuPtrType(n.symbol.typ.sElemTyp, implicitPtr = false), symKind = n.symbol.symKind))
  of gpuDot:
    if n.dField.kind == gpuIdent and n.dField.symbol.name == "len" and
       n.dParent.symbol.typ != nil and n.dParent.symbol.typ.kind == gtSpan:
      # span.len → len ident
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

    if fnName == "len" or fnName.startsWith("len_"):
      if n.cArgs.len >= 1:
        if n.cArgs[0].kind == gpuIdent and
           n.cArgs[0].symbol.typ != nil and n.cArgs[0].symbol.typ.kind == gtSpan:
          # Handle builtin calls: len(span) → span_len
          n = GpuAst(kind: gpuIdent,
                     symbol: newSymbol(n.cArgs[0].symbol.name & "_len", typ = initGpuType(gtInt32)))
          return
        if n.cArgs[0].kind == gpuBlock and n.cArgs[0].isExpr and
           n.cArgs[0].statements.len >= 2:
          # len(toOpenArray_lowered_block) → statements[1] (the len expr)
          n = n.cArgs[0].statements[1].clone()
          return
        if n.cArgs[0].kind == gpuCall and
           n.cArgs[0].cName.kind == gpuIdent and
           (n.cArgs[0].cName.symbol.name == "toOpenArray" or n.cArgs[0].cName.symbol.name.startsWith("toOpenArray_")) and
           n.cArgs[0].cArgs.len >= 3:
          # len(toOpenArray(ptr, first, last)) → last - first + 1
          let tc = n.cArgs[0]
          let idxTyp = tc.cArgs[1].getLoweringExprType()
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
          # getLoweringExprType never returns nil so idxTyp is taken above,
          # the gtInt32 variant guards genLit's nil-lType FieldDefect
          return
      return

    for child in mitems(n):
      # Recurse into children before handling toOpenArray
      child.rewriteNode(sigMap)

    if (fnName == "toOpenArray" or fnName.startsWith("toOpenArray_")) and n.cArgs.len >= 3:
      # Handle toOpenArray: inline as ptr + first
      # toOpenArray(ptr, first, last) →
      #   { ptr + first, last - first + 1 } as two separate values
      # Inline: create a gpuBinOp for ptr+first
      let dataExpr = if n.cArgs[0].kind == gpuBlock and n.cArgs[0].isExpr and n.cArgs[0].statements.len >= 2:
                       n.cArgs[0].statements[0]
                     else:
                       n.cArgs[0]
      let ptrPlus = GpuAst(kind: gpuBinOp,
                           bType: dataExpr.getLoweringExprType(),
                           bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("+")),
                           bLeft: dataExpr,
                           bRight: n.cArgs[1])
      let idxTyp = n.cArgs[1].getLoweringExprType()
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
      # getLoweringExprType never returns nil so idxTyp is taken above,
      # the gtInt32 variant guards genLit's nil-lType FieldDefect
      # Return as block containing both expressions
      n = GpuAst(kind: gpuBlock, isExpr: true, statements: @[ptrPlus, lenExpr])
      return

    for child in mitems(n):
      # Recurse into children first (so toOpenArray in args is transformed)
      child.rewriteNode(sigMap)

    # Then handle callee span params
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

    for child in mitems(n):
      # Recurse into args
      child.rewriteNode(sigMap)
  else:
    for child in mitems(n):
      child.rewriteNode(sigMap)

# ═══════════════════════════════════════════════════════════════════════
# Span materialization
# ═══════════════════════════════════════════════════════════════════════

proc lowerSpans(ctx: var GpuContext) =
  let sigMap = collectSpanSigs(ctx)
  if sigMap.len == 0:
    return

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
# Struct kernel-param expansion (cherry-pickable)
# ═══════════════════════════════════════════════════════════════════════
#
# Expands kernel params of structs carrying (nested) pointer fields
# into ptr/scalar leaves and rebuilds the struct in the body.
# Metal, WebGPU, OpenCL and MSL register it. CUDA and GLSL
# cross structs natively.

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
  name*: string
  typ*: GpuType
  isBuffer*: bool

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

proc expandStructKernelParamsImpl(ctx: var GpuContext) =
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
            "expandStructKernelParams: leaf '" & leaf.name & "' collides with an existing kernel parameter"
          let sym = newSymbol(leaf.name, typ = leaf.typ, symKind = gsGlobalKernelParam)
          let space = if leaf.isBuffer: asDevice else: asRMEM
          params.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: sym),
                              typ: leaf.typ, addressSpace: space)
        let constr = buildLeafConstr(p.typ, p.ident.ident())
        let local = GpuAst(kind: gpuVar, vName: newGpuIdent(p.ident.ident()),
                           vType: p.typ, vInit: constr, vMutable: false,
                           addressSpace: asRMEM)
        doAssert fn.pBody.kind == gpuBlock, "expandStructKernelParams: kernel body is not a block"
        fn.pBody.statements.insert(local, 0)
      else:
        params.add p
    fn.pParams = params


func registerExpandStructKernelParams*(reg: var PassRegistry) =
  ## Metal/WebGPU/MSL expand, CUDA/OpenCL/GLSL cross structs natively
  ## (GLSL pointer fields cross as 64-bit buffer device addresses)
  reg.register(
    name = "expandStructKernelParams",
    kind = pkTransform,
    phase = phaseMain,
    description = "Expands kernel params of structs carrying (nested) pointer fields into ptr/scalar leaves and rebuilds the struct in the body",
    dependsOn = @["ensureBlock", "pruneCompileTimeParams"],
    run = expandStructKernelParamsImpl
  )

# ═══════════════════════════════════════════════════════════════════════
# Array/span parameter normalization (cherry-pickable)
# ═══════════════════════════════════════════════════════════════════════
#
# `var array[N, T]` / `var openArray[T]` params are element pointers (T*), matching Nim's C backend.
# The IR is gtPtr(gtArray) / gtPtr(gtSpan), HiddenDeref on access and HiddenAddr on call args, normalized to:
# - gtPtr(gtArray[N, T]) -> gtPtr(T) when decayArrays, gtPtr(gtSpan) -> gtSpan
# - index-context derefs fold into the index (`(*p)[i]` ≡ `p[i]`),
#    the pass keeps derefs under gpuAddr, `&(*p)` is the array address not `T**`
# - gpuAddr(array) call args become the bare array (C decay) for array-ptr params
# - gpuAddr(array) on a span param becomes toOpenArray(array, 0, len-1), lowerSpans inlines
#
# The gpuAddr(gpuDot) case keys on the TYPE (gtArray), not the field name:
# any array-typed field passed by addr rewrites identically

type
  ArraySpanParamKind* = enum
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
  ## Folds index-context derefs on array/span pointers into the index,
  ## rewrites gpuAddr(array) call args per the callee's param kind
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
        p.typ = initGpuPtrType(p.typ.to.aTyp, implicitPtr = p.typ.implicit)
        p.ident.symbol.typ = p.typ
    elif p.typ.kind == gtPtr and p.typ.to.kind == gtSpan:
      p.typ = p.typ.to
      p.ident.symbol.typ = p.typ

# ═══════════════════════════════════════════════════════════════════════

func registerNormalizeArraySpanParams*(reg: var PassRegistry, decayArrays: static bool) =
  ## Registers array/span param normalization.
  ##
  ## - decayArrays true: the C model, element pointers T*
  ## - decayArrays false: arrays stay first-class, array params emit as arrays
  reg.register("normalizeArraySpanParams", pkTransform, phaseEarly,
    "var array/openArray params become element pointers: fold HiddenDeref on " &
      "access, bare arrays (or toOpenArray) at call sites, gtPtr(gtArray)->gtPtr(T) per decayArrays",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      # callee param kinds from the ORIGINAL types, before this pass rewrites them
      var calleeKinds = initTable[string, seq[ArraySpanParamKind]]()
      for fnKey in ctx.allFnTab.keys:
        let fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          calleeKinds[fn.pName.symbol.name] = collectArraySpanParamKinds(fn)
      for fnKey in ctx.genericInsts.keys:
        let fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          calleeKinds[fn.pName.symbol.name] = collectArraySpanParamKinds(fn)
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          normalizeArraySpanBody(fn.pBody, calleeKinds, decayArrays)
          normalizeArraySpanFn(fn, decayArrays)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          normalizeArraySpanBody(fn.pBody, calleeKinds, decayArrays)
          normalizeArraySpanFn(fn, decayArrays)

  )
