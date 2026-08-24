## Phase 5: Backend Pass Extraction
##
## Passes that eliminate "smart" logic from backends, making codegen
## purely about string emission. Each pass pre-computes analysis or
## decomposes complex IR patterns into simpler ones so backends
## only need to render straightforward IR to target-language strings.

import std / [sequtils, tables, sets, strutils, strformat, options]
import ../ir/gpu_types
import ./pass_datatypes
import ./passes_legalizations

proc isGlobalFn*(fn: GpuAst): bool =
  doAssert fn.kind == gpuProc
  result = attGlobal in fn.pAttributes

# ═══════════════════════════════════════════════════════════════════════════
# Pass 1: rewriteIndexDeref
# ═══════════════════════════════════════════════════════════════════════════

proc rewriteIndexDerefImpl*(ctx: var GpuContext; n: var GpuAst) =
  ## Transform `gpuIndex(gpuDeref(ident))` into `gpuIndex(iArr=dOf(ident))`
  ## when the deref'd ident is a pointer (not array) type.
  ##
  ## This is a pure IR normalization: `arr[idx]` where `arr` is a pointer
  ## is syntactically `(*arr)[idx]` which is equivalent to `arr[idx]` in C.
  ## The pass rewrites the IR to the simpler form.
  case n.kind
  of gpuIndex:
    if n.iArr.kind == gpuDeref:
      # Check if the deref'd node is a pointer to a statically-sized array.
      # If it's a plain ptr (not ptr-to-array), rewrite gpuIndex(gpuDeref(x)) -> gpuIndex(x).
      let isPtrToArray = (block:
        let derefOf = n.iArr.dOf
        if derefOf.kind == gpuIdent and derefOf.symbol != nil:
          let symTyp = derefOf.symbol.typ
          symTyp != nil and symTyp.kind == gtPtr and
          symTyp.to != nil and symTyp.to.kind == gtArray
        else:
          false)
      if not isPtrToArray:
        n = GpuAst(kind: gpuIndex, iArr: n.iArr.dOf, iIndex: n.iIndex)
      else:
        for ch in mitems(n):
          rewriteIndexDerefImpl(ctx, ch)
  else:
    for ch in mitems(n):
      rewriteIndexDerefImpl(ctx, ch)


# ═══════════════════════════════════════════════════════════════════════════
# Pass 2: decomposeMemcpyVars
# ═══════════════════════════════════════════════════════════════════════════

proc genMemcpyCall(lhs, rhs: GpuAst; sizeExpr: GpuAst): GpuAst =
  ## Create a gpuCall node for `__builtin_memcpy(dst, src, size)`.
  let memcpySym = newSymbol("__builtin_memcpy", iSym = "__builtin_memcpy", symKind = gsProc)
  let memcpyIdent = GpuAst(kind: gpuIdent, symbol: memcpySym)
  var call = GpuAst(kind: gpuCall)
  call.cName = memcpyIdent
  call.cArgs = @[lhs, rhs, sizeExpr]
  result = call

proc genMemcpySizeOf*(ctx: var GpuContext; expr: GpuAst): GpuAst =
  ## Create a `sizeof(T)` gpuCall equivalent for the type of the expression.
  let sizeSym = newSymbol("__builtin_sizeof", iSym = "__builtin_sizeof", symKind = gsProc)
  let sizeIdent = GpuAst(kind: gpuIdent, symbol: sizeSym)
  var call = GpuAst(kind: gpuCall)
  call.cName = sizeIdent
  call.cArgs = @[expr]
  result = call

proc decomposeMemcpyVarsImpl*(ctx: var GpuContext; n: var GpuAst) =
  ## Walk gpuVar and gpuAssign nodes. For those requiring memcpy:
  ## 1. For gpuVar: split into declaration (no init) + memcpy call
  ## 2. For gpuAssign: replace with memcpy call
  ##
  ## After this pass, no backend needs to handle `vRequiresMemcpy` or
  ## `aRequiresMemcpy` — they only emit `=` for simple assignments.
  case n.kind
  of gpuVar:
    if n.vRequiresMemcpy and n.vInit.kind != gpuDiscard:
      # Split into:
      #   1. The same gpuVar but with gpuDiscard init (no RequiresMemcpy)
      #   2. memcpy(&vName, &vInit, sizeof(vType))
      let addrOfVar = GpuAst(kind: gpuAddr, aOf: n.vName)
      let addrOfInit = GpuAst(kind: gpuAddr, aOf: n.vInit)
      let sizeCall = genMemcpySizeOf(ctx, n.vName)
      let memcpyNode = genMemcpyCall(addrOfVar, addrOfInit, sizeCall)

      # Replace n with a gpuBlock containing var decl + memcpy call
      n.vRequiresMemcpy = false
      n.vInit = GpuAst(kind: gpuDiscard)
      n = GpuAst(kind: gpuBlock, blockLabel: "_memcpy", statements: @[n, memcpyNode])
    else:
      for ch in mitems(n):
        decomposeMemcpyVarsImpl(ctx, ch)

  of gpuAssign:
    if n.aRequiresMemcpy:
      # Replace with memcpy(&aLeft, &aRight, sizeof(aLeft))
      let addrOfLeft = GpuAst(kind: gpuAddr, aOf: n.aLeft)
      let addrOfRight = GpuAst(kind: gpuAddr, aOf: n.aRight)
      let sizeCall = genMemcpySizeOf(ctx, n.aLeft)
      n = genMemcpyCall(addrOfLeft, addrOfRight, sizeCall)
    else:
      for ch in mitems(n):
        decomposeMemcpyVarsImpl(ctx, ch)

  of gpuIf:
    decomposeMemcpyVarsImpl(ctx, n.ifCond)
    decomposeMemcpyVarsImpl(ctx, n.ifThen)
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
        # Inline anonymous blocks only (created by memcpy decomposition).
        # Labeled blocks (scope blocks from blitting) must be preserved
        # to maintain C++ scope isolation and prevent duplicate var declarations.
        for s in stmt.statements:
          newStmts.add s
      else:
        newStmts.add stmt
    n.statements = newStmts
    # Rename duplicate {.inject.} var declarations across sibling blocks.
    # Template expansions can produce the same var names in anonymous blocks
    # that will be flattened by codegen. Rename to avoid collisions.
    dedupVarNames(n.statements)

  of gpuProc:
    decomposeMemcpyVarsImpl(ctx, n.pBody)

  else:
    for ch in mitems(n):
      decomposeMemcpyVarsImpl(ctx, ch)


# ═══════════════════════════════════════════════════════════════════════════
# Pass 3: annotateTypesForCodegen
# ═══════════════════════════════════════════════════════════════════════════

type
  TypeDescKind* = enum
    tdkVoid
    tdkBool
    tdkInt8      # signed 8-bit
    tdkUint8     # unsigned 8-bit
    tdkInt16
    tdkUint16
    tdkInt32
    tdkUint32
    tdkInt64
    tdkUint64
    tdkFloat32
    tdkFloat64
    tdkFloat16
    tdkBf16
    tdkSize
    tdkPtr       # pointer to another type descriptor
    tdkArray     # fixed-size array
    tdkStruct    # named struct with fields
    tdkString    # const char*
    tdkVoidPtr   # void*
    tdkGeneric   # generic instantiation name
    tdkUnresolved

  TypeDesc* = ref object
    case kind*: TypeDescKind
    of tdkVoid, tdkBool, tdkInt8, tdkUint8, tdkInt16, tdkUint16,
       tdkInt32, tdkUint32, tdkInt64, tdkUint64, tdkFloat32, tdkFloat64,
       tdkFloat16, tdkBf16, tdkSize, tdkString, tdkVoidPtr, tdkUnresolved:
      discard
    of tdkPtr:
      tdTo*: TypeDesc         # pointed-to type
      tdImplicit*: bool       # was a var T (implicit pointer)
      tdMutable*: bool        # read_write vs read
    of tdkArray:
      tdElem*: TypeDesc       # element type
      tdLen*: int             # length (0 = runtime-sized)
    of tdkStruct:
      tdStructName*: string         # struct name
      tdFields*: seq[tuple[name: string, typ: TypeDesc]]
    of tdkGeneric:
      tdGenericName*: string         # base name
      tdArgs*: seq[TypeDesc]  # generic args
proc gpuTypeToDesc*(t: GpuType): TypeDesc =
  ## Convert a GpuType to a backend-neutral TypeDescriptor.
  if t.isNil:
    return TypeDesc(kind: tdkUnresolved)

  result = case t.kind
  of gtVoid:     TypeDesc(kind: tdkVoid)
  of gtBool:     TypeDesc(kind: tdkBool)
  of gtUint8:    TypeDesc(kind: tdkUint8)
  of gtUint16:   TypeDesc(kind: tdkUint16)
  of gtInt16:    TypeDesc(kind: tdkInt16)
  of gtUint32:   TypeDesc(kind: tdkUint32)
  of gtInt32:    TypeDesc(kind: tdkInt32)
  of gtUint64:   TypeDesc(kind: tdkUint64)
  of gtInt64:    TypeDesc(kind: tdkInt64)
  of gtFloat32:  TypeDesc(kind: tdkFloat32)
  of gtFloat64:  TypeDesc(kind: tdkFloat64)
  of gtFloat16:  TypeDesc(kind: tdkFloat16)
  of gtBf16:     TypeDesc(kind: tdkBf16)
  of gtSize_t:   TypeDesc(kind: tdkSize)
  of gtString:   TypeDesc(kind: tdkString)
  of gtVoidPtr:  TypeDesc(kind: tdkVoidPtr)
  of gtPtr:
    var tdTo = gpuTypeToDesc(t.to)
    if t.to.kind == gtUA:
      tdTo = gpuTypeToDesc(t.to.uaTo)
    TypeDesc(kind: tdkPtr, tdTo: tdTo,
             tdImplicit: t.implicit, tdMutable: t.mutable)
  of gtArray:
    TypeDesc(kind: tdkArray, tdElem: gpuTypeToDesc(t.aTyp), tdLen: t.aLen)
  of gtObject:
    var fields: seq[tuple[name: string, typ: TypeDesc]]
    for f in t.oFields:
      fields.add (f.name, gpuTypeToDesc(f.typ))
    TypeDesc(kind: tdkStruct, tdStructName: t.name, tdFields: fields)
  of gtUA:
    TypeDesc(kind: tdkArray, tdElem: gpuTypeToDesc(t.uaTo), tdLen: 0)
  of gtGenericInst:
    var args: seq[TypeDesc]
    for a in t.gArgs:
      args.add gpuTypeToDesc(a)
    TypeDesc(kind: tdkGeneric, tdGenericName: t.gName, tdArgs: args)
  of gtStatic:
    TypeDesc(kind: tdkInt32)
  of gtSpan:
    # Span is lowered to ptr+len before this pass
    TypeDesc(kind: tdkUnresolved)
  of gtInvalid:
    TypeDesc(kind: tdkUnresolved)

proc getTypeDesc*(ctx: var GpuContext; arg: GpuAst): TypeDesc =
  ## Pre-compute the type descriptor for a given IR node.
  ## This replaces the recursive getType/getFieldType calls during codegen.
  case arg.kind
  of gpuIdent:
    if arg.symbol != nil:
      result = gpuTypeToDesc(arg.symbol.typ)
    else:
      result = TypeDesc(kind: tdkUnresolved)
  of gpuAddr:
    let inner = ctx.getTypeDesc(arg.aOf)
    result = TypeDesc(kind: tdkPtr, tdTo: inner, tdImplicit: false, tdMutable: false)
  of gpuDeref:
    let argTyp = ctx.getTypeDesc(arg.dOf)
    if argTyp.kind == tdkPtr:
      result = argTyp.tdTo
    else:
      result = TypeDesc(kind: tdkUnresolved)
  of gpuCall:
    let fn = arg.cName
    let key = fn.symbol.iSym
    if key in ctx.fnTable:
      let entry = ctx.fnTable[key]
      if not entry.body.isNil and entry.body.kind == gpuProc:
        result = gpuTypeToDesc(entry.body.pRetType)
      else:
        result = TypeDesc(kind: tdkVoid)
    elif fn in ctx.genericInsts:
      result = gpuTypeToDesc(ctx.genericInsts[fn].pRetType)
    elif fn in ctx.allFnTab:
      result = gpuTypeToDesc(ctx.allFnTab[fn].pRetType)
    elif fn in ctx.builtinFns:
      result = gpuTypeToDesc(ctx.builtinFns[fn].pRetType)
    else:
      result = TypeDesc(kind: tdkUnresolved)
  of gpuIndex:
    let arrType = ctx.getTypeDesc(arg.iArr)
    case arrType.kind
    of tdkPtr:   result = arrType.tdTo
    of tdkArray: result = arrType.tdElem
    else:        result = TypeDesc(kind: tdkUnresolved)
  of gpuDot:
    let parentTyp = ctx.getTypeDesc(arg.dParent)
    if parentTyp.kind == tdkStruct:
      let fieldName = arg.dField.ident()
      for (fn, ft) in parentTyp.tdFields:
        if fn == fieldName:
          return ft
    result = TypeDesc(kind: tdkUnresolved)
  of gpuLit:
    result = gpuTypeToDesc(arg.lType)
  of gpuBlock:
    if arg.isExpr and arg.statements.len > 0:
      result = ctx.getTypeDesc(arg.statements[^1])
    else:
      result = TypeDesc(kind: tdkUnresolved)
  of gpuPrefix:
    result = ctx.getTypeDesc(arg.pVal)
  of gpuConv:
    result = gpuTypeToDesc(arg.convTo)
  of gpuCast:
    result = gpuTypeToDesc(arg.cTo)
  else:
    result = TypeDesc(kind: tdkUnresolved)


# ═══════════════════════════════════════════════════════════════════════════
# Pass 4: emitFunctionSignatures
# ═══════════════════════════════════════════════════════════════════════════

proc genFunctionSig(procNode: GpuAst): string =
  ## Pre-compute the function signature string for a gpuProc node.
  let retType = procNode.pRetType
  let fnName = procNode.pName.ident()

  var params: seq[string]
  for p in procNode.pParams:
    params.add(p.ident.ident())
  let fnArgs = params.join(", ")

  if retType.kind == gtPtr and retType.to.kind == gtArray:
    let arrayTyp = retType.to.aTyp
    let innerLen = $retType.to.aLen
    result = &"__fnSig({fnName})({fnArgs})->[{innerLen}]"
  else:
    result = &"__fnSig({fnName})({fnArgs})"

proc emitFunctionSignaturesImpl*(ctx: var GpuContext) =
  ## Walk all gpuProc nodes in fnTable and pre-compute function signature
  ## metadata. The signature is stored and codegen reads it directly
  ## instead of calling genFunctionType.
  for key, entry in ctx.fnTable.mpairs:
    if not entry.body.isNil and entry.body.kind == gpuProc:
      let sig = genFunctionSig(entry.body)
      entry.sigString = sig
      # Store as a comment node in the proc body preamble
      let sigComment = GpuAst(kind: gpuComment, comment: "sig:" & sig)
      if not entry.body.pBody.isNil:
        let sigComment = GpuAst(kind: gpuComment, comment: "sig:" & sig)
        entry.body.pBody.statements.insert(sigComment, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Pass 5: mangleNames (with base58)
# ═══════════════════════════════════════════════════════════════════════════

proc mangleNamesImpl*(ctx: var GpuContext) =
  ## Walk all functions in fnTable and apply NamePolicy per function:
  ## - fkGenericInst → npHashSuffix (append 7-char base58 of hash)
  ## - fkDefined + non-overloaded → npClean (no suffix — C++ mangling handles it)
  ## - Operator functions → npPatch (+ → add, etc.)
  ## For variables: apply npHashSuffix when dedup would cause collision.
  ##
  ## Updates symbol.name accordingly.
  var seenNames = initHashSet[string]()

  for key, entry in ctx.fnTable.mpairs:
    let ident = entry.ident
    if ident.isNil or ident.symbol.isNil:
      continue

    let name = ident.symbol.name
    let iSym = ident.symbol.iSym

    var effectivePolicy = entry.namePolicy
    if effectivePolicy == npUnassigned:
      # Determine policy if not already assigned
      if fkGenericInst in entry.kind:
        effectivePolicy = npHashSuffix
      elif fkDefined in entry.kind:
        if name in ["+", "-", "*", "/", ".."]:
          effectivePolicy = npPatch
        elif name in seenNames or name.len == 0:
          effectivePolicy = npHashSuffix
        else:
          effectivePolicy = npClean
      else:
        # builtin/external: keep as-is
        effectivePolicy = npClean
      entry.namePolicy = effectivePolicy

    case effectivePolicy

    of npClean:
      discard

    of npHashSuffix:
      # Use the iSym as the hash source (stable across compilation units)
      var hashVal: int64 = 0
      for i, c in iSym:
        hashVal = hashVal xor (int64(ord(c)) shl ((i and 7) * 8))
      if hashVal == 0:
        hashVal = 1
      let suffix = shortHash(hashVal)
      ident.symbol.name = name & "_" & suffix

    of npPatch:
      # Operator rename
      let newName = case name
        of "+": "add"
        of "-": "sub"
        of "*": "mul"
        of "/": "div"
        of "..": "range"
        else: name
      ident.symbol.name = newName
      ident.symbol.iSym = ident.symbol.iSym.replace(name, newName)
    else:
      discard

    seenNames.incl ident.symbol.name


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: WGSL-specific passes
# ═══════════════════════════════════════════════════════════════════════════

# Forward declarations for Phase 6 helper functions
proc getStructType*(n: GpuAst): GpuType
proc determineIdent*(arg: GpuAst): GpuAst
proc determineMutability*(arg: GpuAst): bool
proc resolveIdentAddressSpace*(ctx: GpuContext, n: GpuAst): AddressSpace
proc shortAddrSpace*(addrSpace: AddressSpace): string
proc patchTypeImpl*(t: GpuType): GpuType
proc patchSymbolImpl*(n: GpuAst): GpuAst
proc genGenericName*(ctx: GpuContext; n: GpuAst; params: seq[GpuParam]; callerParams: Table[string, GpuParam]): string
proc makeFnGeneric*(ctx: var GpuContext; fn: GpuAst; gi: GenericInst): GpuAst
proc getGenericArguments*(ctx: GpuContext; args: seq[GpuAst]; params: seq[GpuParam]; callerParams: Table[string, GpuParam]): seq[GenericArg]

proc injectAddressOfImpl*(ctx: var GpuContext; n: var GpuAst) =
  ## Replaces storage-buffer idents with `gpuAddr(ident)` in global fns.
  ## Also handles bool patching: bool globals are wrapped in gpuConv(expr, gtBool).
  case n.kind
  of gpuIdent:
    if n.symbol.iSym in ctx.globals and (let p = ctx.globals[n.symbol.iSym]; p.typ.kind == gtPtr):
      n = GpuAst(kind: gpuAddr, aOf: n)
    elif n.symbol.iSym in ctx.globals and (let p = ctx.globals[n.symbol.iSym]; p.typ.kind == gtBool):
      n = GpuAst(kind: gpuConv, convTo: GpuType(kind: gtBool), convExpr: n)
  of gpuDeref:
    if n.dOf.kind == gpuIdent and n.dOf.symbol.iSym in ctx.globals and
       (let p = ctx.globals[n.dOf.symbol.iSym]; p.typ.kind == gtPtr):
      n = n.dOf
  else:
    for ch in n.mitems:
      ctx.injectAddressOfImpl(ch)

proc pullConstantPragmaVarsImpl*(ctx: var GpuContext; blk: var GpuAst) =
  ## Filters out `var foo {.const_mem.}: dtype` from global blocks and adds to globals.
  doAssert blk.kind == gpuBlock
  var i = 0
  while i < blk.len:
    let g = blk.statements[i]
    if g.kind == gpuVar and g.addressSpace == asConstant:
      doAssert g.vInit.kind == gpuDiscard, "{.const_mem.} var must not have init!"
      let param = GpuParam(ident: g.vName, typ: g.vType, addressSpace: asDevice)
      ctx.globals[param.ident.symbol.iSym] = param
      blk.statements.delete(i)
    else:
      inc i

proc removeStructPointerFieldsImpl*(blk: var GpuAst) =
  ## Removes ptr fields from struct definitions (WGSL limitation).
  doAssert blk.kind == gpuBlock
  for typ in blk.mitems:
    if typ.kind == gpuAlias: continue
    doAssert typ.kind == gpuTypeDef
    var i = 0
    while i < typ.tFields.len:
      let f = typ.tFields[i]
      if f.typ.kind == gtPtr:
        typ.tFields.delete(i)
      else:
        inc i

proc rewriteCompoundAssignmentImpl*(n: GpuAst): GpuAst =
  ## Rewrites `x += y` → `x = x + y`.
  ##
  ## The LHS of a compound assignment can arrive from the frontend wrapped in a
  ## `gpuAddr` (Nim's `HiddenAddr` read-modify-write sugar for non-simple
  ## lvalues, e.g. macro-expanded statement-list expressions like ceramic's
  ## `tv[m,n]`). After the rewrite the lvalue must be used directly on both
  ## sides: an `&`-wrapped LHS is not a modifiable lvalue on any backend and
  ## `(&x) + y` is not a valid read. The RHS operand is a CLONE of the LHS
  ## because legalization mutates the assignment LHS in place (hoisting
  ## block-expression intermediates) — the two sides must not alias the same
  ## node.
  doAssert n.kind == gpuBinOp
  if n.bOp.ident() in ["<=", "==", ">=", "!="]: return n
  template genAssign(left, rnode, op: typed): untyped =
    let right = GpuAst(kind: gpuBinOp, bType: n.bType, bOp: op,
                       bLeft: left.clone(), bRight: rnode)
    GpuAst(kind: gpuAssign, aLeft: left, aRight: right, aRequiresMemcpy: false)
  let op = n.bOp.ident()
  if op.len >= 2 and op[^1] == '=':
    var opAst = GpuAst(kind: gpuIdent, symbol: newSymbol(op[0 .. ^2]))
    opAst.symbol.iSym = opAst.symbol.name
    var lhs = n.bLeft
    if lhs.kind == gpuAddr:
      lhs = lhs.aOf
    result = genAssign(lhs, n.bRight, opAst)
  else:
    result = n

proc makeCodeValidImpl*(ctx: var GpuContext; n: var GpuAst; inGlobal: bool) =
  ## Addresses AST patterns needing rewrite for WGSL.
  ## Handles compound assignment rewrites, struct pointer field replacement,
  ## object constructor stripping, call signature updates, and var type updates.
  ##
  ## This is a combined pass that was previously inline in wgsl_lang.nim's makeCodeValid.
  ## For Phase 6, the compound assignment rewrite is extracted separately.
  ## The remaining logic (struct pointer fields, call sigs, var updates) stays
  ## here as makeCodeValid for now until further decomposition.
  case n.kind
  of gpuBinOp:
    n = rewriteCompoundAssignmentImpl(n)
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
  of gpuObjConstr:
    let t = n.ocType
    var i = 0
    while i < n.ocFields.len:
      let f = n.ocFields[i]
      if (t, f.name) in ctx.structsWithPtrs:
        if f.typ.kind == gtPtr:
          n.ocFields.delete(i)
        else:
          inc i
      else:
        inc i
  of gpuDot:
    var p = n.dParent
    if p.kind notin [gpuIdent, gpuDeref]:
      for ch in n.mitems:
        ctx.makeCodeValidImpl(ch, inGlobal)
    else:
      let id = getStructType(p)
      doAssert n.dField.kind == gpuIdent
      let field = n.dField.ident()
      if id.kind != gtVoid and (id, field) in ctx.structsWithPtrs:
        let v = ctx.structsWithPtrs[(id, field)]
        if inGlobal:
          n = GpuAst(kind: gpuAddr, aOf: v)
        else:
          n = v
  of gpuAssign:
    if n.aLeft.kind == gpuDot and n.aLeft.dParent.kind in [gpuIdent, gpuDeref]:
      let dot = n.aLeft
      let id = getStructType(dot.dParent)
      if id.kind != gtVoid:
        doAssert dot.dField.kind == gpuIdent
        let field = dot.dField.ident()
        if (id, field) in ctx.structsWithPtrs:
          raiseAssert "Assignment of a struct pointer field is not supported"
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
  of gpuCall:
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
    let fnName = n.cName
    if fnName in ctx.fnTab:
      let fn = ctx.fnTab[fnName]
      let params = fn.pParams
      for i, arg in n:
        let argId = arg.determineIdent()
        if argId.kind != gpuDiscard and argId.ident().len > 0:
          var p = params[i]
          let argSpace = ctx.resolveIdentAddressSpace(argId)
          if p.addressSpace != argSpace:
            p.addressSpace = argSpace
            ctx.varAddressSpaces[p.ident.symbol.iSym] = argSpace
            fn.pParams[i] = p
  of gpuVar:
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
    if n.vType.kind == gtPtr:
      let rightId = n.vInit.determineIdent()
      let space = ctx.resolveIdentAddressSpace(rightId)
      ctx.varAddressSpaces[n.vName.symbol.iSym] = space
      # The RHS ident's own type may be a non-pointer (e.g. `addr arr[0]`
      # over an array): only a pointer RHS carries a mutability flag.
      let rightTyp = if rightId.symbol != nil: rightId.symbol.typ else: nil
      let rhsIsPtr = not rightTyp.isNil and rightTyp.kind == gtPtr
      n.vType.mutable = rhsIsPtr and rightTyp.mutable
      n.vName.symbol.typ.mutable = rhsIsPtr and rightTyp.mutable
  else:
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)

proc checkCodeValidImpl*(ctx: var GpuContext; n: GpuAst) =
  ## Checks WGSL validity constraints.
  case n.kind
  of gpuVar:
    if n.vType.kind == gtPtr and n.vMutable:
      let code = "<WGSL>"
      raiseAssert "var to pointer type invalid in WGSL"
  else:
    for ch in n:
      ctx.checkCodeValidImpl(ch)

proc updateSymsInGlobalsImpl*(ctx: var GpuContext; n: GpuAst) =
  ## Update symbols in global functions to reflect symbol kind/mutability.
  case n.kind
  of gpuIdent:
    if n.symbol.iSym in ctx.globals:
      n.symbol.symKind = gsGlobalKernelParam
      if n.symbol.typ.kind == gtPtr:
        let g = ctx.globals[n.symbol.iSym]
        n.symbol.typ.mutable = g.typ.kind == gtPtr
  else:
    for ch in n:
      ctx.updateSymsInGlobalsImpl(ch)

proc scanGenericsImpl*(ctx: var GpuContext; n: GpuAst; callerParams: Table[string, GpuParam]) =
  ## Scans for gpuCall nodes, generates generic instantiations for pointer args.
  case n.kind
  of gpuCall:
    let fn = n.cName
    if fn in ctx.allFnTab:
      let params = ctx.allFnTab[fn].pParams
      let gi = GenericInst(name: genGenericName(ctx, n, params, callerParams),
                           args: getGenericArguments(ctx, n.cArgs, params, callerParams))
      let anyPointers = params.anyIt(it.typ.kind == gtPtr)
      if anyPointers:
        n.cName = GpuAst(kind: gpuIdent,
                         symbol: newSymbol(gi.name, symKind = gsProc, iSym = gi.name))
        let gName = n.cName
        if gName notin ctx.fnTab:
          let fnCalled = ctx.allFnTab[fn].clone()
          let fnGen = makeFnGeneric(ctx, fnCalled, gi)
          ctx.fnTab[gName] = fnGen
          ctx.allFnTab[gName] = fnGen
          var callParams = initTable[string, GpuParam]()
          for p in fnGen.pParams:
            callParams[p.ident.symbol.iSym] = p
          ctx.scanGenericsImpl(fnGen, callParams)
      elif fn notin ctx.fnTab:
        let fnCalled = ctx.allFnTab[fn]
        ctx.fnTab[fn] = fnCalled
        for ch in fnCalled:
          ctx.scanGenericsImpl(ch, callerParams)
      for arg in n.cArgs:
        ctx.scanGenericsImpl(arg, callerParams)
    else:
      for ch in n:
        ctx.scanGenericsImpl(ch, callerParams)
  of gpuObjConstr:
    for f in n.ocFields:
      if f.typ.kind == gtPtr:
        doAssert f.value.kind in [gpuAddr, gpuIdent]
        let id = f.value.determineIdent()
        doAssert id.symbol.symKind == gsGlobalKernelParam
        ctx.structsWithPtrs[(n.ocType, f.name)] = id
  else:
    for ch in n:
      ctx.scanGenericsImpl(ch, callerParams)

proc getStructType*(n: GpuAst): GpuType =
  ## Given an ident, return the struct type or void if not a struct.
  doAssert n.kind in [gpuIdent, gpuDeref]
  var p = n
  if p.kind == gpuDeref:
    p = n.dOf
  result = if p.symbol.typ.kind == gtPtr and p.symbol.typ.to.kind == gtObject:
             p.symbol.typ.to
           elif p.symbol.typ.kind == gtObject:
             p.symbol.typ
           else: GpuType(kind: gtVoid)

proc genGenericName*(ctx: GpuContext; n: GpuAst; params: seq[GpuParam]; callerParams: Table[string, GpuParam]): string =
  ## Generates unique name for a generic WGSL function instantiation.
  doAssert n.kind == gpuCall
  result = n.cName.ident() & '_'
  for i, arg in n.cArgs:
    let p = params[i]
    var s: string
    if p.typ.kind != gtPtr:
      s = "l"
    else:
      let argIdent = arg.determineIdent()
      var lArg: GpuAst = arg
      if argIdent.kind != gpuDiscard and argIdent.symbol.iSym in callerParams:
        lArg = callerParams[argIdent.symbol.iSym].ident
      let addrSpace = ctx.resolveIdentAddressSpace(lArg)
      let mutable = lArg.determineMutability()
      let m = if mutable: "mut" else: ""
      s = shortAddrSpace(addrSpace) & m
    result.add s
    if i < n.cArgs.high:
      result.add '_'

proc makeFnGeneric*(ctx: var GpuContext; fn: GpuAst; gi: GenericInst): GpuAst =
  ## Returns a cloned function with params updated for the generic instantiation.
  result = fn
  let pnSym = newSymbol(gi.name, symKind = gsProc)
  result.pName = GpuAst(kind: gpuIdent, symbol: pnSym)
  for i, p in result.pParams.mpairs:
    let arg = gi.args[i]
    p.addressSpace = arg.addrSpace
    ctx.varAddressSpaces[p.ident.symbol.iSym] = arg.addrSpace
    if p.ident.symbol.typ.kind == gtPtr:
      p.ident.symbol.typ.mutable = arg.mutable
    p.ident = patchSymbolImpl(p.ident)
    p.typ = p.ident.symbol.typ
  proc getIf(params: seq[GpuParam]; n: GpuAst): Option[GpuParam] =
    doAssert n.kind == gpuIdent
    for p in params:
      if p.ident.symbol.iSym == n.symbol.iSym: return some(p)
  proc updateSyms(n: var GpuAst; params: seq[GpuParam]) =
    case n.kind
    of gpuIdent:
      let pOpt = params.getIf(n)
      if pOpt.isSome:
        let p = pOpt.get
        n.symbol.symKind = p.ident.symbol.symKind
        n.symbol.typ = p.typ
    else:
      for ch in n.mitems:
        updateSyms(ch, params)
  updateSyms(result.pBody, result.pParams)

proc patchSymbolImpl*(n: GpuAst): GpuAst =
  doAssert n.kind == gpuIdent
  result = n
  if n.symbol != nil and n.symbol.symKind == gsGlobalKernelParam:
    result.symbol.typ = patchTypeImpl(result.symbol.typ)

proc patchTypeImpl*(t: GpuType): GpuType =
  result = t
  if result.kind == gtBool:
    result.kind = gtInt32
  elif result.kind == gtPtr and result.to.kind == gtBool:
    result.to.kind = gtInt32

proc shortAddrSpace*(addrSpace: AddressSpace): string =
  case addrSpace
  of asDevice: "s"
  of asConstant: "u"
  of asSMEM: "w"
  of asRMEM: "l"

proc determineMutability*(arg: GpuAst): bool =
  case arg.kind
  of gpuIdent: (not arg.symbol.isNil) and arg.symbol.typ != nil and arg.symbol.typ.kind == gtPtr
  of gpuAddr: true
  of gpuDeref: true
  of gpuCall: false
  of gpuIndex: arg.iArr.determineMutability()
  of gpuDot: arg.dParent.determineMutability()
  of gpuLit: false
  of gpuBinOp: false
  of gpuBlock: arg.statements[^1].determineMutability()
  of gpuPrefix: false
  of gpuConv: false
  of gpuCast: arg.cExpr.determineMutability()
  else:
    raiseAssert "Cannot determine mutability from: " & $arg

proc determineIdent*(arg: GpuAst): GpuAst =
  template dfl(): untyped = GpuAst(kind: gpuDiscard)
  case arg.kind
  of gpuIdent: arg
  of gpuAddr: arg.aOf.determineIdent()
  of gpuDeref: arg.dOf.determineIdent()
  of gpuCall: dfl()
  of gpuIndex: arg.iArr.determineIdent()
  of gpuDot: arg.dParent.determineIdent()
  of gpuLit: dfl()
  of gpuBinOp: dfl()
  of gpuBlock: arg.statements[^1].determineIdent()
  of gpuPrefix: dfl()
  of gpuConv: dfl()
  of gpuCast: arg.cExpr.determineIdent()
  of gpuObjConstr: dfl()
  of gpuMaterialize: arg.mExpr.determineIdent()
  else:
    raiseAssert "Cannot determine ident from: " & $arg

# ─── Value address-space resolution ────────────────────────────────────────────
# MSL requires an explicit address space on pointer struct fields and casts.
# The IR carries none on `gtPtr`, so the printers resolve it from the value's
# dataflow with an asRMEM fallback. The collection pass below precomputes the
# var spaces (`varAddressSpaces`) for every backend; the pointer-field variant
# table (`ptrFieldVariants`) is Metal-only.

proc exprType*(ctx: GpuContext, n: GpuAst): GpuType =
  ## Best-effort type of an expression node (nil when unknown). The printers
  ## use it to detect array-typed operands of `addr`, which need pointer
  ## decay rather than `&`.
  case n.kind
  of gpuIdent: n.symbol.typ
  of gpuLit: n.lType
  of gpuBinOp: n.bType
  of gpuPrefix: ctx.exprType(n.pVal)
  of gpuCall: ctx.getFnReturnType(n.cName)
  of gpuAddr: ctx.exprType(n.aOf)
  of gpuDeref: ctx.exprType(n.dOf)
  of gpuIndex:
    let arrT = ctx.exprType(n.iArr)
    if arrT.isNil:
      nil
    elif arrT.kind == gtArray: arrT.aTyp
    elif arrT.kind == gtPtr: arrT.to
    elif arrT.kind == gtUA: arrT.uaTo
    else: nil
  of gpuObjConstr: n.ocType
  of gpuConv: n.convTo
  of gpuCast: n.cTo
  of gpuMaterialize: ctx.exprType(n.mExpr)
  of gpuDot:
    # The field's type on the parent's struct type (ptr/UA layers
    # stripped): generic field-type inference for field-access chains.
    # The value address-space resolution uses it for nested
    # pointer-field chains.
    block:
      var pT = ctx.exprType(n.dParent)
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

proc dotParentType(ctx: GpuContext, n: GpuAst): GpuType =
  ## Struct type of a field-access base, with the pointer/UA layer stripped.
  var t = ctx.exprType(n)
  if t != nil and t.kind == gtPtr:
    t = t.to
  if t != nil and t.kind == gtUA:
    t = t.uaTo
  result = t

proc ptrFieldNames*(t: GpuType): seq[string] =
  ## Pointer field names of a struct type, in declaration order.
  case t.kind
  of gtObject:
    for f in t.oFields:
      if f.typ.kind == gtPtr:
        result.add f.name
  of gtGenericInst:
    for f in t.gFields:
      if f.typ.kind == gtPtr:
        result.add f.name
  else: discard

proc resolveValueAddressSpace*(ctx: GpuContext, n: GpuAst): AddressSpace =
  ## Address space of a pointer-typed value, resolved from the value's
  ## dataflow with an `asRMEM` fallback.
  if n == nil:
    return asRMEM
  case n.kind
  of gpuIdent:
    if n.symbol == nil:
      asRMEM
    elif n.symbol.iSym in ctx.varAddressSpaces:
      ctx.varAddressSpaces[n.symbol.iSym]
    elif n.symbol.symKind in {gsDeviceKernelParam, gsGlobalKernelParam}:
      # Explicit `ptr T` params are device pointers. Implicit `var T`
      # params are thread.
      if n.symbol.typ != nil and n.symbol.typ.kind == gtPtr and
         not n.symbol.typ.implicit:
        asDevice
      else:
        asRMEM
    else:
      asRMEM
  of gpuAddr: ctx.resolveValueAddressSpace(n.aOf)
  of gpuCast: ctx.resolveValueAddressSpace(n.cExpr)
  of gpuConv: ctx.resolveValueAddressSpace(n.convExpr)
  of gpuBinOp: ctx.resolveValueAddressSpace(n.bLeft)
  of gpuDot:
    let ptype = ctx.dotParentType(n.dParent)
    if ptype != nil and ptype in ctx.ptrFieldVariants:
      let base = ctx.ptrFieldVariants[ptype][0]
      let fieldIdx = ptrFieldNames(ptype).find(n.dField.ident())
      if fieldIdx >= 0 and fieldIdx < base.len:
        return base[fieldIdx] # base variant space of the field
    ctx.resolveValueAddressSpace(n.dParent)
  of gpuIndex: ctx.resolveValueAddressSpace(n.iArr)
  of gpuDeref: ctx.resolveValueAddressSpace(n.dOf)
  of gpuPrefix: ctx.resolveValueAddressSpace(n.pVal)
  of gpuMaterialize: ctx.resolveValueAddressSpace(n.mExpr)
  else: asRMEM

proc resolveIdentAddressSpace*(ctx: GpuContext, n: GpuAst): AddressSpace =
  ## Authoritative address space of a symbol reference: the recorded var or
  ## param space, else `asRMEM`. Non-ident expressions (`addr`, `deref`,
  ## `index`, `dot`, `cast`, `block`, `materialize`) resolve through their
  ## base ident; other expressions are `asRMEM`.
  var base = n
  while base != nil and base.kind != gpuIdent:
    case base.kind
    of gpuAddr: base = base.aOf
    of gpuDeref: base = base.dOf
    of gpuIndex: base = base.iArr
    of gpuDot: base = base.dParent
    of gpuCast: base = base.cExpr
    of gpuBlock: base = base.statements[^1]
    of gpuMaterialize: base = base.mExpr
    else: return asRMEM
  if base == nil or base.symbol == nil:
    return asRMEM
  ctx.varAddressSpaces.getOrDefault(base.symbol.iSym, asRMEM)

proc variantSuffix*(spaces: seq[AddressSpace]): string =
  ## `_`-prefixed suffix of a space tuple, e.g. `_smem`, `_devicesmem`.
  result = "_"
  for space in spaces:
    case space
    of asDevice: result.add "device"
    of asConstant: result.add "const"
    of asSMEM: result.add "smem"
    of asRMEM: result.add "rmem"

proc siteSpaceTuple*(ctx: GpuContext, n: GpuAst): seq[AddressSpace] =
  ## Resolved pointer-field space tuple of an object construction site, in
  ## field declaration order.
  for fname in ptrFieldNames(n.ocType):
    var space = asRMEM
    for f in n.ocFields:
      if f.name == fname and f.value.kind != gpuDiscard:
        space = ctx.resolveValueAddressSpace(f.value)
        break
    result.add space

proc collectValueAddressSpacesImpl*(ctx: var GpuContext, n: GpuAst) =
  ## Records var spaces and objconstr pointer-field spaces for resolution.
  if n == nil:
    return
  case n.kind
  of gpuVar:
    if n.vName.symbol != nil:
      ctx.varAddressSpaces[n.vName.symbol.iSym] = n.addressSpace
  of gpuObjConstr:
    let spaces = ctx.siteSpaceTuple(n)
    var variants = ctx.ptrFieldVariants.getOrDefault(n.ocType, @[])
    if spaces notin variants:
      variants.add spaces
    ctx.ptrFieldVariants[n.ocType] = variants
  else: discard
  for ch in n:
    ctx.collectValueAddressSpacesImpl(ch)

proc paramAddressSpace(p: GpuParam): AddressSpace =
  ## Space of a parameter: the recorded space when one is set (generic
  ## instantiation records arg spaces at clone sites), otherwise derived
  ## from the type — `device` for explicit `ptr T` params, `thread` for
  ## implicit `var T` params and scalars.
  if p.addressSpace != asRMEM:
    p.addressSpace
  elif p.typ != nil and p.typ.kind == gtPtr and not p.typ.implicit:
    asDevice
  else:
    asRMEM

proc collectValueAddressSpaces*(ctx: var GpuContext) =
  ## Precomputes the value-space tables for every function body and global.
  ## Params of generic fns get authoritative entries at their clone sites
  ## (`makeFnGeneric`), which run after this pass.
  for (_, fn) in ctx.fnTab.pairs:
    for p in fn.pParams:
      if p.ident.symbol != nil:
        ctx.varAddressSpaces[p.ident.symbol.iSym] = paramAddressSpace(p)
    ctx.collectValueAddressSpacesImpl(fn)
  for (iSym, g) in ctx.globals.pairs:
    ctx.varAddressSpaces[iSym] = asDevice
  for blk in ctx.globalBlocks:
    ctx.collectValueAddressSpacesImpl(blk)

proc getGenericArguments*(ctx: GpuContext; args: seq[GpuAst]; params: seq[GpuParam]; callerParams: Table[string, GpuParam]): seq[GenericArg] =
  for i, arg in args:
    let p = params[i]
    if p.typ.kind != gtPtr:
      result.add GenericArg(addrSpace: asRMEM, mutable: false)
    else:
      let argIdent = arg.determineIdent()
      var lArg: GpuAst = arg
      if argIdent.kind != gpuDiscard and argIdent.symbol.iSym in callerParams:
        lArg = callerParams[argIdent.symbol.iSym].ident
      let addrSpace = ctx.resolveIdentAddressSpace(lArg)
      let mutable = lArg.determineMutability()
      result.add GenericArg(addrSpace: addrSpace, mutable: mutable)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: patchBoolToI32 pass
# ═══════════════════════════════════════════════════════════════════════════

proc patchBoolToI32Impl*(ctx: var GpuContext; n: var GpuAst) =
  ## Inserts `i32()` cast for idents referencing patched-bool storage types.
  ## In WGSL, `bool` cannot be a storage variable; the type is patched to `i32`.
  ## Any use of such a variable in an expression context needs an explicit
  ## i32() cast or, inversely, a bool() conversion back to bool.
  ##
  ## This pass: For idents with globals of ptr-to-bool or bool, wraps them
  ## appropriately so the emitted code remains valid.
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in ctx.globals:
      let g = ctx.globals[n.symbol.iSym]
      if g.typ.kind == gtBool:
        # Bool global — type was patched to i32 in storage,
        # but expression context expects bool. Insert conv.
        n = GpuAst(kind: gpuConv, convTo: GpuType(kind: gtBool), convExpr: n)
      elif g.typ.kind == gtPtr and g.typ.to.kind == gtBool:
        # ptr-to-bool global — pointer type stays but inner was patched to i32.
        # No conversion needed at pointer level.
        discard
  else:
    for ch in n.mitems:
      ctx.patchBoolToI32Impl(ch)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: renameGlslReserved pass
# ═══════════════════════════════════════════════════════════════════════════

const glslReservedPass6*: array[40, string] = [
  "output", "input", "in", "out", "attribute", "uniform", "varying",
  "buffer", "shared", "layout", "main", "void", "return",
  "if", "else", "for", "while", "break", "continue", "struct",
  "const", "true", "false", "bool", "int", "uint", "float", "double",
  "vec2", "vec3", "vec4", "mat2", "mat3", "mat4",
  "sampler", "texture", "image", "subroutine", "discard", "precise"
]

proc glslSafeNamePass6*(name: string): string =
  if name in glslReservedPass6:
    result = name & "_vk"
  else:
    result = name

proc renameGlslReservedImpl*(ctx: var GpuContext; n: var GpuAst; symToRename: Table[string, string]) =
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in symToRename:
      n.symbol.name = symToRename[n.symbol.iSym]
  else:
    for ch in n.mitems:
      renameGlslReservedImpl(ctx, ch, symToRename)

proc renameGlslReservedPass*(ctx: var GpuContext) =
  ## Renames identifiers in global kernel params that clash with GLSL reserved words.
  for (fnIdent, fn) in ctx.fnTab.mpairs:
    if fn.isGlobalFn():
      var renames = initTable[string, string]()
      for p in fn.pParams.mitems:
        let oldName = p.ident.ident()
        let safeName = glslSafeNamePass6(oldName)
        if oldName != safeName:
          renames[p.ident.symbol.iSym] = safeName
          p.ident.symbol.name = safeName
      if renames.len > 0:
        renameGlslReservedImpl(ctx, fn.pBody, renames)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Vulkan-specific passes
# ═══════════════════════════════════════════════════════════════════════════

proc renameIdentRefsImpl*(n: var GpuAst; symToRename: Table[string, string])

proc compareGpuTypeShallow*(a, b: GpuType): bool =
  ## Shallow comparison of GpuType for SSBO validation.
  ## Compares kind and immediate fields (not deep recursion).
  if a.isNil or b.isNil: return false
  if a.kind != b.kind: return false
  case a.kind
  of gtPtr:
    if a.to.isNil and b.to.isNil: return true
    if a.to.isNil or b.to.isNil: return false
    result = a.to.kind == b.to.kind
  else:
    result = true

proc lowerSsboParamsImpl*(ctx: var GpuContext) =
  ## Scans all kernels, builds canonical SSBO list (deduped by position),
  ## validates type consistency, normalizes parameter names across kernels.
  for (fnIdent, fn) in ctx.fnTab.mpairs:
    if fn.isGlobalFn():
      var ssboIdx = 0
      for p in fn.pParams:
        if p.typ.kind == gtPtr:
          if ssboIdx < ctx.ssboCanonicalInfo.len:
            let (canonName, canonInner) = ctx.ssboCanonicalInfo[ssboIdx]
            if not canonInner.compareGpuTypeShallow(p.typ.to):
              raiseAssert &"Type mismatch at SSBO pos {ssboIdx}"
            if p.ident.ident() != canonName:
              var renames = initTable[string, string]()
              renames[p.ident.symbol.iSym] = canonName
              renameIdentRefsImpl(fn.pBody, renames)
              p.ident.symbol.name = canonName
          else:
            ctx.ssboCanonicalInfo.add (p.ident.ident(), p.typ.to.clone())
          inc ssboIdx

proc renameIdentRefsImpl*(n: var GpuAst; symToRename: Table[string, string]) =
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in symToRename:
      n.symbol.name = symToRename[n.symbol.iSym]
  else:
    for ch in n.mitems:
      renameIdentRefsImpl(ch, symToRename)

proc lowerPushConstantsImpl*(ctx: var GpuContext) =
  ## Lifts non-pointer, non-workspace params into a uniform push-constant block.
  ## Marks them in the context for codegen to emit.
  var pushConstParams: seq[GpuParam]
  for (fnIdent, fn) in ctx.fnTab.mpairs:
    if fn.isGlobalFn():
      for p in fn.pParams:
        if p.typ.kind != gtPtr and p.addressSpace != asSMEM:
          # Check if already added
          let alreadyAdded = pushConstParams.anyIt(
            it.ident.ident() == p.ident.ident() and it.typ.kind == p.typ.kind)
          if not alreadyAdded:
            pushConstParams.add p
  if pushConstParams.len > 0:
    var pcBlock = GpuAst(kind: gpuBlock)
    for p in pushConstParams:
      let comment = GpuAst(kind: gpuComment,
        comment: &"__push_const:{p.ident.ident()}:{$p.typ.kind}")
      pcBlock.statements.add comment
    ctx.globalBlocks.add pcBlock


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: OpenCL-specific passes
# ═══════════════════════════════════════════════════════════════════════════

proc lowerByrefParamsImpl*(ctx: var GpuContext; n: var GpuAst) =
  ## Transforms passByRef parameters into `const Type*` ptr params for OpenCL
  ## (CUDA emits `const Type&` C++ references natively — no body changes there).
  ## The large-struct optimization: structs >= 24 bytes are passed by hidden
  ## const reference instead of by value (isLargeStruct), avoiding the call-site
  ## copy.
  ##
  ## Symbol is a ref type shared between param and body idents. When we rename
  ## the param symbol from "t" to "_p_t", body idents also see "_p_t". Since
  ## _p_t is a pointer, we wrap each body ident in gpuDeref so `t.data[...]`
  ## becomes `(*_p_t).data[...]` (valid C for pointer-to-struct member access).
  ##
  ## No local copy is prepended: body refs resolve through the pointer
  ## (deref-wrap strategy) — a `Type t = *_p_t;` init would be dead code, since
  ## every body ref is already deref-wrapped.
  proc wrapInDeref(body: var GpuAst; renamedSym: Symbol) =
    case body.kind
    of gpuIdent:
      if body.symbol == renamedSym:
        body = GpuAst(kind: gpuDeref, dOf: body)
    else:
      for ch in body.mitems:
        wrapInDeref(ch, renamedSym)
  case n.kind
  of gpuProc:
    let isKernel = attGlobal in n.pAttributes
    if not isKernel:
      var renamedSyms: seq[Symbol]
      for p in n.pParams:
        if p.passByRef:
          let oldSym = p.ident.symbol
          p.ident.symbol.name = "_p_" & p.ident.ident()
          renamedSyms.add oldSym
      # Walk body: wrap renamed idents in gpuDeref (so t → (*_p_t))
      for s in renamedSyms:
        wrapInDeref(n.pBody, s)
    for ch in n.mitems:
      ctx.lowerByrefParamsImpl(ch)
  else:
    for ch in n.mitems:
      ctx.lowerByrefParamsImpl(ch)

proc insertByrefAddrsImpl*(ctx: var GpuContext; n: var GpuAst) =
  ## Wraps byref args in `gpuAddr` nodes at call sites.
  case n.kind
  of gpuCall:
    let fnParams = ctx.getFnParams(n.cName)
    for i, arg in n.cArgs:
      if i < fnParams.len and fnParams[i].passByRef:
        if arg.kind != gpuMaterialize and arg.kind in {gpuIdent, gpuIndex, gpuDeref}:
          n.cArgs[i] = GpuAst(kind: gpuAddr, aOf: arg)
    for ch in n.mitems:
      ctx.insertByrefAddrsImpl(ch)
  else:
    for ch in n.mitems:
      ctx.insertByrefAddrsImpl(ch)
# ═══════════════════════════════════════════════════════════════════════════

proc registerPreprocessingPasses*(reg: var PassRegistry) =
  ## Register preprocessing passes in the correct order.
  ## Order:
  ## 1. rewriteIndexDeref (early — normalizes pointer/index patterns)
  ## 2. decomposeMemcpyVars (after rewriteIndexDeref, before lowering)
  ## 3. annotateTypesForCodegen (after decomposition, IR is stable)
  ## 4. emitFunctionSignatures (after types annotated)
  ## 5. mangleNames (last — runs just before codegen)
  ## (Phase 6 passes are registered separately per backend)

  # ── Pass 1: rewriteIndexDeref ──
  reg.register("rewriteIndexDeref", pkTransform, phaseMain,
    "Normalizes gpuIndex(gpuDeref(...)) to gpuIndex(dOf(...)) for pointer types",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          rewriteIndexDerefImpl(ctx, fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          rewriteIndexDerefImpl(ctx, fn.pBody)
  )

  # ── Pass 2: decomposeMemcpyVars ──
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

  # ── Pass 3: annotateTypesForCodegen ──
  reg.register("annotateTypesForCodegen", pkTransform, phaseMain,
    "Pre-computes backend-neutral type descriptors on all IR nodes",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      discard
      # Type descriptors are computed on-demand via getTypeDesc().
      # This pass registers the mechanism — actual computation is lazy.
  )

  # ── Pass 4: emitFunctionSignatures ──
  reg.register("emitFunctionSignatures", pkTransform, phaseMain,
    "Pre-computes function signature strings on gpuProc nodes",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      emitFunctionSignaturesImpl(ctx)
  )

  # ── Pass 5: mangleNames ──
  reg.register("mangleNames", pkTransform, phaseMain,
    "Applies NamePolicy (base58 hash, clean, patch) to all fnTable entries",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      mangleNamesImpl(ctx)
  )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Backend-specific pass registrations
# These are called from gpu_compiler.nim for each backend
# ═══════════════════════════════════════════════════════════════════════════

proc registerWgslPasses*(reg: var PassRegistry) =
  ## Register WGSL-specific preprocessing passes.
  reg.register("injectAddressOf", pkTransform, phaseMain,
    "Replaces storage-buffer idents with &ident in global fns",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      # Apply pre-WGSL-standard preprocessing
      # 1. pullConstantPragmaVars on globalBlocks[0]
      if ctx.globalBlocks.len > 0:
        pullConstantPragmaVarsImpl(ctx, ctx.globalBlocks[0])
      # 2. removeStructPointerFields on globalBlocks[1]
      if ctx.globalBlocks.len > 1:
        removeStructPointerFieldsImpl(ctx.globalBlocks[1])
      # 3. Remove args from global functions, update syms
      for (fnIdent, fn) in ctx.fnTab.mpairs:
        if fn.isGlobalFn():
          for p in fn.pParams:
            ctx.globals[p.ident.symbol.iSym] = p
          fn.pParams.setLen(0)
          updateSymsInGlobalsImpl(ctx, fn)
      # 4. Scan generics
      let fns = toSeq(ctx.fnTab.pairs)
      for (fnIdent, fn) in fns:
        let fnOrig = ctx.allFnTab[fnIdent]
        var callParams = initTable[string, GpuParam]()
        for p in fnOrig.pParams:
          callParams[p.ident.symbol.iSym] = p
        ctx.scanGenericsImpl(fn, callParams)
  )
  reg.register("injectAddressOfApply", pkTransform, phaseMain,
    "Applies gpuAddr wrapping on global fn idents",
    dependsOn = @["injectAddressOf"],
    run = proc(ctx: var GpuContext): void =
      for (fnIdent, fn) in ctx.fnTab.mpairs:
        if fn.isGlobalFn():
          ctx.injectAddressOfImpl(fn)
  )
  reg.register("makeCodeValid", pkTransform, phaseMain,
    "Addresses WGSL AST patterns (compound assign, struct ptr fields)",
    dependsOn = @["injectAddressOf"],
    run = proc(ctx: var GpuContext): void =
      # Collect value address spaces before call args and pointer aliases
      # are resolved from the authoritative map.
      ctx.collectValueAddressSpaces()
      for (fnIdent, fn) in ctx.fnTab.mpairs:
        ctx.makeCodeValidImpl(fn, inGlobal = fn.isGlobalFn())
  )
  reg.register("checkCodeValidWgsl", pkTransform, phaseMain,
    "Validates WGSL constraints after transformations",
    dependsOn = @["makeCodeValid"],
    run = proc(ctx: var GpuContext): void =
      for (fnIdent, fn) in ctx.fnTab.pairs:
        ctx.checkCodeValidImpl(fn)
  )

proc registerVulkanPasses*(reg: var PassRegistry) =
  ## Register Vulkan-specific preprocessing passes.
  reg.register("lowerSsboParams", pkTransform, phaseMain,
    "Scans kernels, builds canonical SSBO list, normalizes param names",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      lowerSsboParamsImpl(ctx)
  )
  reg.register("lowerPushConstants", pkTransform, phaseMain,
    "Lifts non-ptr non-workspace params into push-constant block metadata",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      lowerPushConstantsImpl(ctx)
  )

proc registerOpenclPasses*(reg: var PassRegistry) =
  ## Register OpenCL-specific preprocessing passes.
  reg.register("lowerByrefParams", pkTransform, phaseMain,
    "Transforms passByRef params to const Type* ptrs (deref-wrapped body idents)",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          ctx.lowerByrefParamsImpl(fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          ctx.lowerByrefParamsImpl(fn.pBody)
  )
  reg.register("insertByrefAddrs", pkTransform, phaseMain,
    "Wraps byref args in gpuAddr nodes at call sites",
    dependsOn = @["lowerByrefParams"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          ctx.insertByrefAddrsImpl(fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          ctx.insertByrefAddrsImpl(fn.pBody)
  )

# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: materializeIndexBuiltinParams pass (Metal)
# ═══════════════════════════════════════════════════════════════════════════
# MSL device functions have no implicit thread index: a body that references a
# canonical coordinate builtin must bind it to a param the caller forwards.
# This pass computes each function's transitive builtin needs once and appends
# one param per need into `pParams`, for kernels and device functions alike,
# with the param's symbol carrying the builtin kind (`coordBuiltin`, resolved
# from the canonical name exactly as the frontend marks builtin symbols). The
# Metal printer then discriminates the attribute form (kernels) from the plain
# form (device functions) by that symbol field alone, and call sites forward
# the canonical names.

proc collectCoordBuiltinIdents(n: GpuAst, acc: var seq[(string, GpuCoordBuiltinKind)]) =
  ## Records coordinate-builtin identifiers in first-use order, deduped by name.
  ## The symbol's `coordBuiltin` decides, never the name: a local or declared
  ## param shadowing a canonical name carries `gbkNone` and is skipped.
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.coordBuiltin != gbkNone:
      let name = n.ident()
      if not acc.anyIt(it[0] == name):
        acc.add (name, n.symbol.coordBuiltin)
  else:
    for ch in n:
      collectCoordBuiltinIdents(ch, acc)

proc collectCallees(n: GpuAst, callees: var seq[GpuAst]) =
  ## Records the functions a body calls, in first-use order, deduped.
  ## Barrier calls are skipped: they lower to a native statement, never a
  ## function call.
  case n.kind
  of gpuCall:
    if n.cName.symbol.synchroBuiltin == gbkNone and not callees.anyIt(it == n.cName):
      callees.add n.cName
    for ch in n:
      collectCallees(ch, callees)
  else:
    for ch in n:
      collectCallees(ch, callees)

proc fnBuiltinNeedsImpl(ctx: GpuContext, fn: GpuAst,
                        acc: var seq[(string, GpuCoordBuiltinKind)],
                        visited: var seq[string]) =
  ## Appends the coordinate builtins `fn` must bind, transitively: the builtins
  ## its own body references, then the needs of every device function it calls,
  ## each name first-seen once. A visited set guards recursive calls.
  let key = fn.pName.symbol.iSym
  if key in visited:
    return
  visited.add key
  collectCoordBuiltinIdents(fn.pBody, acc)
  var callees: seq[GpuAst]
  collectCallees(fn.pBody, callees)
  for calleeIdent in callees:
    let callee = ctx.allFnTab.getOrDefault(calleeIdent,
                                           ctx.genericInsts.getOrDefault(calleeIdent))
    if not callee.isNil:
      if callee.kind == gpuProc:
        fnBuiltinNeedsImpl(ctx, callee, acc, visited)

proc fnBuiltinNeeds(ctx: GpuContext, fn: GpuAst): seq[(string, GpuCoordBuiltinKind)] =
  ## Transitive coordinate-builtin needs of `fn`: its own body, then every
  ## device function it calls. The same first-seen order is used at every
  ## emission site, so the identity-named identifiers bind to the params.
  var visited: seq[string]
  fnBuiltinNeedsImpl(ctx, fn, result, visited)

proc builtinParamType(kind: GpuCoordBuiltinKind): GpuType =
  ## IR type of a coordinate builtin bound as a param: scalar `uint32` for the
  ## flat thread index, the MSL `uint3` vector spelling otherwise. `uint3` is a
  ## synthetic generic name carrying the printer's native spelling; no struct
  ## is ever registered for it.
  if kind in {gbkThreadIndexInThreadgroup, gbkThreadIndexInSimdgroup}:
    GpuType(kind: gtUint32)
  else:
    GpuType(kind: gtGenericInst, gName: "uint3")

proc appendBuiltinParams(fn: GpuAst,
                         needs: seq[(string, GpuCoordBuiltinKind)]) =
  ## Appends one plain param per transitive coordinate-builtin need, after the
  ## declared params, for kernels and device functions alike. The param's
  ## symbol carries the builtin kind (`coordBuiltin` resolved from the
  ## canonical name via the builtin catalog, the same marking the frontend
  ## applies to builtin identifiers), so the printer emits the attribute form
  ## for kernels and the plain form for device functions by the symbol alone.
  for (name, kind) in needs:
    let typ = builtinParamType(kind)
    let sym = newSymbol(name, iSym = name & "_builtin", symKind = gsDeviceKernelParam)
    sym.coordBuiltin = coordBuiltinKind(name)
    let ident = GpuAst(kind: gpuIdent, symbol: sym)
    fn.pParams.add GpuParam(ident: ident, typ: typ,
                            addressSpace: asRMEM, passByRef: false)

proc appendBuiltinForwardingArgs(n: var GpuAst,
                                 needs: seq[(string, GpuCoordBuiltinKind)]) =
  ## Appends one forwarding arg per callee need, in the callee's needs order.
  ## The identity-named identifiers bind to the caller's own params (kernel
  ## attribute params or device-fn hidden params) in the emitted source.
  for (name, _) in needs:
    let sym = newSymbol(name, iSym = name & "_builtin", symKind = gsLocal)
    n.cArgs.add GpuAst(kind: gpuIdent, symbol: sym)

proc materializeCallArgsImpl(ctx: GpuContext, n: var GpuAst,
                             needs: Table[string, seq[(string, GpuCoordBuiltinKind)]]) =
  ## Appends the forwarding args to every `gpuCall` whose callee has
  ## coordinate-builtin needs, in the callee's needs order. The walk happens
  ## before the append so the argument list is not mutated mid-iteration.
  case n.kind
  of gpuCall:
    let callee = ctx.allFnTab.getOrDefault(n.cName,
                                           ctx.genericInsts.getOrDefault(n.cName))
    var calleeNeeds: seq[(string, GpuCoordBuiltinKind)]
    if not callee.isNil:
      if callee.kind == gpuProc:
        calleeNeeds = needs.getOrDefault(callee.pName.symbol.iSym, @[])
    for ch in mitems(n):
      materializeCallArgsImpl(ctx, ch, needs)
    appendBuiltinForwardingArgs(n, calleeNeeds)
  else:
    for ch in mitems(n):
      materializeCallArgsImpl(ctx, ch, needs)

proc materializeIndexBuiltinParamsImpl*(ctx: var GpuContext) =
  ## Materializes coordinate-builtin binding for the Metal backend:
  ## - every function receives its transitive needs as params appended to
  ##   `pParams`, after the declared params, the symbol carrying the builtin kind
  ## - every call site forwards the callee's needs as trailing args
  ##
  ## MSL device functions have no implicit thread index, so a body that
  ## references a canonical coordinate builtin binds it to a param the caller
  ## forwards. The closure analysis runs first, once per function, so every
  ## rewrite reads the same memoized needs in the same first-seen order.
  var needs = initTable[string, seq[(string, GpuCoordBuiltinKind)]]()
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      needs[fn.pName.symbol.iSym] = fnBuiltinNeeds(ctx, fn)
  for fnKey in ctx.genericInsts.keys:
    var fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      needs[fn.pName.symbol.iSym] = fnBuiltinNeeds(ctx, fn)

  # A pulled-in device function is registered in both `allFnTab` and
  # `genericInsts` under the same symbol: rewrite each function once.
  # Kernels and device functions alike receive the needs as params; the
  # printer's attribute form vs plain form is decided by the symbol's
  # `coordBuiltin` at emission time.
  var done = initHashSet[string]()
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      let key = fn.pName.symbol.iSym
      if key in done:
        continue
      done.incl key
      appendBuiltinParams(fn, needs[key])
      materializeCallArgsImpl(ctx, fn.pBody, needs)
  for fnKey in ctx.genericInsts.keys:
    var fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      let key = fn.pName.symbol.iSym
      if key in done:
        continue
      done.incl key
      appendBuiltinParams(fn, needs[key])
      materializeCallArgsImpl(ctx, fn.pBody, needs)

proc registerMetalPasses*(reg: var PassRegistry) =
  ## Register Metal-specific preprocessing passes.
  reg.register("materializeIndexBuiltinParams", pkTransform, phaseMain,
    "Materializes coordinate-builtin binding: builtin params appended to pParams, call-site forwarding args",
    dependsOn = @["emitFunctionSignatures"],
    run = proc(ctx: var GpuContext): void =
      materializeIndexBuiltinParamsImpl(ctx)
  )
