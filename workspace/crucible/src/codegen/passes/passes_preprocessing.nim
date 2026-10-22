## Backend Pass Extraction
##
## Passes that eliminate smart logic from backends, codegen renders
## straightforward IR to target-language strings only.

import std / [sequtils, tables, sets, strutils, strformat, options]
import ../ir/gpu_ast_datatypes
import ../frontend/nim_to_gpu_type_resolvers
import ../ir/gpu_ast_type_resolvers
import ./pass_datatypes
import ./passes_legalizations

# ############################################################
#
# Preprocessing
#
# ############################################################
#
# Symbol naming: mangling policies, generics scanning and span
# rewriting ahead of lowering. Every backend registers them identically.


{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

func registerPreprocessingPasses*(reg: var PassRegistry) =
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

  reg.register("annotateTypesForCodegen", pkTransform, phaseMain,
    "Pre-computes backend-neutral type descriptors on all IR nodes",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      discard
      # descriptors are computed on-demand via getTypeDesc(), the pass only
      # registers the mechanism
  )

  reg.register("mangleNames", pkTransform, phaseMain,
    "Applies NamePolicy (base58 hash, clean, patch) to all fnTable entries",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      mangleNamesImpl(ctx)
  )




func isGlobalFn*(fn: GpuAst): bool =
  doAssert fn.kind == gpuProc
  result = attGlobal in fn.pAttributes

# ═══════════════════════════════════════════════════════════════════════════
# rewriteIndexDeref
# ═══════════════════════════════════════════════════════════════════════════

proc rewriteIndexDerefImpl(ctx: var GpuContext, n: var GpuAst) =
  ## `(*arr)[idx]` ≡ `arr[idx]` in C, the IR keeps the simpler form.
  case n.kind
  of gpuIndex:
    if n.iArr.kind == gpuDeref:
      # ptr-to-array keeps the deref (C array decay), plain ptr rewrites
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


# ═══════════════════════════════════════════════════════════════════════
# decomposeMemcpyVars
# ═══════════════════════════════════════════════════════════════════════════

proc genMemcpyCall(lhs, rhs: GpuAst, sizeExpr: GpuAst): GpuAst =
  let memcpySym = newSymbol("__builtin_memcpy", iSym = "__builtin_memcpy", symKind = gsProc)
  let memcpyIdent = GpuAst(kind: gpuIdent, symbol: memcpySym)
  var call = GpuAst(kind: gpuCall)
  call.cName = memcpyIdent
  call.cArgs = @[lhs, rhs, sizeExpr]
  result = call

func genSizeofCall*(ctx: var GpuContext, expr: GpuAst): GpuAst =
  let sizeSym = newSymbol("__builtin_sizeof", iSym = "__builtin_sizeof", symKind = gsProc)
  let sizeIdent = GpuAst(kind: gpuIdent, symbol: sizeSym)
  var call = GpuAst(kind: gpuCall)
  call.cName = sizeIdent
  call.cArgs = @[expr]
  result = call

proc decomposeMemcpyVarsImpl*(ctx: var GpuContext, n: var GpuAst) =
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
# Type descriptors
# ═══════════════════════════════════════════════════════════════════════════

type
  TypeDescKind* = enum
    tdkVoid
    tdkBool
    tdkInt8
    tdkUint8
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
    tdkF8E4M3
    tdkF8E5M2
    tdkSize
    tdkPtr       # pointer to a type descriptor
    tdkArray
    tdkStruct    # named struct with fields
    tdkString    # const char star
    tdkVoidPtr
    tdkGeneric   # generic instantiation name
    tdkUnresolved

  TypeDesc* = ref object
    case kind*: TypeDescKind
    of tdkVoid, tdkBool, tdkInt8, tdkUint8, tdkInt16, tdkUint16,
       tdkInt32, tdkUint32, tdkInt64, tdkUint64, tdkFloat32, tdkFloat64,
       tdkFloat16, tdkBf16, tdkF8E4M3, tdkF8E5M2, tdkSize, tdkString,
       tdkVoidPtr, tdkUnresolved:
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
      tdArgs*: seq[TypeDesc]
func gpuTypeToDesc*(t: GpuType): TypeDesc =
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
  of gtF8E4M3:   TypeDesc(kind: tdkF8E4M3)
  of gtF8E5M2:   TypeDesc(kind: tdkF8E5M2)
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
    for f in t.fields:
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
  of gtCompileTimeValue:
    # the type slot holds the compile-time argument, there is no descriptor
    TypeDesc(kind: tdkUnresolved)

proc getTypeDesc*(ctx: var GpuContext, arg: GpuAst): TypeDesc =
  ## Replaces the recursive getType/getFieldType calls during codegen
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
# mangleNames
# ═══════════════════════════════════════════════════════════════════════════

func mangleNamesImpl*(ctx: var GpuContext) =
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
      ident.symbol.name = mangledHashName(name, iSym)

    of npPatch:
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


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

func rewriteCompoundAssignmentImpl*(n: GpuAst): GpuAst =
  ## LHS may arrive gpuAddr-wrapped, both sides use a bare lvalue, the RHS
  ## clones the LHS (legalization mutates it in place, the sides must not alias)
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

# ─── Value address-space resolution (Metal, WebGPU) ────────────────────────────
# gtPtr carries no address space, the Metal and WebGPU printers resolve it
# via value dataflow with an asRMEM fallback.

func ptrFieldNames*(t: GpuType): seq[string] =
  ## Pointer field names of a struct type, in declaration order.
  case t.kind
  of gtObject, gtGenericInst:
    for f in t.fields:
      if f.typ.kind == gtPtr:
        result.add f.name
  else: discard

proc resolveValueAddressSpace*(ctx: GpuContext, n: GpuAst): AddressSpace =
  ## AsRMEM fallback
  if n == nil:
    return asRMEM
  case n.kind
  of gpuIdent:
    if n.symbol == nil:
      asRMEM
    elif n.symbol.iSym in ctx.varAddressSpaces:
      ctx.varAddressSpaces[n.symbol.iSym]
    elif n.symbol.symKind in {gsDeviceKernelParam, gsGlobalKernelParam}:
      # explicit `ptr T` is a device pointer, implicit `var T` is thread
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
    let ptype = ctx.getDotParentType(n.dParent)
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
  ## Returns the address space recorded for the base ident's var or param,
  ## asRMEM when nothing is recorded.
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

proc siteSpaceTuple*(ctx: GpuContext, n: GpuAst): seq[AddressSpace] =
  ## Pointer-field spaces of an objconstr site, in declaration order
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
  ## Returns the param's recorded space when one is set, else derives it
  ## from the type: device for explicit `ptr T`, thread otherwise.
  if p.addressSpace != asRMEM:
    p.addressSpace
  elif p.typ != nil and p.typ.kind == gtPtr and not p.typ.implicit:
    asDevice
  else:
    asRMEM

proc collectValueAddressSpaces*(ctx: var GpuContext) =
  ## Generic-fn params get authoritative entries at their clone sites (makeFnGeneric)
  for (_, fn) in ctx.fnTab.pairs:
    for p in fn.pParams:
      if p.ident.symbol != nil:
        ctx.varAddressSpaces[p.ident.symbol.iSym] = paramAddressSpace(p)
    ctx.collectValueAddressSpacesImpl(fn)
  for (iSym, g) in ctx.globals.pairs:
    ctx.varAddressSpaces[iSym] = asDevice
  for blk in ctx.globalBlocks:
    ctx.collectValueAddressSpacesImpl(blk)

# ═══════════════════════════════════════════════════════════════════════════