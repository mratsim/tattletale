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

  reg.register("mangleNames", pkTransform, phaseMain,
    "Applies NamePolicy (base58 hash, clean, patch) to all fnTable entries",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      mangleNamesImpl(ctx)
  )





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
func gpuTypeToDesc(t: GpuType): TypeDesc =
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

proc getTypeDesc(ctx: var GpuContext, arg: GpuAst): TypeDesc =
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

func mangleNamesImpl(ctx: var GpuContext) =
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


