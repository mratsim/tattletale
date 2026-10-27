# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, strutils, sequtils, options, tables, sets]

import ../ir/ast_datatypes
import ../ir/ast_type_constructors
import ../../builtins/builtins_functions
import ./nim_ast_helpers
import ./nim_type_resolvers
import ../ir/ast_type_resolvers
import ../passes/pass_registry

# ############################################################
#
#                    Nim AST -> Crucible AST
#
# ############################################################

proc toCrucibleAst*(ctx: var GpuContext, reg: var TypeRegistry, node: NimNode, staticParamsPositions: seq[int] = @[]): GpuAst

proc parseProcParameters(ctx: var GpuContext, reg: var TypeRegistry, params: NimNode, attrs: set[GpuAttribute],
                         staticParamsPositions: seq[int] = @[]): seq[GpuParam] =
  doAssert params.kind == nnkFormalParams, "Argument is not FormalParams, but: " & $params.treerepr
  var curIdx = 0
  for i in 1 ..< params.len:
    let param = params[i]
    let numParams = param.len - 2
    let typIdx = param.len - 2 # second to last is the type
    let marked = curIdx in staticParamsPositions or isTypeDescNode(param[typIdx-1])
    for j in 0 ..< numParams:
      var p = ctx.toCrucibleAst(reg, param[j])
      let symKind = if attGlobal in attrs: gsGlobalKernelParam
                    else: gsDeviceKernelParam
      p.symbol.symKind = symKind
      var paramType: GpuType
      if marked:
        paramType = GpuType(kind: gtCompileTimeValue)
      else:
        paramType = resolveType(reg, param[typIdx-1].getTypeInst())
      p.symbol.typ = paramType
      let byref = paramType.kind != gtCompileTimeValue and isLargeStruct(paramType)
      # birth space only: `ptr T` defaults device, `var T` (ntyVar) registers, both re-spaced by call sites and {.smem.} hoisting
      let paramAddrSpace =
        if paramType.kind == gtPtr and not paramType.implicit: asDevice
        else: asRMEM
      result.add(GpuParam(ident: p, typ: paramType,
                          addressSpace: paramAddrSpace, passByRef: byref,
                          flavor: (if paramType.kind == gtPtr and paramType.implicit: kVar else: kIn)))
    curIdx += numParams

proc toInstantiatedProcSignature(ctx: var GpuContext, reg: var TypeRegistry,
    params: NimNode, attrs: set[GpuAttribute],
    staticParamsPositions: seq[int] = @[]): GpuProcSignature =
  ## Only called on generically instantiated procs.
  GpuProcSignature(
    params: ctx.parseProcParameters(reg, params, attrs, staticParamsPositions),
    retType: resolveProcReturnType(reg, params)
  )

proc getFnName(ctx: var GpuContext, reg: var TypeRegistry, n: NimNode): GpuAst =
  ## Symbol name, or the `{.cudaName.}` pragma argument.
  template toAst(fn): untyped = GpuAst(kind: gpuIdent, symbol: newSymbol(fn, symKind = gsProc))

  if n.kind == nnkSym:
    var sig = n.repr & "_" & n.signatureHash()
    if n.symKind in {nskProc, nskFunc} and isGenericProc(n):
      # `signatureHash` erases static-param bindings, so instantiations share one key.
      sig &= "_" & n.symBodyHash()
    if sig in ctx.sigTab:
      result = ctx.sigTab[sig]
    else:
      let impl = n.getImpl
      if impl.kind in [nnkProcDef, nnkFuncDef]:
        let pragma = impl.pragma
        if pragma.kind != nnkEmpty and pragma[0].kind == nnkExprColonExpr:
          if pragma[0][0].kind in [nnkIdent, nnkSym] and pragma[0][0].strVal == "cudaName":
            result = toAst pragma[0][1].strVal
            ctx.sigTab[sig] = result
          else:
            result = ctx.toCrucibleAst(reg, n)
        else:
          result = ctx.toCrucibleAst(reg, n)
      else:
        result = ctx.toCrucibleAst(reg, n)
      # patch operator names (`+` -> `add`) into valid C++ identifiers before iSym generation
      if result.symbol.name in ["+", "-", "*", "/"]:
        let oldName = result.symbol.name
        case result.symbol.name
        of "+": result.symbol.name = "add"
        of "-": result.symbol.name = "sub"
        of "*": result.symbol.name = "mul"
        of "/": result.symbol.name = "div"
        else: discard
        result.symbol.iSym = result.symbol.iSym.replace(oldName, result.symbol.name)
      if n.strVal in ctx.symChoices:
        if not hasPragma(n.getImpl(), "magic") and not hasPragma(n.getImpl(), "builtin"):
          let id = ctx.sigTab[sig]
          id.symbol.name = id.symbol.iSym
      else:
        ctx.symChoices.incl result.symbol.name
  else:
    result = toAst n.repr
  result.symbol.symKind = gsProc # make sure it's a proc

proc addToFnTable(ctx: var GpuContext, ident: GpuAst, body: GpuAst, kind: set[FunctionKind]) =
  let key = ident.symbol.iSym
  # last writer wins
  ctx.fnTable[key] = FnTableEntry(
    ident: ident,
    body: body,
    kind: kind,
    namePolicy: npUnassigned
  )

proc registerGenericInstOrExternalProc(ctx: var GpuContext, reg: var TypeRegistry, node: NimNode, name: GpuAst) =
  ## Stores the instantiation in `genericInsts`, attaching `{.device.}`,
  ## and renames the callee to the instantiation's iSym.
  # templates must be fully expanded by Nim before reaching crucible
  let rawImpl = node[0].getImpl()
  if rawImpl.kind in [nnkTemplateDef, nnkMacroDef]:
    error("Unresolved " & $rawImpl.kind & " encountered in GPU code: " & node[0].repr &
          ". Template/macro expansion must complete before the cuda: block.")

  let inst = rawImpl
  let sig = node[0].getTypeInst()

  var staticParamsPositions: seq[int]
  block:
    var curIdx = 0
    for i in 1 ..< inst.params.len:
      let p = inst.params[i]
      let numParams = p.len - 2
      let typIdx = p.len - 2
      # `static int` arrives as nnkCommand, `static[int]` as nnkBracketExpr
      if p[typIdx].kind in {nnkCommand, nnkBracketExpr} and
          p[typIdx].len > 0 and p[typIdx][0].repr == "static":
        for j in 0 ..< numParams:
          staticParamsPositions.add(curIdx + j)
      curIdx += numParams

  inst.params = sig.params
  let attrs = collectProcAttributes(inst.pragma)
  let procSig = ctx.toInstantiatedProcSignature(reg, sig.params, attrs, staticParamsPositions)
  if name in ctx.processedProcs:
    return
  else:
    ctx.processedProcs[name] = procSig

  if node[0].repr in NimGpuNumericBuiltinsFunctions:
    if hasPragma(inst, "magic"):
      let retType = resolveType(reg, sig.params[0])
      var builtinFn = GpuAst(kind: gpuProc, pName: name, pRetType: retType, pAttributes: {attDevice})
      ctx.builtinFns[name] = builtinFn
      return

  if node[0].repr in NimGpuNumericBuiltinsOperators or
     node[0].repr in NimGpuBooleanOperators:
    if hasPragma(inst, "magic"):
      let retType = resolveType(reg, sig.params[0])
      var builtinFn = GpuAst(kind: gpuProc, pName: name, pRetType: retType, pAttributes: {attDevice})
      ctx.builtinFns[name] = builtinFn
      return

  # function-style magic builtins (toOpenArray, ...), registered without parsing bodies
  if node[0].repr in NimGpuFnBuiltins:
    let retType = resolveType(reg, sig.params[0])
    var builtinFn = GpuAst(kind: gpuProc, pName: name, pRetType: retType, pAttributes: {attDevice})
    ctx.builtinFns[name] = builtinFn
    ctx.addToFnTable(name, builtinFn, {fkBuiltin})
    return

  let fn = ctx.toCrucibleAst(reg, inst, staticParamsPositions)
  if fn.kind == gpuDiscard:
    doAssert inst.isBuiltIn()
    return
  fn.pAttributes.incl attDevice
  fn.pName = name
  # Display name gets the 7-char hash suffix (npHashSuffix shape), one
  # distinct iSym per generic instantiation
  fn.pName.symbol.name = mangledHashName(fn.pName.symbol.name, fn.pName.symbol.iSym)
  name.symbol.name = fn.pName.symbol.name
  ctx.genericInsts[fn.pName] = fn
  # Iterating passes like `ensureBlock` only see instantiations in `allFnTab`.
  ctx.allFnTab[fn.pName] = fn

# ═══════════════════════════════════════════════════════════════════════
# Small builders and queries
# ═══════════════════════════════════════════════════════════════════════

proc fnReturnsValue(ctx: GpuContext, fn: GpuAst): bool =
  ## Checks fnTable, allFnTab, genericInsts, builtinFns, processedProcs.
  let key = fn.symbol.iSym
  if key in ctx.fnTable:
    let entry = ctx.fnTable[key]
    if not entry.body.isNil and entry.body.kind == gpuProc:
      result = entry.body.pRetType.kind != gtVoid
    else:
      result = false
  elif fn in ctx.allFnTab:
    result = ctx.allFnTab[fn].pRetType.kind != gtVoid
  elif fn in ctx.genericInsts:
    result = ctx.genericInsts[fn].pRetType.kind != gtVoid
  elif fn in ctx.builtinFns:
    result = ctx.builtinFns[fn].pRetType.kind != gtVoid
  elif fn in ctx.processedProcs:
    result = ctx.processedProcs[fn].retType.kind != gtVoid
  else:
    error "The function: " & $fn & " is not known anywhere."

proc buildEmitStmt(ctx: var GpuContext, reg: var TypeRegistry, emitArg: NimNode): GpuAst =
  result = GpuAst(kind: gpuEmit)
  for part in emitArg:
    case part.kind
    of nnkStrLit, nnkTripleStrLit, nnkRStrLit:
      result.parts.add GpuEmitPart(kind: peLiteral, literal: part.strVal)
    else:
      result.parts.add GpuEmitPart(kind: peExpr, expr: ctx.toCrucibleAst(reg, part))

proc toConstDefName(ctx: var GpuContext, identNode: NimNode): GpuAst =
  ## A const definition's own name must never inline into itself.
  if identNode.kind in {nnkIdent, nnkOpenSymChoice}:
    result = newGpuIdent()
    result.symbol.name = identNode.repr
    return
  let sanitized = identNode.repr.multiReplace(("`", "_"))
  let s = sanitized & "_" & identNode.signatureHash()
  if s notin ctx.sigTab:
    result = newGpuIdent()
    result.symbol.name = sanitized
    result.symbol.iSym = s
    if result.symbol.name == "_":
      result.symbol.name = "underscore"
    elif result.symbol.name.startsWith("tmpTuple_"):
      result.symbol.name = "tmpTuple_" & $ctx.genSymCount
      result.symbol.iSym = result.symbol.name & "_" & identNode.signatureHash()
      inc ctx.genSymCount
    ctx.sigTab[s] = result
  else:
    result = ctx.sigTab[s]

# ═══════════════════════════════════════════════════════════════════════
#          Nim -> Crucible AST
# ═══════════════════════════════════════════════════════════════════════

proc toCrucibleAst*(ctx: var GpuContext, reg: var TypeRegistry, node: NimNode, staticParamsPositions: seq[int] = @[]): GpuAst =
  ## XXX: `result` variable not supported yet

  case node.kind
  # ═══════════════════════════════════════════════════════════════════════
  # Trivia
  # ═══════════════════════════════════════════════════════════════════════

  of nnkEmpty: result = GpuAst(kind: gpuDiscard) # nothing to do
  of nnkDiscardStmt:
    result = ctx.toCrucibleAst(reg, node[0])

  of nnkCommentStmt:
    result = GpuAst(kind: gpuDiscard)

  # ═══════════════════════════════════════════════════════════════════════
  # Blocks and scopes
  # ═══════════════════════════════════════════════════════════════════════

  of nnkStmtList:
    result = GpuAst(kind: gpuBlock)
    let prevScope = ctx.currentScope
    ctx.scopeSymsStack.add(ctx.currentScopeSyms)
    ctx.currentScopeSyms = @[]
    ctx.currentScope = result
    for el in node:
      result.statements.add ctx.toCrucibleAst(reg, el)
    ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
    ctx.currentScope = prevScope

  of nnkBlockStmt:
    let blockLabel = if node[0].kind in {nnkSym, nnkIdent}: node[0].strVal
                     elif node[0].kind == nnkEmpty: ""
                     else: error "Unexpected node in block label field: " & $node.treerepr
    result = GpuAst(kind: gpuBlock,
                    blockLabel: blockLabel,
                    )
    let prevScopeBlk = ctx.currentScope
    ctx.scopeSymsStack.add(ctx.currentScopeSyms)
    ctx.currentScopeSyms = @[]
    ctx.currentScope = result
    for i in 1 ..< node.len: # index 0 is the block label
      result.statements.add ctx.toCrucibleAst(reg, node[i])
    ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
    ctx.currentScope = prevScopeBlk

  of nnkBlockExpr:
    let blockLabel = if node[0].kind in {nnkSym, nnkIdent}: node[0].strVal
                     elif node[0].kind == nnkEmpty: ""
                     else: error "Unexpected node in block label field: " & $node.treerepr
    result = GpuAst(kind: gpuBlock, blockLabel: blockLabel, isExpr: true,
                    )
    let prevScopeBlkExpr = ctx.currentScope
    ctx.scopeSymsStack.add(ctx.currentScopeSyms)
    ctx.currentScopeSyms = @[]
    ctx.currentScope = result
    for el in node:
      if el.kind != nnkEmpty:
        result.statements.add ctx.toCrucibleAst(reg, el)
    ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
    ctx.currentScope = prevScopeBlkExpr
    # the type lives on node[^1], the blit pass reads it via scope lookups
  of nnkStmtListExpr: # statements that return a value
    # single-expression form unwraps to the expression itself, multi-statement
    # builds a block expr the blit pass lowers
    let exprs = node.filterIt(it.kind != nnkEmpty)
    if exprs.len == 1:
      result = ctx.toCrucibleAst(reg, exprs[0])
    else:
      result = GpuAst(kind: gpuBlock, isExpr: true,
                      )
      let prevScopeStmtListExpr = ctx.currentScope
      ctx.scopeSymsStack.add(ctx.currentScopeSyms)
      ctx.currentScopeSyms = @[]
      ctx.currentScope = result
      for el in node:
        if el.kind != nnkEmpty:
          result.statements.add ctx.toCrucibleAst(reg, el)
      ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
      ctx.currentScope = prevScopeStmtListExpr
    # the type lives on node[^1], the blit pass reads it via scope lookups

  # ═══════════════════════════════════════════════════════════════════════
  # Definitions
  # ═══════════════════════════════════════════════════════════════════════

  of nnkProcDef, nnkFuncDef:
    # generics only register by name here, the body parses at call time (genericInsts)
    let name = ctx.getFnName(reg, node.name)
    if node[2].kind == nnkGenericParams: # is a generic
      ctx.generics.incl name.symbol.name # need to use raw name, *not* symbol
      result = GpuAst(kind: gpuDiscard)
    elif node.body.kind == nnkEmpty: # just a forward declaration
      result = GpuAst(kind: gpuDiscard)
    else:
      result = GpuAst(kind: gpuProc)
      result.pName = name
      result.pName.symbol.symKind = gsProc ## This is a procedure identifier
      let params = node[3]
      doAssert params.kind == nnkFormalParams
      result.pRetType = resolveProcReturnType(reg, params)
      if result.pRetType.kind == gtInvalid:
        ctx.generics.incl name.symbol.name # need to use raw name, *not* symbol
        return GpuAst(kind: gpuDiscard)

      # Process pragmas
      if node.pragma.kind != nnkEmpty:
        doAssert node.pragma.len > 0, "Pragma kind non empty, but no pragma?"
        result.pRawPragmas = collectRawPragmas(node.pragma)
        result.pAttributes = collectProcAttributes(node.pragma)
        result.pWorkgroupSize = parseWorkgroupSize(node.pragma)
        if result.pAttributes.len == 0: # means a `builtin` pragma was applied
          ctx.builtinFns[name] = result # store in builtinFns, so that we know if it returns a value when called
          ctx.addToFnTable(name, result, {fkBuiltin})
          return GpuAst(kind: gpuDiscard)
      # Process parameters
      result.pParams = ctx.parseProcParameters(reg, params, result.pAttributes, staticParamsPositions)
      result.pBody = ctx.toCrucibleAst(reg, node.body)
      # Validation and transform passes run via ctx.runPasses()
      # Add to table of known functions (both old and new)
      if result.pName notin ctx.allFnTab:
        ctx.allFnTab[result.pName] = result
        ctx.addToFnTable(name, result, {fkDefined})

  of nnkLetSection, nnkVarSection:
    result = GpuAst(kind: gpuBlock)
    for declaration in node:
      var varNode = GpuAst(kind: gpuVar)
      case declaration[0].kind
      of nnkIdent, nnkSym:
        varNode.vName = ctx.toCrucibleAst(reg, declaration[0])
        # unannotated locals are per-thread registers, an address-space pragma overrides below
        varNode.addressSpace = asRMEM
      of nnkPragmaExpr:
        varNode.vName = ctx.toCrucibleAst(reg, declaration[0][0])
        doAssert declaration[0][1].kind == nnkPragma
        varNode.addressSpace = collectAddressSpace(declaration[0][1])
      else: error "Unexpected node kind for variable: " & $declaration.treeRepr
      varNode.vType = resolveType(reg, declaration)
      if varNode.addressSpace != asRMEM and
          varNode.vType.kind in {gtPtr, gtSpan, gtArray}:
        varNode.vType = varNode.vType.withAddressSpace(varNode.addressSpace)
      varNode.vName.symbol.typ = varNode.vType # type cached on the symbol for later lookup
      varNode.vMutable = node.kind == nnkVarSection
      ## XXX: array initialization needs a memcpy, not yet handled
      if declaration.len > 2 and declaration[2].kind != nnkEmpty:
        varNode.vInit = ctx.toCrucibleAst(reg, declaration[2])
        if varNode.vType.kind == gtPtr:
          # an initialized pointer takes the space of its initializer's storage
          let initSpace = ctx.resolveAddressSpace(varNode.vInit)
          if initSpace != varNode.vType.addressSpace:
            varNode.vType = varNode.vType.withAddressSpace(initSpace)
            varNode.vName.symbol.typ = varNode.vType
        if varNode.vType.kind in {gtObject, gtGenericInst} and
            varNode.vInit.kind == gpuObjConstr:
          # one declared Holder, each construction gets its own field types:
          # Holder(p: addr outp[0]) types the field a device ptr,
          # Holder(p: addr tmp) a thread ptr, the declared type stays untouched
          var instType = varNode.vType.clone()
          for i in 0 ..< instType.fields.len:
            for fc in varNode.vInit.ocFields:
              if fc.name == instType.fields[i].name and
                  fc.typ.kind in {gtPtr, gtSpan, gtArray} and
                  fc.typ.addressSpace != instType.fields[i].typ.addressSpace:
                instType.fields[i].typ = fc.typ
          varNode.vType = instType
          varNode.vName.symbol.typ = instType
        varNode.vRequiresMemcpy = requiresMemcpy(declaration[2])
        result.statements.add(varNode)
      else:
        varNode.vInit = GpuAst(kind: gpuDiscard)
        result.statements.add(varNode)
      # Register variable in current scope's symbol table
      if ctx.currentScope != nil:
        scopeAdd(ctx.currentScopeSyms, varNode.vName.symbol.name, varNode.vName.symbol)

  of nnkTemplateDef, nnkMacroDef:
    # templates arrive already expanded by Nim
    return GpuAst(kind: gpuDiscard)

  of nnkTypeSection:
    result = GpuAst(kind: gpuBlock)
    for el in node: # walk each type def
      doAssert el.kind == nnkTypeDef
      result.statements.add ctx.toCrucibleAst(reg, el)

  of nnkTypeDef:
    doAssert node.len == 3, "TypeDef node does not have 3 children: " & $node.len
    if node[1].kind == nnkGenericParams: # instantiations are handled at nnkObjConstr
      result = GpuAst(kind: gpuDiscard)
    else:
      let typ = resolveType(reg, node[0])
      # aliases resolved to primitives need no typedef, the target knows the type
      let isBuiltin = typ.kind notin {gtObject, gtGenericInst}
      case node[2].kind
      of nnkObjectTy:
        result = GpuAst(kind: gpuTypeDef, tTyp: typ)
        result.tFields = resolveRecordFields(reg, node[2])
      of nnkCall:
        result = if isBuiltin: GpuAst(kind: gpuDiscard)
                 else: GpuAst(kind: gpuTypeDef, tTyp: typ)
      of nnkSym:      # a type alias `type foo = bar`
        if node[2].typeKind in {ntyObject, ntyTuple, ntyGenericInst}:
          result = GpuAst(kind: gpuAlias, aTyp: typ,
                          aTo: ctx.toCrucibleAst(reg, node[2]))
        else:
          result = GpuAst(kind: gpuDiscard)
      else:
        error "Unexpected node kind in TypeDef: " & $node[2].kind

      # deduplicates via the known-types set
      ctx.types[typ] = result
      # all types are emitted from ctx.types, not from here
      result = GpuAst(kind: gpuDiscard)

  of nnkConstDef:
    let identNode = if node[0].kind == nnkPragmaExpr: node[0][0] else: node[0]
    result = GpuAst(kind: gpuConstexpr,
                    cIdent: ctx.toConstDefName(identNode),
                    cValue: ctx.toCrucibleAst(reg, node[2]),
                    cType: resolveType(reg, node))
    result.cIdent.symbol.typ = result.cType # type cached on the symbol for later lookup
    result.cIdent.symbol.symKind = gsLocal # constexpr is a per-thread compile-time value

  of nnkConstSection:
    result = GpuAst(kind: gpuBlock)
    for el in node: # walk each type def
      doAssert el.kind == nnkConstDef
      result.statements.add ctx.toCrucibleAst(reg, el)

  # ═══════════════════════════════════════════════════════════════════════
  # Control flow
  # ═══════════════════════════════════════════════════════════════════════

  of nnkAsgn:
    result = GpuAst(kind: gpuAssign)
    result.aLeft = ctx.toCrucibleAst(reg, node[0])
    result.aRight = ctx.toCrucibleAst(reg, node[1])
    result.aRequiresMemcpy = requiresMemcpy(node[1])

  of nnkIfStmt:
    result = GpuAst(kind: gpuIf, ifIsExpr: false)
    let branch = node[0]  # First branch
    result.ifCond = ctx.toCrucibleAst(reg, branch[0])
    result.ifThen = ctx.toCrucibleAst(reg, branch[1])
    for i in 1 ..< node.len:
      case node[i].kind
      of nnkElifBranch:
        result.ifElifs.add (cond: ctx.toCrucibleAst(reg, node[i][0]),
                            body: ctx.toCrucibleAst(reg, node[i][1]))
      of nnkElse:
        result.ifElse = ctx.toCrucibleAst(reg, node[i][0])
      else:
        error "Unexpected child in nnkIfStmt: " & $node[i].kind
    if result.ifElse.isNil:
      result.ifElse = GpuAst(kind: gpuDiscard)

  of nnkIfExpr:
    proc requireSimpleBranch(n: NimNode) =
      let ok = n.kind != nnkStmtList or n.len == 1
      doAssert ok, "GPU if-expression branches must be single expressions, " &
        "not blocks with statements (would hurt performance)"
    doAssert node.len >= 1
    doAssert node.len == 1 or node[^1].kind in {nnkElseExpr, nnkElse},
      "GPU if-expression must have an else branch"
    proc buildIfExpr(ctx: var GpuContext, reg: var TypeRegistry, branches: NimNode, idx: int): GpuAst =
      let child = branches[idx]
      case child.kind
      of nnkElifExpr, nnkElifBranch:
        let body = unwrapSingleStmt(child[1])
        requireSimpleBranch(body)
        result = GpuAst(kind: gpuIf, ifIsExpr: true)
        result.ifCond = ctx.toCrucibleAst(reg, child[0])
        result.ifThen = ctx.toCrucibleAst(reg, body)
        if idx + 1 < branches.len:
          result.ifElse = buildIfExpr(ctx, reg, branches, idx + 1)
        else:
          result.ifElse = GpuAst(kind: gpuDiscard)
      of nnkElseExpr, nnkElse:
        let body = unwrapSingleStmt(child[0])
        requireSimpleBranch(body)
        result = ctx.toCrucibleAst(reg, body)
      else:
        error "Unexpected child in nnkIfExpr: " & $child.kind
    result = buildIfExpr(ctx, reg, node, 0)

  of nnkForStmt:
    result = GpuAst(kind: gpuFor)
    doAssert node[0].kind in {nnkIdent, nnkSym}, "The variable in the for loop is not an identifier or symbol, but: " & $node[0].treerepr
    result.fVar = ctx.toCrucibleAst(reg, node[0])
    if result.fVar.isNil or result.fVar.symbol.isNil:
      result.fVar.symbol = newSymbol($node[0].repr)
    result.fVar.symbol.symKind = gsLocal
    result.fVar.symbol.typ = initGpuType(gtInt32) ## XXX: do not force this type
    if ctx.currentScope != nil:
      scopeAdd(ctx.currentScopeSyms, result.fVar.symbol.name, result.fVar.symbol)
    result.fRangeKind = rkInclusive
    # fStep must never be nil, the AST serializers reject nil fields
    result.fStep = GpuAst(kind: gpuLit, lValue: "1", lType: initGpuType(gtInt32))
    if node[1].kind == nnkInfix:
      result.fStart = ctx.toCrucibleAst(reg, node[1][1])
      result.fEnd = ctx.toCrucibleAst(reg, node[1][2])
      if node[1][0].repr == "..<":
        result.fRangeKind = rkExclusive
      # `..` stays rkInclusive, backends emit <=
    elif node[1].kind == nnkCall and node[1].len >= 4 and "countup" in node[1][0].repr:
      # countup(a, b, step): the last arg is the step, the generic fallback would read it as the end
      result.fStart = ctx.toCrucibleAst(reg, node[1][1])
      result.fEnd = ctx.toCrucibleAst(reg, node[1][2])
      result.fStep = ctx.toCrucibleAst(reg, node[1][3])
    elif node[1].kind == nnkCall and node[1].len >= 2 and node[1][1].kind == nnkObjConstr:
      let objConstr = node[1][1]
      for i in 1 ..< objConstr.len:
        let field = objConstr[i]
        if field.kind == nnkExprColonExpr:
          let fieldName = field[0].strVal
          if fieldName == "a":
            result.fStart = ctx.toCrucibleAst(reg, field[1])
          elif fieldName == "b":
            # Slice.b is inclusive, the backend uses fRangeKind
            result.fEnd = ctx.toCrucibleAst(reg, field[1])
            result.fRangeKind = rkInclusive
    elif node[1].len >= 2:
      result.fStart = ctx.toCrucibleAst(reg, node[1][1])
      result.fEnd = ctx.toCrucibleAst(reg, node[1][^1])
    else:
      result.fStart = GpuAst(kind: gpuLit, lValue: "0", lType: initGpuType(gtInt32))
      result.fEnd = GpuAst(kind: gpuLit, lValue: "0", lType: initGpuType(gtInt32))
    result.fBody = ctx.toCrucibleAst(reg, node[^1])

  of nnkWhileStmt:
    result = GpuAst(kind: gpuWhile)
    result.wCond = ctx.toCrucibleAst(reg, node[0]) # the condition
    result.wBody = ctx.toCrucibleAst(reg, node[1])

  of nnkReturnStmt:
    if node[0].kind == nnkAsgn and node[0][0].strVal == "result":
      # skip the result and just get the RHS
      result = GpuAst(kind: gpuReturn,
                      rValue: ctx.toCrucibleAst(reg, node[0][1]))
    else:
      result = GpuAst(kind: gpuReturn,
                      rValue: ctx.toCrucibleAst(reg, node[0]))

  of nnkWhenStmt:
    error "We shouldn't be seeing a `when` statement after the semantic check of the Nim code."

  # ═══════════════════════════════════════════════════════════════════════
  # Calls, operators, and access
  # ═══════════════════════════════════════════════════════════════════════

  of nnkCall, nnkCommand:
    let name = ctx.getFnName(reg, node[0]) # cannot use `strVal`, might be a symchoice
    if node[0].repr in ctx.generics or name notin ctx.allFnTab:
      # registers the instantiation, or pulls in an external proc by impl
      ## XXX: for CUDA backend need to annotate all pulled in procs with `{.device.}`!
      ctx.registerGenericInstOrExternalProc(reg, node, name)

    # compile-time args become gpuDiscard placeholders, the lowering pass
    # erases them with the callee's gtCompileTimeValue params
    var args: seq[GpuAst]
    let fnParams = ctx.getFnParams(name)
    for i in 1 ..< node.len:
      if (i - 1) < fnParams.len and fnParams[i - 1].typ.kind == gtCompileTimeValue:
        args.add GpuAst(kind: gpuDiscard)
      else:
        args.add ctx.toCrucibleAst(reg, node[i])
    if name in ctx.builtinFns and node[0].repr in NimGpuNumericBuiltinsOperators:
      var op = GpuAst(kind: gpuIdent, symbol: newSymbol(NimGpuNumericBuiltinsOperators[node[0].repr]))
      op.symbol.iSym = op.symbol.name
      result = GpuAst(kind: gpuBinOp, bOp: op, bLeft: args[0], bRight: args[1], bIsOverloaded: false)
      result.bType = resolveType(reg, node.getTypeInst())
    elif name in ctx.builtinFns and node[0].repr in NimGpuBooleanOperators:
      var op = GpuAst(kind: gpuIdent, symbol: newSymbol(NimGpuBooleanOperators[node[0].repr]))
      op.symbol.iSym = op.symbol.name
      result = GpuAst(kind: gpuBinOp, bOp: op, bLeft: args[0], bRight: args[1], bIsOverloaded: false)
      result.bType = resolveType(reg, node.getTypeInst())
    else:
      result = GpuAst(kind: gpuCall)
      result.cName = name
      result.cArgs = args

  of nnkInfix:
    # resolveOverloadedOperators converts gpuBinOp to gpuCall for non-primitive types
    result = GpuAst(kind: gpuBinOp,
                    bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("")),
                    bLeft: ctx.toCrucibleAst(reg, node[1]),
                    bRight: ctx.toCrucibleAst(reg, node[2]))
    # generic instantiations still register here, Nim AST is available
    let typ = node[0].getTypeImpl()
    doAssert typ.kind == nnkProcTy, "Infix node is not a proc but: " & $typ.treerepr
    let leftTyp = resolveType(reg, typ[0][1])
    let rightTyp = resolveType(reg, typ[0][2])
    proc ofBasicType(t: GpuType, allowPtrLhs: bool): bool =
      result = (t.kind in gtBool .. gtSize_t)
      if allowPtrLhs:
        result = result or ((t.kind == gtPtr) and t.implicit and t.to.kind in gtBool .. gtSize_t)
    if not leftTyp.ofBasicType(true) or not rightTyp.ofBasicType(false):
      result.bIsOverloaded = true
      let name = ctx.getFnName(reg, node[0])
      result.bOp.symbol = newSymbol(name.symbol.name, symKind = gsProc)
      result.bOp.symbol.iSym = name.symbol.iSym
      if node[0].repr in ctx.generics or name notin ctx.allFnTab:
        ctx.registerGenericInstOrExternalProc(reg, node, name)
    else:
      result.bIsOverloaded = false
      let isBoolean = leftTyp.kind == gtBool
      let tbl = if isBoolean: NimGpuBooleanOperators else: NimGpuNumericBuiltinsOperators
      let mappedOp = tbl.getOrDefault(node[0].repr, node[0].repr)
      proc isCompoundAssign(op: string): bool =
        ## Compound assignment like `+=` or `*=`. Exclude comparison op
        op.len >= 2 and op[^1] == '=' and op[^2] notin {'=', '<', '>', '!'}
      if mappedOp.isCompoundAssign():
        result = GpuAst(kind: gpuAssign,
                        aLeft: result.bLeft,
                        aRight: result.bRight,
                        aOp: mappedOp)

        # the compound assignment template takes the target by `var T`:
        # complex lvalue arguments are address-wrapped
        if result.aLeft.kind == gpuAddr:
          result.aLeft = result.aLeft.aOf
        # a `var T` scalar parameter materializes as a pointer.
        # The target derefs like the plain-assignment path (a bare ident is pointer arithmetic)
        if result.aLeft.kind == gpuIdent and not result.aLeft.symbol.isNil and
            not result.aLeft.symbol.typ.isNil and result.aLeft.symbol.typ.kind == gtPtr:
          result.aLeft = GpuAst(kind: gpuDeref, dOf: result.aLeft)
        if result.aLeft.kind == gpuLit:
          result.aLeft.lType = leftTyp
        elif result.aRight.kind == gpuLit:
          result.aRight.lType = rightTyp
      else:
        result.bOp.symbol = newSymbol(mappedOp)
        result.bOp.symbol.iSym = result.bOp.symbol.name
        if result.bLeft.kind == gpuLit:
          result.bLeft.lType = leftTyp
        elif result.bRight.kind == gpuLit:
          result.bRight.lType = rightTyp
        result.bType = resolveType(reg, typ[0][0])

  of nnkDotExpr:
    # typed macro: only real field accesses arrive here, method-call syntax does not
    if node[0].kind == nnkObjConstr:
      # field access on a static VALUE Nim inlined into the body, translate the literal
      for i in 1 ..< node[0].len:
        let item = node[0][i]
        if item.kind == nnkExprColonExpr and item[0].strVal == node[1].strVal:
          return ctx.toCrucibleAst(reg, item[1])
      error("Static value has no field '" & node[1].strVal & "': " & node[0].repr, node)
    result = GpuAst(kind: gpuDot)
    result.dParent = ctx.toCrucibleAst(reg, node[0])
    result.dField = ctx.toCrucibleAst(reg, node[1])
    # carries the field's array type for normalizeArraySpanBody
    # (in-tree consumer: ceramic's tensor `data` field)
    if result.dField.kind == gpuIdent and
       node[1].getTypeInst().typeKind == ntyArray:
      result.dField.symbol.typ = resolveType(reg, node[1].getTypeInst())

  of nnkCheckedFieldExpr:
    # the discriminant check has no C form, a runtime case object would silently lose memory safety
    error("nnkCheckedFieldExpr: case-object field access is not supported on GPU — " &
          "the discriminant check cannot be emitted; only compile-time " &
          "(const/static) case objects are supported", node)

  of nnkBracketExpr:
    case node[0].typeKind
    of ntyTuple:
      let typ = resolveType(reg, node[0].getTypeImpl)
      doAssert node[1].kind == nnkIntLit
      let idx = node[1].intVal
      let field = typ.fields[idx].name
      result = GpuAst(kind: gpuDot,
                      dParent: ctx.toCrucibleAst(reg, node[0]),
                      dField: ctx.toCrucibleAst(reg, ident(field)))
    else:
      result = GpuAst(kind: gpuIndex)
      result.iArr = ctx.toCrucibleAst(reg, node[0])
      result.iIndex = ctx.toCrucibleAst(reg, node[1])

  of nnkIdent, nnkOpenSymChoice:
    result = newGpuIdent()
    result.symbol.name = node.repr # for sym choices
    if result.symbol.name == "_":
      result.symbol.name = "underscore"

  of nnkSym:
    # Sanitize the identifier name: backticks are used by Nim for gensym
    # symbols (e.g., ``field`gensym229``) but are invalid in C.
    let sanitized = node.repr.multiReplace(("`", "_"))
    var s = sanitized & "_" & node.signatureHash()
    if node.symKind in {nskProc, nskFunc} and isGenericProc(node):
      s &= "_" & node.symBodyHash()
    # sigTab shares symbol kind and type across reference sites
    # (the kind is set by the caller of toCrucibleAst)
    if symKind(node) == nskConst:
      let impl = getImpl(node)
      let nameIdent = if impl[0].kind == nnkPragmaExpr: impl[0][0] else: impl[0]
      if node.lineinfo != nameIdent.lineinfo:
        return ctx.toCrucibleAst(reg, impl[2])
    if s notin ctx.sigTab:
      result = newGpuIdent()
      result.symbol.name = sanitized
      result.symbol.iSym = s
      if result.symbol.name == "_":
        result.symbol.name = "underscore"
      elif result.symbol.name.startsWith("tmpTuple_"):
        result.symbol.name = "tmpTuple_" & $ctx.genSymCount
        result.symbol.iSym = result.symbol.name & "_" & node.signatureHash()
        inc ctx.genSymCount
      ctx.sigTab[s] = result
    else:
      result = ctx.sigTab[s]
    if node.hasPragma("builtin"):
      result.symbol.symKind = gsBuiltin
      # resolves the canonical name to its IR kind exactly once, before getFnName clobbers symKind
      result.symbol.coordBuiltin = coordBuiltinKind(result.symbol.name)
      result.symbol.synchroBuiltin = synchroBuiltinKind(result.symbol.name)
      result.symbol.reductionBuiltin = reductionBuiltinKind(result.symbol.name)
      result.symbol.atomicsBuiltin = atomicsBuiltinKind(result.symbol.name)

  # literal types
  of nnkPrefix:
    result = GpuAst(kind: gpuPrefix,
                    pVal: ctx.toCrucibleAst(reg, node[1]))
    result.pOp = NimGpuBooleanOperators.getOrDefault(node[0].strVal, node[0].strVal)

  # ═══════════════════════════════════════════════════════════════════════
  # Literals and constructors
  # ═══════════════════════════════════════════════════════════════════════

  of nnkIntLit, nnkInt32Lit:
    result = GpuAst(kind: gpuLit)
    result.lValue = $node.intVal
    result.lType = initGpuType(gtInt32)
  of nnkUIntLit, nnkUint64Lit:
    result = GpuAst(kind: gpuLit)
    result.lValue = $node.intVal
    result.lType = initGpuType(gtUInt64) ## XXX: base on target platform!
  of nnkUInt16Lit:
    result = GpuAst(kind: gpuLit)
    result.lValue = $node.intVal
    result.lType = initGpuType(gtUInt16) ## XXX: base on target platform!
  of nnkUInt32Lit:
    result = GpuAst(kind: gpuLit)
    result.lValue = $node.intVal
    result.lType = initGpuType(gtUInt32) ## XXX: base on target platform!
  of nnkFloat64Lit, nnkFloatLit:
    result = GpuAst(kind: gpuLit)
    result.lValue = $node.floatVal # no suffix needed for double (C/C++ CUDA)
    result.lType = initGpuType(gtFloat64)
  of nnkFloat32Lit:
    result = GpuAst(kind: gpuLit)
    result.lValue = $node.floatVal
    result.lType = initGpuType(gtFloat32)
  of nnkRStrLit:
    result = GpuAst(kind: gpuLit)
    result.lValue = node.strVal
    result.lType = initGpuType(gtString)
  of nnkStrLit:
    # regular string literals escaped, no quote prefix/suffix added
    result = GpuAst(kind: gpuLit)
    result.lValue = node.strVal.escape("", "")
    result.lType = initGpuType(gtString)
  of nnkNilLit:
    result = GpuAst(kind: gpuLit)
    result.lValue = "NULL"
    result.lType = initGpuVoidPtr()

  of nnkPar:
    if node.len == 1: # just take body
      result = ctx.toCrucibleAst(reg, node[0])
    else:
      error("`nnkPar` with more than one argument currently not supported. Got: " & $node.treerepr)

  of nnkObjConstr:
    let typ = resolveType(reg, node)
    if typ.kind notin {gtObject, gtGenericInst}:
      error "ObjConstr must have an object type: " & $typ
    let flds = typ.fields
    if flds.len == 0:
      # each backend emits language-appropriate init for an empty struct ({} / TypeName())
      result = GpuAst(kind: gpuObjConstr, ocType: typ)
    else:
      result = GpuAst(kind: gpuObjConstr, ocType: typ)
      var ocFields: seq[GpuFieldInit]
      for i in 1 ..< node.len:
        doAssert node[i].kind == nnkExprColonExpr
        ocFields.add GpuFieldInit(name: node[i][0].strVal,
                                  value: ctx.toCrucibleAst(reg, node[i][1]),
                                  typ: GpuType(kind: gtVoid))

      for i in 0 ..< flds.len:
        let idx = findIdx(ocFields, flds[i].name)
        if idx >= 0:
          var f = ocFields[idx]
          # a ptr/span/array field is implicitly generic over its init value's space
          let valueType = ctx.getOperandType(f.value)
          if not valueType.isNil:
            f.typ = flds[i].typ.withAddressSpace(getTypeAccessedSpace(valueType))
          else:
            f.typ = flds[i].typ
          result.ocFields.add f
        else:
          let dfl = GpuAst(kind: gpuLit, lValue: "DEFAULT", lType: GpuType(kind: gtVoid))
          result.ocFields.add GpuFieldInit(name: flds[i].name,
                                           value: dfl,
                                           typ: flds[i].typ)
  of nnkTupleConstr:
    let typ = resolveType(reg, node)
    result = GpuAst(kind: gpuObjConstr, ocType: typ)
    let flds = typ.fields
    var ocFields: seq[GpuFieldInit]
    for i in 0 ..< node.len:
      case node[i].kind
      of nnkExprColonExpr:
        ocFields.add GpuFieldInit(name: node[i][0].strVal,
                                  value: ctx.toCrucibleAst(reg, node[i][1]),
                                  typ: GpuType(kind: gtVoid))
      else:
        ocFields.add GpuFieldInit(name: "F" & $i,
                                  value: ctx.toCrucibleAst(reg, node[i]),
                                  typ: GpuType(kind: gtVoid))

    for i in 0 ..< flds.len:
      let idx = findIdx(ocFields, flds[i].name)
      if idx >= 0:
        var f = ocFields[idx]
        let valueType = ctx.getOperandType(f.value)
        if not valueType.isNil:
          f.typ = flds[i].typ.withAddressSpace(getTypeAccessedSpace(valueType))
        else:
          f.typ = flds[i].typ
        result.ocFields.add f
      else:
        let dfl = GpuAst(kind: gpuLit, lValue: "DEFAULT", lType: GpuType(kind: gtVoid))
        result.ocFields.add GpuFieldInit(name: flds[i].name,
                                         value: dfl,
                                         typ: flds[i].typ)


  of nnkBracket:
    let aLitTyp = resolveType(reg, node[0])
    var aValues = newSeq[GpuAst]()
    for el in node:
      aValues.add ctx.toCrucibleAst(reg, el)
    result = GpuAst(kind: gpuArrayLit,
                    aValues: aValues,
                    aLitType: aLitTyp)

  # ═══════════════════════════════════════════════════════════════════════
  # Conversions and address ops
  # ═══════════════════════════════════════════════════════════════════════

  of nnkHiddenStdConv, nnkHiddenSubConv:
    doAssert node[0].kind == nnkEmpty
    result = ctx.toCrucibleAst(reg, node[1])
  of nnkConv:
    let convExpr = ctx.toCrucibleAst(reg, node[1])
    var convTo = resolveType(reg, node[0])
    if convTo.kind in {gtPtr, gtSpan}:
      # the target keeps its operand's space
      let space = ctx.resolveAddressSpace(convExpr)
      convTo = convTo.withAddressSpace(space)
    result = GpuAst(kind: gpuConv, convTo: convTo, convExpr: convExpr)
  of nnkCast:
    let castExpr = ctx.toCrucibleAst(reg, node[1])
    var castTo = resolveType(reg, node[0])
    if castTo.kind in {gtPtr, gtSpan}:
      let space = ctx.resolveAddressSpace(castExpr)
      castTo = castTo.withAddressSpace(space)
    result = GpuAst(kind: gpuCast, cTo: castTo, cExpr: castExpr)

  of nnkAddr, nnkHiddenAddr:
    # HiddenAddr appears for accesses to `var` passed arguments
    result = GpuAst(kind: gpuAddr, aOf: ctx.toCrucibleAst(reg, node[0]))

  of nnkDerefExpr, nnkHiddenDeref:
    # hidden and regular deref treated the same, some backends strip derefs inside gpuIndex
    result = GpuAst(kind: gpuDeref, dOf: ctx.toCrucibleAst(reg, node[0]))

  # ═══════════════════════════════════════════════════════════════════════
  # Pragmas and asm
  # ═══════════════════════════════════════════════════════════════════════

  of nnkAsmStmt:
    doAssert node[0].kind == nnkEmpty
    result = GpuAst(kind: gpuInlineAsm)
    var s = ""
    var ops = newSeq[GpuAst]()
    for i in 1 ..< node.len:
      case node[i].kind
      of nnkStrLit:
        s.add node[i].strVal
      of nnkSym:
        # Nim resolves backtick identifiers to Syms, the printer substitutes
        # the mangled name at the TAG_IDENT_INLINE_ASM placeholder
        ops.add ctx.toCrucibleAst(reg, node[i])
        s.add TAG_IDENT_INLINE_ASM & $(ops.len - 1)
      else:
        doAssert false, "unexpected asm node: " & $node[i].kind & " in " & node.repr
    result.stmt = s
    result.ops = ops

  of nnkPragmaExpr:
    # {.genSym.}, {.inject.}, ... have no meaning for the emitted source
    result = ctx.toCrucibleAst(reg, node[0])
  of nnkBindStmt, nnkMixinStmt:
    # compile-time template-resolution directives
    result = GpuAst(kind: gpuDiscard)
  of nnkPragma:
    # statement-level pragmas are compile-time annotations, except {.emit: ... .}
    # which injects raw target source (gpuEmit), arg at index 1
    var emitArgs: seq[NimNode] = @[]
    for child in node:
      if child.kind in {nnkExprColonExpr, nnkCall} and
         child.len >= 2 and
         child[0].kind in {nnkIdent, nnkSym} and
         child[0].strVal == "emit":
        emitArgs.add child[1]
    if emitArgs.len == 0:
      result = GpuAst(kind: gpuComment, comment: node.repr.strip())
    elif emitArgs.len == 1:
      result = ctx.buildEmitStmt(reg, emitArgs[0])
    else:
      result = GpuAst(kind: gpuBlock)
      for emitArg in emitArgs:
        result.statements.add ctx.buildEmitStmt(reg, emitArg)
  else:
    error "Unhandled node kind in toCrucibleAst: " & $node.treerepr
    result = GpuAst(kind: gpuBlock)
