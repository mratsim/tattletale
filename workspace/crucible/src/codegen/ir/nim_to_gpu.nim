# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, strutils, sequtils, options, tables, sets]

import ./gpu_types
import ./gpu_type_constructors
import ../builtins/nim_builtins
import ./resolvers
import ../passes/pass_registry


proc toGpuAst*(ctx: var GpuContext, reg: var TypeRegistry, node: NimNode): GpuAst

proc isTypeDescNode(n: NimNode): bool =
  ## True when the node is a TYPE used as a value: a generic type parameter
  ## or a `typedesc` literal (e.g. `T` in `make_tensor(T, L)`). In the typed
  ## AST such an argument is a type symbol whose type is `typedesc[T]`
  ## (`typeKind == ntyTypeDesc`). CUDA has no type values, so these cannot be
  ## lowered to a runtime value and are erased at gpuCall construction; the
  ## matching `typedesc` param is dropped in `parseProcParameters`.
  ## A genuine value symbol (var/let/param/result/const of a value type)
  ## never has `ntyTypeDesc` type, so it is never erased (INV-C3).
  ## If the type cannot be determined, treat as a value (do not erase):
  ## the node keeps its previous behavior rather than being silently dropped.
  try:
    let typ = n.getTypeInst()
    result = typ.typeKind == ntyTypeDesc
  except CatchableError:
    result = false

proc parseProcParameters(ctx: var GpuContext, reg: var TypeRegistry, params: NimNode, attrs: set[GpuAttribute]): seq[GpuParam] =
  ## Returns all parameters of the given procedure from the `params` node
  ## of type `nnkFormalParams`.
  ## `typedesc`/type-param params (`_: typedesc[T]`) are dropped: they carry
  ## no runtime value in the emitted GPU source and the caller side erases the
  ## matching argument (see the nnkCall handler), keeping call/callee arity
  ## consistent.
  doAssert params.kind == nnkFormalParams, "Argument is not FormalParams, but: " & $params.treerepr
  for i in 1 ..< params.len:
    let param = params[i]
    let numParams = param.len - 2 # 3 if one param, one more for each of same type, example:
    let typIdx = param.len - 2 # second to last is the type
    # IdentDefs
    #   Ident "x"
    #   Ident "y"
    #   Ident "res"
    #   PtrTy
    #     Ident "float32"   # `param.len - 2`
    #   Empty               # `param.len - 1`
    if isTypeDescNode(param[typIdx-1]):
      continue # typedesc[T] param — no CUDA value; skip whole IdentDefs (multi-name too)
    let paramType = resolveType(reg, param[typIdx-1].getTypeInst())
    for i in 0 ..< numParams:
      var p = ctx.toGpuAst(reg, param[i])
      let symKind = if attGlobal in attrs: gsGlobalKernelParam
                    else: gsDeviceKernelParam
      p.symbol.typ = paramType     ## Update the type of the symbol
      p.symbol.symKind = symKind ## and the symbol kind
      let byref = isLargeStruct(paramType)
      let param = GpuParam(ident: p, typ: paramType, passByRef: byref)
      result.add(param)

proc toInstantiatedProcSignature(ctx: var GpuContext, reg: var TypeRegistry,
    params: NimNode, attrs: set[GpuAttribute]): GpuProcSignature =
  ## Creates a `GpuProcSignature` from the given `params` node of type `nnkFormalParams`
  ##
  ## NOTE: This procedure is only called from generically instantiated procs. Therefore,
  ## we shouldn't need to worry about getting `gtInvalid` return types here.
  GpuProcSignature(
    params: ctx.parseProcParameters(reg, params, attrs),
    retType: resolveProcReturnType(reg, params)
  )

proc getFnName(ctx: var GpuContext, reg: var TypeRegistry, n: NimNode): GpuAst =
  ## Returns the name for the function. Either the symbol name _or_
  ## the `{.cudaName.}` pragma argument.
  template toAst(fn): untyped = GpuAst(kind: gpuIdent, symbol: newSymbol(fn, symKind = gsProc))
  # check if the implementation has a pragma

  if n.kind == nnkSym:
    let sig = n.repr & "_" & n.signatureHash()
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
            result = ctx.toGpuAst(reg, n)
        else:
          result = ctx.toGpuAst(reg, n)
      else:
        result = ctx.toGpuAst(reg, n)
      # Patch operator names in the symbol name AND iSym so that iSym generation uses
      # the patched name. E.g. `+` -> `add` to produce valid C++ identifier `add___hash`.
      # This must happen BEFORE registerGenericInstOrExternalProc overwrites name with iSym.
      if result.symbol.name in ["+", "-", "*", "/"]:
        let oldName = result.symbol.name
        case result.symbol.name
        of "+": result.symbol.name = "add"
        of "-": result.symbol.name = "sub"
        of "*": result.symbol.name = "mul"
        of "/": result.symbol.name = "div"
        else: discard
        result.symbol.iSym = result.symbol.iSym.replace(oldName, result.symbol.name)
      # handle overloads with different signatures
      if n.strVal in ctx.symChoices:
        let id = ctx.sigTab[sig]
        id.symbol.name = id.symbol.iSym
      else:
        ctx.symChoices.incl result.symbol.name
  else:
    result = toAst n.repr
  result.symbol.symKind = gsProc # make sure it's a proc
proc addToFnTable(ctx: var GpuContext, ident: GpuAst, body: GpuAst, kind: set[FunctionKind]) =
  ## Add an entry to the unified function table.
  let key = ident.symbol.iSym
  # Always add/replace — last writer wins (consistent with old behavior)
  ctx.fnTable[key] = FnTableEntry(
    ident: ident,
    body: body,
    kind: kind,
    namePolicy: npUnassigned
  )

proc registerGenericInstOrExternalProc(ctx: var GpuContext, reg: var TypeRegistry, node: NimNode, name: GpuAst) =
  ## Looks up the implementation of the given function and stores it in our table
  ## of generic instantiations.
  ##
  ## For any looked up procedure, we attach the `{.device.}` pragma.
  ##
  ## Mutates the `name` of the given function to match its generic name.

  # Check if the implementation is a template — templates must be fully
  # expanded by Nim before reaching crucible.
  let rawImpl = node[0].getImpl()
  if rawImpl.kind in [nnkTemplateDef, nnkMacroDef]:
    error("Unresolved " & $rawImpl.kind & " encountered in GPU code: " & node[0].repr &
          ". Template/macro expansion must complete before the cuda: block.")

  let inst = rawImpl
  let sig = node[0].getTypeInst()
  inst.params = sig.params # copy over the parameters

  # turn the signature into a `GpuProcSignature`
  let attrs = collectProcAttributes(inst.pragma)
  let procSig = ctx.toInstantiatedProcSignature(reg, sig.params, attrs)
  if name in ctx.processedProcs:
    return
  else:
    # Need to add isym here so that if we have recursive calls, we don't end up
    # calling `toGpuAst` recursively forever
    ctx.processedProcs[name] = procSig

  # Ambiguous builtins (system.min/max/abs) have only `{.inline.}` in getImpl(),
  # not `{.magic.}`. Parse their bodies would crash on the if-expr assertion.
  if node[0].repr in NimGpuAmbiguousBuiltins:
    let retType = resolveType(reg, sig.params[0])
    var builtinFn = GpuAst(kind: gpuProc, pName: name, pRetType: retType, pAttributes: {attDevice})
    ctx.builtins[name] = builtinFn
    return

  # Operator builtins (system.* with magic: MulI etc.) — detect by name
  # and register as builtin before reaching toGpuAst (which can't translate
  # magic bodies and triggers a false isBuiltIn() assertion).
  if node[0].repr in NimGpuNumericOperators or
     node[0].repr in NimGpuBooleanOperators:
    # Only for magic/builtin operators — skip user-defined overloads
    if inst.hasMagicPragma:
      let retType = resolveType(reg, sig.params[0])
      var builtinFn = GpuAst(kind: gpuProc, pName: name, pRetType: retType, pAttributes: {attDevice})
      ctx.builtins[name] = builtinFn
      return

  # Function-style magic builtins (toOpenArray, etc.)
  # Named in NimGpuFnBuiltins — register without parsing bodies.
  if node[0].repr in NimGpuFnBuiltins:
    let retType = resolveType(reg, sig.params[0])
    var builtinFn = GpuAst(kind: gpuProc, pName: name, pRetType: retType, pAttributes: {attDevice})
    ctx.builtins[name] = builtinFn
    ctx.addToFnTable(name, builtinFn, {fkBuiltin})
    return

  let fn = ctx.toGpuAst(reg, inst)
  if fn.kind == gpuDiscard:
    doAssert inst.isBuiltIn()
    return
  fn.pAttributes.incl attDevice # make sure this is interpreted as a device function
  doAssert fn.pName.symbol.iSym == name.symbol.iSym, "Not matching"
  # now overwrite the identifier's `iName` field by its `iSym` so that different
  # generic insts have different
  fn.pName.symbol.name = fn.pName.symbol.iSym
  name.symbol.name = fn.pName.symbol.iSym ## update the name of the called function
  ctx.genericInsts[fn.pName] = fn

proc isExpression(n: GpuAst): bool =
  ## Returns whether the given AST node is an expression
  case n.kind
  of gpuCall: # only if it returns something!
    result = n.cIsExpr
  of gpuBinOp, gpuIdent, gpuLit, gpuArrayLit, gpuPrefix, gpuDot, gpuIndex, gpuObjConstr,
     gpuAddr, gpuDeref, gpuConv, gpuCast, gpuConstexpr:
    result = true
  else:
    result = false

proc fnReturnsValue(ctx: GpuContext, fn: GpuAst): bool =
  ## Returns true if the given `fn` (gpuIdent) returns a value.
  ## The function can either be:
  ## - in the new fnTable
  ## - an inbuilt function
  ## - a generic instantiation
  ## - contained in `allFnTab`
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
  elif fn in ctx.builtins:
    result = ctx.builtins[fn].pRetType.kind != gtVoid
  elif fn in ctx.processedProcs:
    result = ctx.processedProcs[fn].retType.kind != gtVoid
  else:
    error "The function: " & $fn & " is not known anywhere."
proc toGpuAst*(ctx: var GpuContext, reg: var TypeRegistry, node: NimNode): GpuAst =
  ## XXX: things still left to do:
  ## - support `result` variable? Currently not supported. Maybe we will won't

  #echo node.treerepr
  case node.kind
  of nnkEmpty: result = GpuAst(kind: gpuDiscard) # nothing to do
  of nnkStmtList:
    result = GpuAst(kind: gpuBlock)
    let prevScope = ctx.currentScope
    ctx.scopeSymsStack.add(ctx.currentScopeSyms)
    ctx.currentScopeSyms = @[]
    ctx.currentScope = result
    for el in node:
      result.statements.add ctx.toGpuAst(reg, el)
    ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
    ctx.currentScope = prevScope
  of nnkBlockStmt:
    # BlockStmt
    #   Sym "unrolledIter_i0"  <- ignore the block label for now!
    #   Call
    #     Sym "printf"
    #     StrLit "i = %u\n"
    #     IntLit 0
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
      result.statements.add ctx.toGpuAst(reg, node[i])
    ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
    ctx.currentScope = prevScopeBlk
  of nnkBlockExpr:
    ## XXX: For CUDA just a block?
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
        result.statements.add ctx.toGpuAst(reg, el)
    ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
    ctx.currentScope = prevScopeBlkExpr
    # Capture type available via Nim AST for blitting (C2): type is on node[^1]
    # Blitting's getExprType will use this via context scope lookups
  of nnkStmtListExpr: # for statements that return a value.
    ## XXX: For CUDA just a block?
    result = GpuAst(kind: gpuBlock, isExpr: true,
                    )
    let prevScopeStmtListExpr = ctx.currentScope
    ctx.scopeSymsStack.add(ctx.currentScopeSyms)
    ctx.currentScopeSyms = @[]
    ctx.currentScope = result
    for el in node:
      if el.kind != nnkEmpty:
        result.statements.add ctx.toGpuAst(reg, el)
    ctx.currentScopeSyms = ctx.scopeSymsStack.pop()
    ctx.currentScope = prevScopeStmtListExpr
    # Capture type available via Nim AST for blitting (C2): type is on node[^1]
    # Blitting's getExprType will use this via context scope lookups
  of nnkDiscardStmt:
    # just process the child node if any
    result = ctx.toGpuAst(reg, node[0])

  of nnkProcDef, nnkFuncDef:
    # if it is a _generic_ function, we don't actually process it here. instead we add it to
    # the `generics` set. When we encounter a `gpuCall` we will then check if the function
    # being called is part of the generic set and look up its _instantiated_ implementation
    # to parse it. The parsed generics are stored in the `genericInsts` table.
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
        if result.pAttributes.len == 0: # means `nimonly` was applied / is a `builtin`
          ctx.builtins[name] = result # store in builtins, so that we know if it returns a value when called
          ctx.addToFnTable(name, result, {fkBuiltin})
          return GpuAst(kind: gpuDiscard)
      # Process parameters
      result.pParams = ctx.parseProcParameters(reg, params, result.pAttributes)
      result.pBody = ctx.toGpuAst(reg, node.body)
      # Validation and transform passes run via ctx.runPasses()
      # Add to table of known functions (both old and new)
      if result.pName notin ctx.allFnTab:
        ctx.allFnTab[result.pName] = result
        ctx.addToFnTable(name, result, {fkDefined})

  of nnkLetSection, nnkVarSection:
    # For a section with multiple declarations, create a block
    result = GpuAst(kind: gpuBlock)
    for declaration in node:
      # Each declaration gets converted to a gpuVar
      var varNode = GpuAst(kind: gpuVar)
      case declaration[0].kind
      of nnkIdent, nnkSym:
        # IdentDefs               # declaration
        #   Sym "res"             # declaration[0]
        #   Sym "uint32"
        #   Empty
        varNode.vName = ctx.toGpuAst(reg, declaration[0])
      of nnkPragmaExpr:
        # IdentDefs               # declaration
        #   PragmaExpr            # declaration[0]
        #     Sym "res"           # declaration[0][0]
        #     Pragma              # declaration[0][1]
        #       Ident "volatile"
        #   Sym "uint32"
        #   Empty
        varNode.vName = ctx.toGpuAst(reg, declaration[0][0])
        doAssert declaration[0][1].kind == nnkPragma
        varNode.vAttributes = collectAttributes(declaration[0][1])
      else: error "Unexpected node kind for variable: " & $declaration.treeRepr
      varNode.vType = resolveType(reg, declaration)
      varNode.vName.symbol.typ = varNode.vType # also store the type in the symbol, for easier lookup later
      # This is a *local* variable (i.e. `function` address space on WGSL) unless it is
      # annotated with `{.shared.}` (-> `workspace` in WGSL)
      varNode.vName.symbol.symKind = if atvShared in varNode.vAttributes: gsShared
                                 elif atvPrivate in varNode.vAttributes: gsPrivate
                                 else: gsLocal
      varNode.vMutable = node.kind == nnkVarSection
      ## XXX: handle initialization for array types. Need a memcpy!
      ## In principle should be straightforward. Turn e.g.
      ## ```nim
      ## let someData: array[8, uint32] = foo()
      ## let x = BigInt(limbs: someData)
      ## ```
      ## into
      ## ```cuda
      ## unsigned int someData[8] = foo();
      ## BigInt x = {{}};
      ## memcpy((&x.limbs), (&someData), sizeof(unsigned int) * 8);
      ## ```
      ## Or something along those lines.
      if declaration.len > 2 and declaration[2].kind != nnkEmpty:  # Has initialization
        varNode.vInit = ctx.toGpuAst(reg, declaration[2])
        varNode.vRequiresMemcpy = requiresMemcpy(declaration[2])
        result.statements.add(varNode)
      else:
        varNode.vInit = GpuAst(kind: gpuDiscard)
        result.statements.add(varNode)
      # Register variable in current scope's symbol table
      if ctx.currentScope != nil:
        scopeAdd(ctx.currentScopeSyms, varNode.vName.symbol.name, varNode.vName.symbol)

  of nnkAsgn:
    result = GpuAst(kind: gpuAssign)
    result.aLeft = ctx.toGpuAst(reg, node[0])
    result.aRight = ctx.toGpuAst(reg, node[1])
    result.aRequiresMemcpy = requiresMemcpy(node[1])

  of nnkIfStmt:
    result = GpuAst(kind: gpuIf, ifIsExpr: false)
    let branch = node[0]  # First branch
    result.ifCond = ctx.toGpuAst(reg, branch[0])
    result.ifThen = ctx.toGpuAst(reg, branch[1])
    if node.len > 1 and node[^1].kind == nnkElse:
      result.ifElse = ctx.toGpuAst(reg, node[^1][0])
    else:
      result.ifElse = GpuAst(kind: gpuDiscard)

  of nnkIfExpr:
    # If-expression — produces a conditional value (ternary in C).
    # AST: nnkIfExpr(nnkElifExpr(cond, expr), ..., nnkElseExpr(expr)?)
    #
    # GPU constraint: all branch bodies must be single expressions.
    # Multi-statement branches like `let tmp = a; tmp + b` are rejected.
    # If-expr without else is rejected (no value when false).
    proc requireSimpleBranch(n: NimNode) =
      ## GPU kernels must avoid register pressure from multi-statement branches.
      let ok = n.kind != nnkStmtList or n.len == 1
      doAssert ok, "GPU if-expression branches must be single expressions, " &
        "not blocks with statements (would hurt performance)"
    doAssert node.len >= 1
    doAssert node.len == 1 or node[^1].kind == nnkElseExpr,
      "GPU if-expression must have an else branch"
    # Build a gpuIf(isExpr: true) — the lowerIfExpr pass converts to gpuTernary
    result = GpuAst(kind: gpuIf, ifIsExpr: true)
    # Build nested gpuIf(isExpr: true) for the if-expr chain.
    # The lowerIfExpr pass converts these to gpuTernary nodes.
    proc buildIfExpr(ctx: var GpuContext, reg: var TypeRegistry, branches: NimNode, idx: int): GpuAst =
      let child = branches[idx]
      case child.kind
      of nnkElifExpr:
        requireSimpleBranch(child[1])
        result = GpuAst(kind: gpuIf, ifIsExpr: true)
        result.ifCond = ctx.toGpuAst(reg, child[0])
        result.ifThen = ctx.toGpuAst(reg, child[1])
        if idx + 1 < branches.len:
          result.ifElse = buildIfExpr(ctx, reg, branches, idx + 1)
        else:
          result.ifElse = GpuAst(kind: gpuDiscard)
      of nnkElseExpr:
        requireSimpleBranch(child[0])
        result = ctx.toGpuAst(reg, child[0])
      else:
        error "Unexpected child in nnkIfExpr: " & $child.kind
    result = buildIfExpr(ctx, reg, node, 0)

  of nnkForStmt:
    result = GpuAst(kind: gpuFor)
    doAssert node[0].kind in {nnkIdent, nnkSym}, "The variable in the for loop is not an identifier or symbol, but: " & $node[0].treerepr
    result.fVar = ctx.toGpuAst(reg, node[0])
    if result.fVar.isNil or result.fVar.symbol.isNil:
      result.fVar.symbol = newSymbol($node[0].repr)
    result.fVar.symbol.symKind = gsLocal
    result.fVar.symbol.typ = initGpuType(gtInt32) ## XXX: do not force this type
    # Register loop variable in current scope
    if ctx.currentScope != nil:
      scopeAdd(ctx.currentScopeSyms, result.fVar.symbol.name, result.fVar.symbol)
    # Range expression — Phase 3: use fRangeKind instead of +1 patching
    result.fRangeKind = rkInclusive # default (safe for C-style < loops)
    if node[1].kind == nnkInfix:
      result.fStart = ctx.toGpuAst(reg, node[1][1])
      result.fEnd = ctx.toGpuAst(reg, node[1][2])
      # Set range kind based on operator
      if node[1][0].repr == "..<":
        result.fRangeKind = rkExclusive
      # else `..` stays rkInclusive (default) — backends emit <= or equivalent
    elif node[1].kind == nnkCall and node[1].len >= 2 and node[1][1].kind == nnkObjConstr:
      let objConstr = node[1][1]
      for i in 1 ..< objConstr.len:
        let field = objConstr[i]
        if field.kind == nnkExprColonExpr:
          let fieldName = field[0].strVal
          if fieldName == "a":
            result.fStart = ctx.toGpuAst(reg, field[1])
          elif fieldName == "b":
            # Slice.b is inclusive — store literal as-is, backend uses fRangeKind
            result.fEnd = ctx.toGpuAst(reg, field[1])
            result.fRangeKind = rkInclusive
    elif node[1].len >= 2:
      result.fStart = ctx.toGpuAst(reg, node[1][1])
      result.fEnd = ctx.toGpuAst(reg, node[1][^1])
    else:
      result.fStart = GpuAst(kind: gpuLit, lValue: "0", lType: initGpuType(gtInt32))
      result.fEnd = GpuAst(kind: gpuLit, lValue: "0", lType: initGpuType(gtInt32))
    result.fBody = ctx.toGpuAst(reg, node[^1])
  of nnkWhileStmt:
    result = GpuAst(kind: gpuWhile)
    result.wCond = ctx.toGpuAst(reg, node[0]) # the condition
    result.wBody = ctx.toGpuAst(reg, node[1])

  of nnkTemplateDef, nnkMacroDef:
    ## NOTE: Currently we process templates, but we expect them to be already
    ## expanded by the Nim compiler. Thus we could in theory expand them manually
    ## but fortunately we don't need to.
    return GpuAst(kind: gpuDiscard)

  of nnkCall, nnkCommand:
    # `name` below is name + signature hash. Check if this is a generic based on node repr
    let name = ctx.getFnName(reg, node[0]) # cannot use `strVal`, might be a symchoice
    if node[0].repr in ctx.generics or name notin ctx.allFnTab:
      # process the generic instantiaton and store *or* pull in a proc defined outside
      # the `cuda` macro by its implementation.
      ## XXX: for CUDA backend need to annotate all pulled in procs with `{.device.}`!
      ctx.registerGenericInstOrExternalProc(reg, node, name)

    # Erase typedesc-typed arguments (types passed as values, e.g. `T` in
    # make_tensor(T, L)): they have no CUDA value representation and are never
    # converted. The callee's matching typedesc param is dropped in
    # parseProcParameters, keeping call/callee arity consistent. Non-typedesc
    # args keep their exact position/order. Operator builtins never receive
    # typedesc operands in valid Nim, so this is safe for those branches too.
    let args = node[1..^1].filterIt(not isTypeDescNode(it)).mapIt(ctx.toGpuAst(reg, it))
    if name in ctx.builtins and node[0].repr in NimGpuNumericOperators:
      var op = GpuAst(kind: gpuIdent, symbol: newSymbol(NimGpuNumericOperators[node[0].repr]))
      op.symbol.iSym = op.symbol.name
      result = GpuAst(kind: gpuBinOp, bOp: op, bLeft: args[0], bRight: args[1], bIsOverloaded: false)
      result.bType = resolveType(reg, node.getTypeInst())
    elif name in ctx.builtins and node[0].repr in NimGpuBooleanOperators:
      var op = GpuAst(kind: gpuIdent, symbol: newSymbol(NimGpuBooleanOperators[node[0].repr]))
      op.symbol.iSym = op.symbol.name
      result = GpuAst(kind: gpuBinOp, bOp: op, bLeft: args[0], bRight: args[1], bIsOverloaded: false)
      result.bType = resolveType(reg, node.getTypeInst())
    else:
      let fnIsExpr = ctx.fnReturnsValue(name)
      result = GpuAst(kind: gpuCall, cIsExpr: fnIsExpr)
      result.cName = name
      result.cArgs = args

  of nnkInfix:
    # Always emit gpuBinOp — resolveOverloadedOperators pass converts
    # to gpuCall for non-primitive types.
    result = GpuAst(kind: gpuBinOp,
                    bOp: GpuAst(kind: gpuIdent, symbol: newSymbol("")),
                    bLeft: ctx.toGpuAst(reg, node[1]),
                    bRight: ctx.toGpuAst(reg, node[2]))
    # Still register generic instantiations (needs Nim AST access at toGpuAst time)
    let typ = node[0].getTypeImpl()
    doAssert typ.kind == nnkProcTy, "Infix node is not a proc but: " & $typ.treerepr
    let leftTyp = resolveType(reg, typ[0][1])
    let rightTyp = resolveType(reg, typ[0][2])
    proc ofBasicType(t: GpuType, allowPtrLhs: bool): bool =
      result = (t.kind in gtBool .. gtSize_t)
      if allowPtrLhs:
        result = result or ((t.kind == gtPtr) and t.implicit and t.to.kind in gtBool .. gtSize_t)
    if not leftTyp.ofBasicType(true) or not rightTyp.ofBasicType(false):
      # Non-primitive types — register the operator as a function
      result.bIsOverloaded = true
      let name = ctx.getFnName(reg, node[0])
      result.bOp.symbol = newSymbol(name.symbol.name, symKind = gsProc)
      result.bOp.symbol.iSym = name.symbol.iSym
      if node[0].repr in ctx.generics or name notin ctx.allFnTab:
        ctx.registerGenericInstOrExternalProc(reg, node, name)
    else:
      # Primitive types: map operator name for C/C++ compatibility
      result.bIsOverloaded = false
      let isBoolean = leftTyp.kind == gtBool
      let tbl = if isBoolean: NimGpuBooleanOperators else: NimGpuNumericOperators
      let mappedOp = tbl.getOrDefault(node[0].repr, node[0].repr)
      result.bOp.symbol = newSymbol(mappedOp)
      result.bOp.symbol.iSym = result.bOp.symbol.name
      # Patch literal types
      if result.bLeft.kind == gpuLit:
        result.bLeft.lType = leftTyp
      elif result.bRight.kind == gpuLit:
        result.bRight.lType = rightTyp
      # Carry the operator's true return type: typ = node[0].getTypeImpl() is
      # the ProcTy; typ[0] is the FormalParams; typ[0][0] is the return type
      # node. Compound-assign operators (`+=`, `-=`, ...) declare no return
      # type in their ProcTy (nnkEmpty), so the result type is the LHS operand
      # type (stripping the `var` wrapper Nim puts on var params).
      # Scoped to the primitive branch only — eager resolution on overloaded
      # operands would crash resolveType (resolvers.nim:316/351).
      let retTypNode = typ[0][0]
      if retTypNode.kind == nnkEmpty:
        var lhsTyp = node[1].getTypeInst()
        if lhsTyp.kind == nnkVarTy:
          lhsTyp = lhsTyp[0]
        result.bType = resolveType(reg, lhsTyp)
      else:
        result.bType = resolveType(reg, retTypNode)
  of nnkDotExpr:
    ## NOTE: As we use a typed macro, we only encounter `DotExpr` for *actual* field accesses and NOT
    ## for calls using method call syntax without parens
    result = GpuAst(kind: gpuDot)
    result.dParent = ctx.toGpuAst(reg, node[0])
    result.dField = ctx.toGpuAst(reg, node[1])

  of nnkBracketExpr:
    case node[0].typeKind
    of ntyTuple:
      # need to replace `[idx]` by field access
      let typ = resolveType(reg, node[0].getTypeImpl)
      doAssert node[1].kind == nnkIntLit
      let idx = node[1].intVal
      let field = typ.oFields[idx].name
      result = GpuAst(kind: gpuDot,
                      dParent: ctx.toGpuAst(reg, node[0]),
                      dField: ctx.toGpuAst(reg, ident(field)))
    else:
      result = GpuAst(kind: gpuIndex)
      result.iArr = ctx.toGpuAst(reg, node[0])
      result.iIndex = ctx.toGpuAst(reg, node[1])

  of nnkIdent, nnkOpenSymChoice:
    result = newGpuIdent()
    result.symbol.name = node.repr # for sym choices
    if result.symbol.name == "_":
      result.symbol.name = "underscore"
  of nnkSym:
    # Sanitize the identifier name: backticks are used by Nim for gensym
    # symbols (e.g., ``field`gensym229``) but are invalid in C.
    let sanitized = node.repr.multiReplace(("`", "_"))
    let s = sanitized & "_" & node.signatureHash()
    # NOTE: The reason we have a tab of known symbols is not to keep the same _reference_ to each
    # symbol, but rather to allow having the same symbol kind and appropriate type for each
    # symbol `GpuAst` (of kind `gpuIdent`), which is set in the caller of this call.
    # For example in `nnkCall` nodes returning the value from the table automatically means the
    # `symbolKind` is local / function argument etc.
    # Resolve `const _ = X()` inline: the constant value lives outside the cuda: block
    # so emitting a GPU identifier would reference an undeclared name.
    if sanitized == "_" and symKind(node) == nskConst:
      # getImpl returns nnkConstDef; [2] is the value expression (e.g. X_marker())
      # Using the whole def would recurse since the name child is also `_`.
      return ctx.toGpuAst(reg, getImpl(node)[2])
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

  # literal types
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
    # For regular string literals escape them (but don't prefix/suffix with `"`)
    result = GpuAst(kind: gpuLit)
    result.lValue = node.strVal.escape("", "")
    result.lType = initGpuType(gtString)
  of nnkNilLit:
    result = GpuAst(kind: gpuLit)
    result.lValue = "NULL"
    result.lType = initGpuVoidPtr()

  of nnkPar:
    if node.len == 1: # just take body
      result = ctx.toGpuAst(reg, node[0])
    else:
      error("`nnkPar` with more than one argument currently not supported. Got: " & $node.treerepr)

  of nnkReturnStmt:
    if node[0].kind == nnkAsgn and node[0][0].strVal == "result":
      # skip the result and just get the RHS
      result = GpuAst(kind: gpuReturn,
                      rValue: ctx.toGpuAst(reg, node[0][1]))
    else:
      result = GpuAst(kind: gpuReturn,
                      rValue: ctx.toGpuAst(reg, node[0]))

  of nnkPrefix:
    result = GpuAst(kind: gpuPrefix,
                    pVal: ctx.toGpuAst(reg, node[1]))
    result.pOp = NimGpuBooleanOperators.getOrDefault(node[0].strVal, node[0].strVal)

  of nnkTypeSection:
    result = GpuAst(kind: gpuBlock)
    for el in node: # walk each type def
      doAssert el.kind == nnkTypeDef
      result.statements.add ctx.toGpuAst(reg, el)
  of nnkTypeDef:
    doAssert node.len == 3, "TypeDef node does not have 3 children: " & $node.len
    if node[1].kind == nnkGenericParams: # if this is a generic, only store existence of it
                                         # will store the instantiatons in `nnkObjConstr`
      result = GpuAst(kind: gpuDiscard)
    else:
      let typ = resolveType(reg, node[0])
      # For type aliases resolved to primitives (e.g. type F = type(x.val) → uint32),
      # don't emit a typedef — the target already knows the type.
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
                          aTo: ctx.toGpuAst(reg, node[2]))
        else:
          result = GpuAst(kind: gpuDiscard)
      else:
        error "Unexpected node kind in TypeDef: " & $node[2].kind

      # include this the set of known types to not generate duplicates
      ctx.types[typ] = result
      # Reset the type we return to void. We now generate _all_ types from the
      # `types`.
      result = GpuAst(kind: gpuDiscard)
  of nnkObjConstr:
    ## this should never see `genericParam` I think
    let typ = resolveType(reg, node)
    # get all fields of the type
    let flds = if typ.kind == gtObject: typ.oFields
               elif typ.kind == gtGenericInst: typ.gFields
               else: error "ObjConstr must have an object type: " & $typ
    if flds.len == 0:
      # Empty struct (e.g. Int[N]) — produce empty-ocFields gpuObjConstr
      # so each backend emits language-appropriate init ({} / TypeName())
      result = GpuAst(kind: gpuObjConstr, ocType: typ)
    else:
      result = GpuAst(kind: gpuObjConstr, ocType: typ)
      # find all fields that have been defined by the user
      var ocFields: seq[GpuFieldInit]
      for i in 1 ..< node.len: # all fields to be init'd
        doAssert node[i].kind == nnkExprColonExpr
        ocFields.add GpuFieldInit(name: node[i][0].strVal,
                                  value: ctx.toGpuAst(reg, node[i][1]),
                                  typ: GpuType(kind: gtVoid))

      # now add fields in order of the type declaration
      for i in 0 ..< flds.len:
        let idx = findIdx(ocFields, flds[i].name)
        if idx >= 0:
          var f = ocFields[idx]
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
    # get all fields of the type
    let flds = typ.oFields
    # find all fields that have been defined by the user
    var ocFields: seq[GpuFieldInit]
    for i in 0 ..< node.len: # all fields to be init'd
      case node[i].kind
      of nnkExprColonExpr:
        ocFields.add GpuFieldInit(name: node[i][0].strVal,
                                  value: ctx.toGpuAst(reg, node[i][1]),
                                  typ: GpuType(kind: gtVoid))
      else:
        ocFields.add GpuFieldInit(name: "F" & $i,
                                  value: ctx.toGpuAst(reg, node[i]),
                                  typ: GpuType(kind: gtVoid))

    # now add fields in order of the type declaration
    for i in 0 ..< flds.len:
      let idx = findIdx(ocFields, flds[i].name)
      if idx >= 0:
        var f = ocFields[idx]
        f.typ = flds[i].typ
        result.ocFields.add f
      else:
        let dfl = GpuAst(kind: gpuLit, lValue: "DEFAULT", lType: GpuType(kind: gtVoid))
        result.ocFields.add GpuFieldInit(name: flds[i].name,
                                         value: dfl,
                                         typ: flds[i].typ)


  of nnkAsmStmt:
    doAssert node.len == 2
    doAssert node[0].kind == nnkEmpty
    result = GpuAst(kind: gpuInlineAsm,
                    stmt: node[1].strVal)

  of nnkBracket:
    let aLitTyp = resolveType(reg, node[0])
    var aValues = newSeq[GpuAst]()
    for el in node:
      aValues.add ctx.toGpuAst(reg, el)
    result = GpuAst(kind: gpuArrayLit,
                    aValues: aValues,
                    aLitType: aLitTyp)

  of nnkCommentStmt:
    result = GpuAst(kind: gpuDiscard)

  of nnkHiddenStdConv, nnkHiddenSubConv:
    doAssert node[0].kind == nnkEmpty
    result = ctx.toGpuAst(reg, node[1])
  of nnkConv:
    # maps type conversion, e.g. `let i: int = 5; i.uint32`
    result = GpuAst(kind: gpuConv, convTo: resolveType(reg, node[0]), convExpr: ctx.toGpuAst(reg, node[1]))
  of nnkCast:
    # only maps real bit casts
    result = GpuAst(kind: gpuCast, cTo: resolveType(reg, node[0]), cExpr: ctx.toGpuAst(reg, node[1]))

  of nnkAddr, nnkHiddenAddr:
    # `HiddenAddr` appears for accesses to `var` passed arguments
    result = GpuAst(kind: gpuAddr, aOf: ctx.toGpuAst(reg, node[0]))

  of nnkDerefExpr, nnkHiddenDeref:
    # treat hidden and regular deref the same nowadays. On some backends may strip derefs, if
    # they appear e.g. in an `gpuIndex` (CUDA)
    result = GpuAst(kind: gpuDeref, dOf: ctx.toGpuAst(reg, node[0]))

  of nnkConstDef:
    let identNode = if node[0].kind == nnkPragmaExpr: node[0][0] else: node[0]
    result = GpuAst(kind: gpuConstexpr,
                    cIdent: ctx.toGpuAst(reg, identNode),
                    cValue: ctx.toGpuAst(reg, node[2]),
                    cType: resolveType(reg, node))
    result.cIdent.symbol.typ = result.cType # also store the type in the symbol, for easier lookup later
    result.cIdent.symbol.symKind = gsLocal #if atvShared in result.vAttributes: gsShared
                               #elif atvPrivate in varNode.vAttributes: gsPrivate
                               #else: gsLocal

  of nnkConstSection:
    result = GpuAst(kind: gpuBlock)
    for el in node: # walk each type def
      doAssert el.kind == nnkConstDef
      result.statements.add ctx.toGpuAst(reg, el)

  of nnkPragmaExpr:
    ## {.genSym.}, {.inject.}, etc. — strip pragma, process inner node.
    ## The pragma has no meaning for CUDA/C++ output.
    result = ctx.toGpuAst(reg, node[0])
  of nnkBindStmt, nnkMixinStmt:
    ## Bind/mixin statements are compile-time directives for template resolution.
    ## Irrelevant for GPU codegen — skip them.
    result = GpuAst(kind: gpuDiscard)
  of nnkPragma:
    ## Pragmas are compile-time annotations (e.g. {.warning.} on operators).
    ## They are irrelevant for GPU codegen — skip them.
    result = GpuAst(kind: gpuDiscard)
  of nnkWhenStmt:
    error "We shouldn't be seeing a `when` statement after sem check of the Nim code."
  else:
    echo "Unhandled node kind in toGpuAst: ", node.kind
    error "Unhandled node kind in toGpuAst: " & $node.treerepr
    result = GpuAst(kind: gpuBlock)
