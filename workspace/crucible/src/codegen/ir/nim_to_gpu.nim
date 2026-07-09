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

proc parseProcParameters(ctx: var GpuContext, reg: var TypeRegistry, params: NimNode, attrs: set[GpuAttribute]): seq[GpuParam] =
  ## Returns all parameters of the given procedure from the `params` node
  ## of type `nnkFormalParams`.
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
    let paramType = resolveType(reg, param[typIdx-1].getTypeInst())
    for i in 0 ..< numParams:
      var p = ctx.toGpuAst(reg, param[i])
      let symKind = if attGlobal in attrs: gsGlobalKernelParam
                    else: gsDeviceKernelParam
      p.iTyp = paramType     ## Update the type of the symbol
      p.symbolKind = symKind ## and the symbol kind
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

template findIdx(col, el): untyped =
  var res = -1
  for i, it in col:
    if it.name == el:
      res = i
      break
  res

proc maybePatchFnName(n: var GpuAst) =
  ## Renames operator function names whose symbols are not valid C++
  ## identifier characters (e.g. `+`→`add`, `-`→`sub`).
  ## Nim-textual operators like `div`, `and` pass through since they
  ## are valid identifier starters and get disambiguated by the
  ## signature hash suffix from registerGenericInstOrExternalProc.
  doAssert n.kind == gpuIdent
  template patch(arg, by: untyped): untyped =
    arg.iSym = arg.iSym.replace(arg.iName, by)
    arg.iName = by
  let name = n.iName
  case name
  of "+": patch(n, "add")
  of "-": patch(n, "sub")
  of "*": patch(n, "mul")
  of "/": patch(n, "div")
  else: discard

proc getFnName(ctx: var GpuContext, reg: var TypeRegistry, n: NimNode): GpuAst =
  ## Returns the name for the function. Either the symbol name _or_
  ## the `{.cudaName.}` pragma argument.
  template toAst(fn): untyped = GpuAst(kind: gpuIdent, iName: fn, symbolKind: gsProc)
  # check if the implementation has a pragma

  if n.kind == nnkSym:
    # Check if `cudaName` pragma used:
    # ProcDef
    #   Sym "syncthreads"
    #   Empty
    #   Empty
    #   FormalParams
    #     Empty
    #   Pragma
    #     ExprColonExpr
    #       Sym "cudaName"           <- if this exists
    #       StrLit "__syncthreads"   <- use this name
    #   Empty
    #   DiscardStmt
    #     Empty
    let sig = n.repr & "_" & n.signatureHash()
    if sig in ctx.sigTab:
      result = ctx.sigTab[sig]
    else:
      let impl = n.getImpl
      if impl.kind in [nnkProcDef, nnkFuncDef]:
        let pragma = impl.pragma
        if pragma.kind != nnkEmpty and pragma[0].kind == nnkExprColonExpr:
          if pragma[0][0].kind in [nnkIdent, nnkSym] and pragma[0][0].strVal == "cudaName":
            # want to replace fn name
            result = toAst pragma[0][1].strVal
            ctx.sigTab[sig] = result
          else:
            result = ctx.toGpuAst(reg, n) # if no `cudaName` pragma
        else:
          result = ctx.toGpuAst(reg, n) # if _no_ pragma
      else:
        result = ctx.toGpuAst(reg, n) # if not proc or func

      result.maybePatchFnName()

      # handle overloads with different signatures
      if n.strVal in ctx.symChoices:
        # this is an overload of another function with different signature (not a generic, but
        # overloads are not allowed in CUDA/WGSL/...). Update `sigTab` entry by using `iSym`
        # for `iName` field for unique name
        let id = ctx.sigTab[sig]
        id.iName = id.iSym
      else:
        ctx.symChoices.incl result.iName # store this name in `symChoices`
  else:
    # else we use the str representation (repr for open / closed sym choice nodes)
    result = toAst n.repr
    #error "This fn identifier is not a symbol?! " & $n.repr
    # If it's not a symbol, there is no signature associated
    # ctx.sigTab[sig] = result
  result.symbolKind = gsProc # make sure it's a proc

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
    return

  let fn = ctx.toGpuAst(reg, inst)
  if fn.kind == gpuDiscard:
    echo "[registerGenericInstOrExternalProc] node[0].repr = ", node[0].repr, " impl.kind = ", inst.kind, " isBuiltIn = ", inst.isBuiltIn()
    echo "  impl treerepr: ", inst.treerepr
    doAssert inst.isBuiltIn()
    return
  fn.pAttributes.incl attDevice # make sure this is interpreted as a device function
  doAssert fn.pName.iSym == name.iSym, "Not matching"
  # now overwrite the identifier's `iName` field by its `iSym` so that different
  # generic insts have different
  fn.pName.iName = fn.pName.iSym
  name.iName = fn.pName.iSym ## update the name of the called function
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
  ## - an inbuilt function
  ## - a generic instantiation
  ## - contained in `allFnTab`
  if fn in ctx.allFnTab:
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
    for el in node:
      result.statements.add ctx.toGpuAst(reg, el)
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
                    blockLabel: blockLabel)
    for i in 1 ..< node.len: # index 0 is the block label
      result.statements.add ctx.toGpuAst(reg, node[i])
  of nnkBlockExpr:
    ## XXX: For CUDA just a block?
    let blockLabel = if node[0].kind in {nnkSym, nnkIdent}: node[0].strVal
                     elif node[0].kind == nnkEmpty: ""
                     else: error "Unexpected node in block label field: " & $node.treerepr
    result = GpuAst(kind: gpuBlock, blockLabel: blockLabel, isExpr: true)
    for el in node:
      if el.kind != nnkEmpty:
        result.statements.add ctx.toGpuAst(reg, el)
  of nnkStmtListExpr: # for statements that return a value.
    ## XXX: For CUDA just a block?
    result = GpuAst(kind: gpuBlock, isExpr: true)
    for el in node:
      if el.kind != nnkEmpty:
        result.statements.add ctx.toGpuAst(reg, el)
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
      ctx.generics.incl name.iName # need to use raw name, *not* symbol
      result = GpuAst(kind: gpuDiscard)
    elif node.body.kind == nnkEmpty: # just a forward declaration
      result = GpuAst(kind: gpuDiscard)
    else:
      result = GpuAst(kind: gpuProc)
      result.pName = name
      result.pName.symbolKind = gsProc ## This is a procedure identifier
      let params = node[3]
      doAssert params.kind == nnkFormalParams
      result.pRetType = resolveProcReturnType(reg, params)
      if result.pRetType.kind == gtInvalid:
        ctx.generics.incl name.iName # need to use raw name, *not* symbol
        return GpuAst(kind: gpuDiscard)

      # Process pragmas
      if node.pragma.kind != nnkEmpty:
        doAssert node.pragma.len > 0, "Pragma kind non empty, but no pragma?"
        result.pAttributes = collectProcAttributes(node.pragma)
        if result.pAttributes.len == 0: # means `nimonly` was applied / is a `builtin`
          ctx.builtins[name] = result # store in builtins, so that we know if it returns a value when called
          return GpuAst(kind: gpuDiscard)
      # Process parameters
      result.pParams = ctx.parseProcParameters(reg, params, result.pAttributes)
      result.pBody = ctx.toGpuAst(reg, node.body)
      # Validation and transform passes run via ctx.runPasses()
      # Add to table of known functions
      if result.pName notin ctx.allFnTab:
        ctx.allFnTab[result.pName] = result

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
      varNode.vName.iTyp = varNode.vType # also store the type in the symbol, for easier lookup later
      # This is a *local* variable (i.e. `function` address space on WGSL) unless it is
      # annotated with `{.shared.}` (-> `workspace` in WGSL)
      varNode.vName.symbolKind = if atvShared in varNode.vAttributes: gsShared
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

  of nnkAsgn:
    result = GpuAst(kind: gpuAssign)
    result.aLeft = ctx.toGpuAst(reg, node[0])
    result.aRight = ctx.toGpuAst(reg, node[1])
    result.aRequiresMemcpy = requiresMemcpy(node[1])

  of nnkIfStmt:
    result = GpuAst(kind: gpuIf)
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
    # Helper to build nested ternary
    proc buildTernary(ctx: var GpuContext, reg: var TypeRegistry, branches: NimNode, idx: int): GpuAst =
      let child = branches[idx]
      case child.kind
      of nnkElifExpr:
        requireSimpleBranch(child[1])
        result = GpuAst(kind: gpuTernary)
        result.tCond = ctx.toGpuAst(reg, child[0])
        result.tThen = ctx.toGpuAst(reg, child[1])
        if idx + 1 < branches.len:
          result.tElse = buildTernary(ctx, reg, branches, idx + 1)
        else:
          result.tElse = GpuAst(kind: gpuDiscard)
      of nnkElseExpr:
        requireSimpleBranch(child[0])
        result = ctx.toGpuAst(reg, child[0])
      else:
        error "Unexpected child in nnkIfExpr: " & $child.kind
    result = buildTernary(ctx, reg, node, 0)

  of nnkForStmt:
    result = GpuAst(kind: gpuFor)
    doAssert node[0].kind in {nnkIdent, nnkSym}, "The variable in the for loop is not an identifier or symbol, but: " & $node[0].treerepr
    result.fVar = ctx.toGpuAst(reg, node[0])
    result.fVar.symbolKind = gsLocal
    result.fVar.iTyp = initGpuType(gtInt32) ## XXX: do not force this type
    # Range expression — may be `0 .. N` (Infix inclusive) or `0 ..< N` (Infix exclusive)
    if node[1].kind == nnkInfix:
      result.fStart = ctx.toGpuAst(reg, node[1][1])
      result.fEnd = ctx.toGpuAst(reg, node[1][2])
      # Inclusive `..` — codegen uses `i < end`, but Nim's `..` is
      # inclusive `[a, b]` (b+1 values). Add 1 so C's `<` matches.
      if node[1][0].repr == "..":
        # Derive type from range end expression
        let endTyp =
          if result.fEnd.kind == gpuLit:
            result.fEnd.lType   # literal like `0 ..< 128` — use float/int literal's own type
          elif result.fEnd.kind == gpuIdent and result.fEnd.iTyp.kind notin {gtVoid}:
            result.fEnd.iTyp   # variable like `0 ..< n` — use the variable's declared type
          elif result.fEnd.kind == gpuBinOp:
            initGpuType(gtInt32)  # expression like `0 ..< (a + b)` — default to int32 (implicit integer promotion)
          else:
            initGpuType(gtInt32)  # call/other expression — fallback
        let one = GpuAst(kind: gpuLit, lValue: "1", lType: endTyp)
        var addOp = GpuAst(kind: gpuIdent, iName: "+")
        result.fEnd = GpuAst(kind: gpuBinOp, bOp: addOp,
                             bLeft: result.fEnd, bRight: one)
    elif node[1].kind == nnkCall and node[1].len >= 2 and node[1][1].kind == nnkObjConstr:
      let objConstr = node[1][1]
      for i in 1 ..< objConstr.len:
        let field = objConstr[i]
        if field.kind == nnkExprColonExpr:
          let fieldName = field[0].strVal
          if fieldName == "a":
            result.fStart = ctx.toGpuAst(reg, field[1])
          elif fieldName == "b":
            # Slice.b is inclusive but codegen uses `<`, so increment the
            # Nim AST literal before converting to GpuAst.
            if field[1].kind in {nnkIntLit, nnkInt32Lit, nnkUIntLit}:
              let modified = newLit(int32(field[1].intVal + 1))
              result.fEnd = ctx.toGpuAst(reg, modified)
            else:
              result.fEnd = ctx.toGpuAst(reg, field[1])
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
    let tName = node[0].strVal

    # Extract parameters
    var tParams = newSeq[string]()
    for i in 1 ..< node[3].len:
      let param = node[3][i]
      tParams.add param[0].strVal
    # and the body
    let tBody = ctx.toGpuAst(reg, node.body)

    # Store template in context
    ctx.templates[tName] = TemplateInfo(
      params: tParams,
      body: tBody
    )

    result = GpuAst(kind: gpuDiscard)

  of nnkCall, nnkCommand:
    # `name` below is name + signature hash. Check if this is a generic based on node repr
    let name = ctx.getFnName(reg, node[0]) # cannot use `strVal`, might be a symchoice
    if node[0].repr in ctx.generics or name notin ctx.allFnTab:
      # process the generic instantiaton and store *or* pull in a proc defined outside
      # the `cuda` macro by its implementation.
      ## XXX: for CUDA backend need to annotate all pulled in procs with `{.device.}`!
      ctx.registerGenericInstOrExternalProc(reg, node, name)

    let args = node[1..^1].mapIt(ctx.toGpuAst(reg, it))
    if name in ctx.builtins and node[0].repr in NimGpuNumericOperators:
      var op = GpuAst(kind: gpuIdent, iName: NimGpuNumericOperators[node[0].repr])
      op.iSym = op.iName
      result = GpuAst(kind: gpuBinOp, bOp: op, bLeft: args[0], bRight: args[1])
    elif name in ctx.builtins and node[0].repr in NimGpuBooleanOperators:
      var op = GpuAst(kind: gpuIdent, iName: NimGpuBooleanOperators[node[0].repr])
      op.iSym = op.iName
      result = GpuAst(kind: gpuBinOp, bOp: op, bLeft: args[0], bRight: args[1])
    else:
      let fnIsExpr = ctx.fnReturnsValue(name)
      result = GpuAst(kind: gpuCall, cIsExpr: fnIsExpr)
      result.cName = name
      result.cArgs = args

  of nnkInfix:
    result = GpuAst(kind: gpuBinOp)
    # Using `getType` to get the types of the arguuments
    let typ = node[0].getTypeImpl() # e.g.
    doAssert typ.kind == nnkProcTy, "Infix node is not a proc but: " & $typ.treerepr
    # BracketExpr
    #   Sym "proc"
    #   Sym "int"  <- return type
    #   Sym "int"  <- left op type
    #   Sym "int"  <- right op type
    let leftTyp = resolveType(reg, typ[0][1])
    let rightTyp = resolveType(reg, typ[0][2])
    # if either is not a base type (`gtBool .. gtSize_t`) we actually deal with a _function call_
    # instead of an binary operation. Will thus rewrite.
    proc ofBasicType(t: GpuType, allowPtrLhs: bool): bool =
      ## Determines if the given type is a basic POD type *or* a simple pointer to it.
      ## This is because some infix nodes, e.g. `x += y` will have LHS arguments that are
      ## `var T`, which appear as an implicit pointer here.
      ##
      ## TODO: Handle the case of backend inbuilt special types (like `vec3`), which may indeed
      ## have inbuilt infix operators. Either by checking if the type has a `{.builtin.}` pragma
      ## _or_ if there is a wrapped proc for this operator and if so do not rewrite as `gpuCall`
      ## if that exists.
      result = (t.kind in gtBool .. gtSize_t)
      if allowPtrLhs:
        result = result or ((t.kind == gtPtr) and t.implicit and t.to.kind in gtBool .. gtSize_t)

    if not leftTyp.ofBasicType(true) or not rightTyp.ofBasicType(false):
      let name = ctx.getFnName(reg, node[0])
      result = GpuAst(kind: gpuCall)
      result.cName = name
      result.cArgs = @[ctx.toGpuAst(reg, node[1]), ctx.toGpuAst(reg, node[2])]
      if node[0].repr in ctx.generics or name notin ctx.allFnTab:
        ctx.registerGenericInstOrExternalProc(reg, node, name)
    else:
      # if left/right is boolean we need logical AND/OR, otherwise bitwise
      let isBoolean = leftTyp.kind == gtBool
      let tbl = if isBoolean: NimGpuBooleanOperators else: NimGpuNumericOperators
      var op = GpuAst(kind: gpuIdent, iName: tbl.getOrDefault(node[0].repr, node[0].repr)) # repr so that open sym choice gets correct name
      op.iSym = op.iName
      result.bOp = op
      result.bLeft = ctx.toGpuAst(reg, node[1])
      result.bRight = ctx.toGpuAst(reg, node[2])

      # We patch the types of int / float literals. WGSL does not automatically convert literals
      # to the target type. Determining the type here _can_ fail. In that case the
      # `lType` field will just be `gtVoid`, like the default.
      if result.bLeft.kind == gpuLit: # and result.bRight.kind != gpuLit:
        # determine literal type based on `bRight`
        result.bLeft.lType = leftTyp
      elif result.bRight.kind == gpuLit: # and result.bLeft.kind != gpuLit:
        # determine literal type based on `bLeft`
        result.bRight.lType = rightTyp

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
    result.iName = node.repr # for sym choices
    if result.iName == "_":
      result.iName = "underscore"
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
      result.iName = sanitized
      result.iSym = s
      if result.iName == "_":
        result.iName = "underscore"
      elif result.iName.startsWith("tmpTuple_"):
        result.iName = "tmpTuple_" & $ctx.genSymCount
        result.iSym = result.iName & "_" & node.signatureHash()
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
    result.cIdent.iTyp = result.cType # also store the type in the symbol, for easier lookup later
    result.cIdent.symbolKind = gsLocal #if atvShared in result.vAttributes: gsShared
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
