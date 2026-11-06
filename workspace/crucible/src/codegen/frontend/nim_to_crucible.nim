# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import ../ir/cirv2_lang
import ../ir/cirv2_build
import nim_type_resolvers
import ../../builtins/builtins_catalog
import ../../builtins/builtins_magic_catalog
import std/[macros, tables, strutils, sets]


# ############################################################
#
#            Nim frontend to CIRv2
#
# ############################################################

type
  ProcEntry = object
    name: string
    fnType: CirType
    hasBody: bool

  SymbolTable = Table[string, tuple[sym: CirExpr, space: AddressSpace]]

  ModuleScope = object
    procs: seq[CirProc]
    procRegistry: Table[uint64, ProcEntry]
    procNameMap: Table[string, uint64]
    vars: SymbolTable
    decls: seq[CirStmt]

  LocalScope = object
    fnName: string
    retType: CirType
    layers: seq[SymbolTable]
    variablesInScope: HashSet[string]

  CirFrontend = object
    resolver: CirTypeResolver
    module: ModuleScope
    locals: LocalScope

# ═══════════════════════════════════════════════════════════════════════
#  Names and scopes
# ═══════════════════════════════════════════════════════════════════════

func mintUniqueName(ctx: var CirFrontend, base: string): string =
  let clean = sanitizeIdent(base)
  var candidate = clean
  var suffix = 0
  while candidate in ctx.locals.variablesInScope or ctx.module.procNameMap.hasKey(candidate):
    inc suffix
    candidate = clean & "_" & $suffix
  ctx.locals.variablesInScope.incl candidate
  result = candidate

func pushScope(ctx: var CirFrontend) =
  ctx.locals.layers.add(SymbolTable())

func popScope(ctx: var CirFrontend) =
  ctx.locals.layers.del(ctx.locals.layers.high)

func declareLocal(ctx: var CirFrontend, reprKey: string, sym: CirExpr, spc = caThread) =
  ctx.locals.layers[^1][reprKey] = (sym, spc)

func getPragmaItem(n: NimNode, name: string): NimNode =
  case n.kind
  of nnkProcDef, nnkFuncDef:
    for p in n.pragma:
      if p.kind in {nnkIdent, nnkSym} and p.strVal == name:
        return p
      if p.kind in {nnkExprColonExpr, nnkCall} and p.len >= 1 and
          p[0].kind in {nnkIdent, nnkSym} and p[0].strVal == name:
        return p
  of nnkSym:
    let impl = n.getImpl()
    case impl.kind
    of nnkProcDef, nnkFuncDef:
      return getPragmaItem(impl, name)
    of nnkIdentDefs, nnkConstDef:
      if impl.len > 0 and impl[0].kind == nnkPragmaExpr:
        for p in impl[0][1]:
          if p.kind in {nnkIdent, nnkSym} and p.strVal == name:
            return p
          if p.kind in {nnkExprColonExpr, nnkCall} and p.len >= 1 and
              p[0].kind in {nnkIdent, nnkSym} and p[0].strVal == name:
            return p
    else:
      discard
  else:
    discard
  return nil

func builtinKindPayload(n: NimNode): NimNode =
  let item = getPragmaItem(n, "builtin")
  if item != nil and item.kind in {nnkExprColonExpr, nnkCall} and item.len >= 2:
    return item[1]
  return nil

func containsCall(n: NimNode): bool =
  if n.isNil:
    return false
  case n.kind
  of nnkCall, nnkCommand, nnkCallStrLit:
    return true
  else:
    discard
  for c in n:
    if containsCall(c):
      return true
  return false

func unwrapConv(n: NimNode): NimNode =
  result = n
  while true:
    case result.kind
    of nnkHiddenStdConv:
      result = result[1]
    of nnkStmtListExpr:
      if result.len == 2 and result[0].kind == nnkEmpty:
        result = result[1]
      else:
        return
    else:
      return

# ═══════════════════════════════════════════════════════════════════════
#  Symbol translation
# ═══════════════════════════════════════════════════════════════════════

func lowerExpr(ctx: var CirFrontend, n: NimNode): CirExpr

func resolveTypeSection(ctx: var CirFrontend, n: NimNode) =
  ## A generic definition registers nothing at declaration time,
  ## resolution of its instantiations happens at the use sites.
  for typeDef in n:
    let head = typeDef[0]
    if head.kind == nnkBracketExpr or head.typeKind == ntyGenericBody:
      continue
    let typeNode = if head.kind == nnkPragmaExpr: head[0] else: head
    discard ctx.resolver.resolveCirType(typeNode)

func builtinSymExpr(ctx: var CirFrontend, n: NimNode): CirExpr =
  let name = sanitizeIdent(n.strVal)
  let typ = ctx.resolver.resolveCirType(n.getTypeInst())
  result = sym(name, typ, builtin = true)
  result.coordBuiltin = getBuiltinKind(CirCoordBuiltinKind, builtinKindPayload(n))

# ═══════════════════════════════════════════════════════════════════════
#  Call translation
# ═══════════════════════════════════════════════════════════════════════

func getFormalParams(fn: NimNode): NimNode =
  for i, c in fn:
    if c.kind == nnkFormalParams:
      return c
  return nil

func resolveCallCallee(callee: NimNode): NimNode =
  ## Nim resolves overloads before the macro runs, a choice node's first candidate is the resolved symbol.
  if callee.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and callee.len > 0:
    return callee[0]
  return callee

func getProcBody(fn: NimNode): NimNode =
  var i = fn.len - 1
  while i >= 0 and fn[i].kind == nnkSym:
    i.dec
  if i >= 0:
    if fn[i].kind == nnkStmtList:
      return fn[i]
    if fn[i].kind != nnkEmpty:
      return nnkStmtList.newTree(fn[i])
  return nil

func isGenericDef(fn: NimNode): bool =
  for c in fn:
    if c.kind == nnkGenericParams:
      return true
    if c.kind == nnkBracket and c.len > 0 and c[0].kind == nnkEmpty:
      for g in c:
        if g.kind == nnkGenericParams:
          return true
  return false

func getProcIdentity(callee: NimNode): uint64 =
  let impl = callee.getImpl()
  let generic = impl.kind in [nnkProcDef, nnkFuncDef] and
    (impl[2].kind == nnkGenericParams or
      (impl.len > 5 and impl[5].kind == nnkBracket and
        impl[5].len > 1 and impl[5][1].kind == nnkGenericParams))
  let key = callee.repr & "_" & callee.signatureHash() &
    (if generic: "_" & callee.symBodyHash() else: "")
  typeIdentityHash(key)

func claimProcName(ctx: var CirFrontend, identity: uint64, base: string): string =
  if ctx.module.procNameMap.getOrDefault(base, identity) == identity:
    ctx.module.procNameMap[base] = identity
    return base
  result = base & "_" & base58Short(identity)
  while ctx.module.procNameMap.getOrDefault(result, identity) != identity:
    result.add("_")
  ctx.module.procNameMap[result] = identity

func registerOrGet(ctx: var CirFrontend, calleeSym: NimNode): ProcEntry =
  let id = getProcIdentity(calleeSym)
  if ctx.module.procRegistry.hasKey(id):
    result = ctx.module.procRegistry[id]
    if result.fnType.isNil:
      # a definition registers before its body lowers, so the type is read
      # once at the first call inside it
      result.fnType = ctx.resolver.resolveCirType(calleeSym.getTypeInst())
      ctx.module.procRegistry[id].fnType = result.fnType
    return
  let fnType = ctx.resolver.resolveCirType(calleeSym.getTypeInst())
  let name = ctx.claimProcName(id,
    mangleOperatorName(calleeSym.strVal) & "_" & base58Short(id))
  result = ProcEntry(name: name, fnType: fnType)
  ctx.module.procRegistry[id] = result

func harvestProc(ctx: var CirFrontend, callee: NimNode, cirName: string)

func lowerCallArgs(ctx: var CirFrontend, callee: NimNode, call: NimNode): seq[CirExpr] =
  ## Compile-time formals and call arguments align positionally.
  ## The compile-time arguments are dropped, their values are in the callee's types.
  var formalIdx = 0
  var argIdx = 0
  let formals = getFormalParams(callee.getTypeInst())
  while formalIdx + 1 < formals.len and argIdx + 1 < call.len:
    formalIdx.inc
    let ftype = formals[formalIdx][formals[formalIdx].len - 2]
    argIdx.inc
    if isCompileTimeFormal(ftype):
      continue
    let arg = ctx.lowerExpr(call[argIdx])
    result.add(arg)

func getNimMagicName(n: NimNode): string =
  ## Value of the definition's `magic: "..."` pragma, "" when absent.
  ## System integer and float operators are magics:
  ## - their effect is the primitive operation
  ## - their Nim body is a documentation stub
  let impl = n.getImpl()
  if impl.kind notin {nnkProcDef, nnkFuncDef}:
    return ""
  for p in impl.pragma:
    if p.kind in {nnkExprColonExpr, nnkCall} and p.len >= 2 and
        p[0].kind in {nnkIdent, nnkSym} and p[0].strVal == "magic":
      if p[1].kind in {nnkStrLit, nnkRStrLit, nnkIdent}:
        return p[1].strVal
      return "magic"
  return ""

func isTriviaBody(body: NimNode): bool =
  for c in body:
    if c.kind notin {nnkCommentStmt, nnkEmpty}:
      return false
  return true

func harvestedCallee(ctx: var CirFrontend, calleeSym: NimNode): CirExpr =
  ## A builtin callee is emitted under its canonical name:
  ## a hash-suffixed instantiation name would denote a proc no backend defines.
  ## The frontend reads the synchro, reduction, shuffle and atomics family kinds off the `{.builtin.}` payload.
  ## Every printer and pass dispatches on those kinds.
  let calleeBuiltin = getPragmaItem(calleeSym, "builtin") != nil
  let entry = ctx.registerOrGet(calleeSym)
  if not calleeBuiltin and not entry.hasBody:
    # a call lowers 1:1 into the callee's body. A callee without a lowerable
    # body has no runtime content, a bodiless magic's documentation stub
    # renders exactly this way, so refuse and name the construct
    let body = getProcBody(calleeSym.getImpl())
    if body.isNil or isTriviaBody(body):
      raiseAssert("call to " & calleeSym.repr & " has no body to lower")
    ctx.harvestProc(calleeSym, entry.name)
  let calleeName = if calleeBuiltin: calleeSym.strVal
                   else: entry.name
  result = sym(calleeName, entry.fnType, calleeBuiltin)
  if calleeBuiltin:
    let payload = builtinKindPayload(calleeSym)
    result.synchroBuiltin = getBuiltinKind(CirSynchroBuiltinKind, payload)
    result.reductionBuiltin = getBuiltinKind(CirReductionBuiltinKind, payload)
    result.shuffleBuiltin = getBuiltinKind(CirShuffleBuiltinKind, payload)
    result.atomicsBuiltin = getBuiltinKind(CirAtomicsBuiltinKind, payload)
  return result

func arithExpr(op: string, t: CirType, x, y: CirExpr): CirExpr =
  case op
  of "+": addE(t, x, y)
  of "-": subE(t, x, y)
  of "*": mulE(t, x, y)
  of "/", "div": divE(t, x, y)
  of "mod": modE(t, x, y)
  of "shl": shlE(t, x, y)
  else: shrE(t, x, y)

func lowerCallValue(ctx: var CirFrontend, callee: NimNode, call: NimNode): CirExpr =
  let magic = getNimMagicName(callee)
  # toOpenArray builds a span and len reads a span's length: the span split
  # pass consumes both forms, no proc harvests, no call emits
  let targetName = if callee.kind == nnkSym: callee.strVal
                   elif callee.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and callee.len > 0:
                     callee[0].strVal
                   else: ""
  if magic == "DotDot":
    let lo = ctx.lowerExpr(call[1])
    let hi = ctx.lowerExpr(call[2])
    let typ = ctx.resolver.resolveStructType(call.getTypeInst(),
        initTable[string, AddressSpace]())
    return oconstrE(typ, kv("a", lo), kv("b", hi))
  if magic == "Slice" and targetName == "toOpenArray":
    # A pointer base is required: the span split pass offsets the base by `first`
    # on the pointer side, and CIRv2 declares pointer arithmetic over pointers.
    let base = ctx.lowerExpr(call[1])
    let first = ctx.lowerExpr(unwrapConv(call[2]))
    let last = ctx.lowerExpr(unwrapConv(call[3]))
    let baseTyp = getExprType(base)
    if baseTyp.isNil or baseTyp.kind notin {ctPtr, ctAptr}:
      raiseAssert("toOpenArray over a " & (if baseTyp.isNil: "base with no derivable type" else: $baseTyp.kind) & " is outside the supported set, pass a pointer to the storage")
    let firstTyp = getExprType(first)
    let lastTyp = getExprType(last)
    let spanTyp = tSpan(baseTyp.addressSpace, baseTyp.pElem)
    let callTyp = tProcT(baseTyp, firstTyp, lastTyp, spanTyp)
    return callE(sym("toOpenArray", callTyp), base, first, last)
  if magic == "LengthOpenArray":
    # a `len(s)` read over an openArray lowers to the `len` call
    # that the span split pass folds to the span's split length binding
    let arg = ctx.lowerExpr(call[1])
    let argTyp = getExprType(arg)
    let readTyp = tProcT(argTyp, tInt(32))
    return callE(sym("len", readTyp), arg)
  if magic in NimGpuNumericBuiltinsOps:
    let args = ctx.lowerCallArgs(callee, call)
    let opType = getExprType(args[0])
    return arithExpr(NimGpuNumericBuiltinsOps[magic], opType, args[0], args[1])
  if magic in NimGpuNumericBuiltinsUnaryOps:
    # a `pred(x, step)` call lowers to `x - step`,
    # a `succ(x, step)` call to `x + step`
    # `step` has default value 1
    let args = ctx.lowerCallArgs(callee, call)
    let opType = getExprType(args[0])
    let step = if args.len > 1: args[1] else: intLit(1, opType)
    return arithExpr(NimGpuNumericBuiltinsUnaryOps[magic], opType, args[0], step)
  let calleeSym = resolveCallCallee(callee)
  # `abs` over scalars lowers as a builtin call, printers use the native name.
  # Only `abs` intercepts: system `abs` over floats has a refused body.
  # `min`/`max` harvest into a ternary idiom.
  # An aggregate `abs` proc takes the harvest path below.
  if calleeSym.strVal == "abs":
    let fnType = ctx.resolver.resolveCirType(calleeSym.getTypeInst())
    var scalar = true
    for p in fnType.params:
      if not isScalarType(p):
        scalar = false
        break
    if scalar:
      let args = ctx.lowerCallArgs(calleeSym, call)
      return callE(sym(calleeSym.strVal, fnType, true), args)
  let calleeExpr = ctx.harvestedCallee(calleeSym)
  let args = ctx.lowerCallArgs(calleeSym, call)
  return callE(calleeExpr, args)

# ═══════════════════════════════════════════════════════════════════════
#  Expression translation
# ═══════════════════════════════════════════════════════════════════════

func lowerOperatorExpr(ctx: var CirFrontend, n: NimNode): CirExpr =
  case n.kind
  of nnkInfix:
    let op = n[0].strVal
    if op == "..":
      return ctx.lowerCallValue(n[0], n)
    case op
    of "+", "-", "*", "div", "mod", "shl", "shr", "/", "and", "or", "xor":
      if not (ctx.isBuiltinScalar(n[1]) and ctx.isBuiltinScalar(n[2])):
        # overloaded operator over non-primitive-scalar operands: the check fails,
        # overload resolution picks the proc, lowered here as a call.
        # Example: `m_i - m_ij` over (MT x 1) tiles. Each operand lowers once,
        # inside the call, since this check runs before operand lowering.
        return ctx.lowerCallValue(n[0], n)
    else:
      discard
    let a = ctx.lowerExpr(unwrapConv(n[1]))
    let b = ctx.lowerExpr(unwrapConv(n[2]))
    case op
    of "==", "!=", "<", "<=", ">", ">=":
      case op
      of "==":
        result = eqE(a, b)
      of "!=":
        result = neqE(a, b)
      of "<":
        result = ltE(a, b)
      of "<=":
        result = leE(a, b)
      of ">":
        result = ltE(b, a)
      else:
        result = leE(b, a)
      return result
    of "and", "or", "xor":
      # bool vs bitwise domain resolves in the expression-form pass: a bitwise form emits uniformly here, operand types decide in the pass
      let at = getExprType(a)
      case op
      of "and":
        return bitAndE(at, a, b)
      of "or":
        return bitOrE(at, a, b)
      else:
        return bitXorE(at, a, b)
    of "+", "-", "*", "/", "div", "mod", "shl", "shr":
      let opType = getExprType(a)
      return arithExpr(op, opType, a, b)
    else:
      raiseAssert("operator " & op & " is outside the supported set")
  of nnkPrefix:
    let op = n[0].strVal
    let operand = ctx.lowerExpr(unwrapConv(n[1]))
    case op
    of "-":
      return negE(getExprType(operand), operand)
    of "not":
      return notE(operand)
    else:
      raiseAssert("prefix operator " & op & " is outside the supported set")
  else:
    raiseAssert "lowerOperatorExpr: unexpected node kind " & $n.kind

func isBuiltinScalar(ctx: var CirFrontend, n: NimNode): bool =
  let t = ctx.resolver.resolveCirType(n.getTypeInst())
  return isScalarType(t)

func getBranchValueExpr(n: NimNode): NimNode =
  result = n
  while result.kind == nnkStmtList and result.len == 1:
    result = result[0]

func isStmtBody(n: NimNode): bool =
  case n.kind
  of nnkStmtList, nnkAsgn, nnkLetSection, nnkVarSection, nnkConstSection,
      nnkIfStmt, nnkWhileStmt, nnkForStmt, nnkReturnStmt, nnkDiscardStmt,
      nnkBlockStmt, nnkCaseStmt:
    result = true
  else:
    result = false

func lowerSplicedStmt(ctx: var CirFrontend, c: NimNode): seq[CirStmt]

func lowerBranchValue(ctx: var CirFrontend, body: NimNode): CirExpr =
  let valueNode = getBranchValueExpr(body)
  if not isStmtBody(valueNode):
    return ctx.lowerExpr(valueNode)
  var stmts: seq[CirStmt] = @[]
  var value: CirExpr = nil
  var list = valueNode
  if list.kind != nnkStmtList:
    list = nnkStmtList.newTree(valueNode)
  for i, c in list:
    if i == list.len - 1:
      value = ctx.lowerExpr(c)
    else:
      stmts.add(ctx.lowerSplicedStmt(c))
  if stmts.len == 0:
    return value
  return stmtsE(stmts, value)

func lowerNestedTernary(ctx: var CirFrontend,
    branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode, idx: int,
    resType: CirType): CirExpr =
  ## Branch `idx` of an if-expression as a ternary whose else variant
  ## contains the next branch, first match wins.
  ## `elseBody` is the innermost else variant.
  let cond = ctx.lowerExpr(branches[idx].cond)
  let thenV = ctx.lowerBranchValue(branches[idx].body)
  let elseV = if idx + 1 < branches.len:
                 lowerNestedTernary(ctx, branches, elseBody, idx + 1, resType)
               else:
                 ctx.lowerBranchValue(elseBody)
  return ternaryE(resType, cond, thenV, elseV)

func lowerStmt(ctx: var CirFrontend, n: NimNode): seq[CirStmt]

func lowerBranchBody(ctx: var CirFrontend, bodyNode: NimNode): seq[CirStmt] =
  ctx.pushScope()
  result = ctx.lowerStmt(bodyNode)
  ctx.popScope()

func lowerIfForm(ctx: var CirFrontend, branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode): seq[CirStmt] =
  ## An elif chain lowers to one if-then-else per branch, first match wins.
  ## Else statements sit innermost. The rewriteStmtsExprs pass places
  ## a condition's statements before the test.
  var tail: seq[CirStmt] = @[]
  if not elseBody.isNil:
    tail = ctx.lowerBranchBody(elseBody)
  for i in countdown(branches.len - 1, 0):
    let cond = ctx.lowerExpr(branches[i].cond)
    let body = ctx.lowerBranchBody(branches[i].body)
    tail = @[iteS(cond, body, tail)]
  result = tail

func lowerSplicedStmt(ctx: var CirFrontend, c: NimNode): seq[CirStmt] =
  ## One non-final child of an expression block, a statement running before
  ## the block's value, wrapped by the block's statement-expression.
  ## An object or tuple construction with no call inside has no runtime code
  ## and is dropped.
  if c.kind == nnkStmtList:
    for child in c:
      result.add(ctx.lowerSplicedStmt(child))
    return result
  if c.kind in {nnkObjConstr, nnkTupleConstr, nnkExprColonExpr, nnkBracketExpr,
                nnkPar, nnkIntLit .. nnkUInt64Lit, nnkFloatLit, nnkStrLit} and
      not containsCall(c):
    return @[]
  return ctx.lowerStmt(c)

func lowerExpr(ctx: var CirFrontend, n: NimNode): CirExpr =
  ## Unsupported constructs refuse immediately.
  if n.kind == nnkNilLit:
    # a nil literal node also tests as isNil, so it is handled before the kind dispatch
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    return nilE(typ)
  case n.kind
  of nnkSym:
    let name = n.strVal
    case name
    of "true":
      return trueE()
    of "false":
      return falseE()
    else:
      discard
    case n.symKind
    of nskParam, nskVar, nskLet, nskForVar, nskResult, nskTemp:
      if getPragmaItem(n, "builtin") != nil:
        return ctx.builtinSymExpr(n)
      var found: CirExpr = nil
      for i in countdown(ctx.locals.layers.high, 0):
        if ctx.locals.layers[i].hasKey(n.repr):
          found = ctx.locals.layers[i][n.repr].sym
          break
      if found.isNil:
        raiseAssert("unknown local symbol " & n.repr)
      return found
    of nskConst:
      if getPragmaItem(n, "builtin") != nil:
        return ctx.builtinSymExpr(n)
      return ctx.lowerExpr(n.getImpl()[2])
    else:
      raiseAssert("symbol " & name & " (" & $n.symKind & ") is not a value")
  of nnkIntLit .. nnkUInt64Lit:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    return intLit(n.intVal, typ)
  of nnkFloat32Lit, nnkFloat64Lit, nnkFloatLit:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    return fltLit(n.floatVal, typ)
  of nnkHiddenStdConv:
    # int/int32 share one bit width and the standard conversion is the identity
    # for values. A float literal sits at its default width: the conv node
    # reaches the expression-form pass, which re-types it to the target
    let inner = n[1]
    if inner.kind in {nnkFloat32Lit, nnkFloat64Lit, nnkFloatLit}:
      let typ = ctx.resolver.resolveCirType(n.getTypeInst())
      return convE(typ, ctx.lowerExpr(inner))
    return ctx.lowerExpr(inner)
  of nnkHiddenSubConv, nnkConv:
    var typ = if n[0].kind == nnkEmpty:
      ctx.resolver.resolveCirType(n.getTypeInst())
    else:
      ctx.resolver.resolveCirType(n[0])
    let operand = ctx.lowerExpr(n[1])
    let ot = getExprType(operand)
    if ot != nil and sameType(ot, typ):
      # an identity conversion: the layout algebra's static wrappers stack up and the value already has the target type
      return operand
    if operand.kind in {eIntLit, eFloatLit}:
      # a literal conversion is exact, the expression-form pass re-types it
      # to the conversion target
      return convE(typ, operand)
    if typ.kind in {ctPtr, ctAptr}:
            typ = typ.withAddressSpace(resolveAddressSpace(operand))
    return convE(typ, operand)
  of nnkCast:
    var typ = ctx.resolver.resolveCirType(n[0])
    let operand = ctx.lowerExpr(n[1])
    if typ.kind in {ctPtr, ctAptr}:
      typ = typ.withAddressSpace(resolveAddressSpace(operand))
    return castE(typ, operand)
  of nnkHiddenDeref, nnkDerefExpr:
    let base = ctx.lowerExpr(n[0])
    let bt = getExprType(base)
    if bt != nil and bt.kind == ctSpan:
      # a sem deref of a span lowers to the span itself, an index over it selects one element
      return base
    return derefE(base)
  of nnkHiddenAddr, nnkAddr:
    let base = ctx.lowerExpr(n[0])
    let bt = getExprType(base)
    # address space: the storage class of the base value's memory
    var addressSpace = caThread
    var walk = base
    while true:
      case walk.kind
      of ePat, eAt:
        let next = if walk.kind == ePat: walk.patBase else: walk.atBase
        let ct = getExprType(next)
        if ct != nil and ct.kind in {ctPtr, ctAptr}:
          addressSpace = ct.addressSpace
          break
        walk = next
      of eDeref:
        let ct = getExprType(walk.derefOperand)
        if ct != nil and ct.kind in {ctPtr, ctAptr}:
          addressSpace = ct.addressSpace
        break
      of eDot:
        walk = walk.dotBase
      of eSym:
        for i in countdown(ctx.locals.layers.high, 0):
          if ctx.locals.layers[i].hasKey(walk.name):
            addressSpace = ctx.locals.layers[i][walk.name].space
            break
        break
      else:
        break
    return addrE(tPtr(addressSpace, bt), base)
  of nnkBracketExpr:
    let baseNode = n[0]
    var loweredBase: CirExpr = nil
    if baseNode.kind == nnkHiddenDeref:
      loweredBase = ctx.lowerExpr(baseNode[0])
    else:
      loweredBase = ctx.lowerExpr(baseNode)
    let idx = ctx.lowerExpr(unwrapConv(n[1]))
    let bt = getExprType(loweredBase)
    case bt.kind
    of ctAptr:
      return patE(loweredBase, idx)
    of ctArray:
      return atE(loweredBase, idx)
    of ctPtr:
      # a var parameter of array type reaches the array through a pointer: the deref yields that array and the index selects one element
      return atE(derefE(loweredBase), idx)
    of ctSpan:
      # a span element read whose base symbol the span split pass retypes
      # to the element pointer
      return patE(loweredBase, idx)
    of ctStruct:
      return dotE(loweredBase, bt.fields[idx.value].name)
    else:
      raiseAssert("indexing over type " & $(bt.kind) & " is outside the supported set")
  of nnkDotExpr:
    let base = ctx.lowerExpr(n[0])
    return dotE(base, sanitizeIdent(n[1].strVal))
  of nnkInfix, nnkPrefix:
    return ctx.lowerOperatorExpr(n)
  of nnkCall, nnkCommand:
    let callee = n[0]
    if callee.kind notin {nnkSym, nnkOpenSymChoice, nnkClosedSymChoice}:
      raiseAssert("value call callee " & callee.repr & " is not a proc symbol")
    return ctx.lowerCallValue(callee, n)
  of nnkObjConstr:
    var lowereds: seq[tuple[name: string, v: CirExpr, isNil: bool]] = @[]
    var valueSpaces = initTable[string, AddressSpace]()
    for i in 1 ..< n.len:
      let fieldName = sanitizeIdent(n[i][0].strVal)
      if n[i][1].kind == nnkEmpty:
        lowereds.add((fieldName, nil, true))
        continue
      let v = ctx.lowerExpr(n[i][1])
      # a pointer-typed field takes the address space its value proves
      valueSpaces[fieldName] = resolveAddressSpace(v)
      lowereds.add((fieldName, v, false))
    let typ = ctx.resolver.resolveStructType(n.getTypeInst(), valueSpaces)
    # completeConstructors completes fields missing from the constructor
    var fields: seq[CirKv] = @[]
    for (name, v, isNil) in lowereds:
      if isNil:
        for f in typ.fields:
          if f.name == name:
            fields.add(kv(name, nilE(f.typ)))
            break
      else:
        fields.add(kv(name, v))
    return oconstrE(typ, fields)
  of nnkExprColonExpr:
    # a named field outside a constructor: the identifier is compile-time
    # vocabulary of the expansion and the value is the runtime content
    return ctx.lowerExpr(n[1])
  of nnkTupleConstr:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    var fields: seq[CirKv] = @[]
    for i in 0 ..< n.len:
      let v = ctx.lowerExpr(n[i])
      fields.add(kv("F" & $i, v))
    return oconstrE(typ, fields)
  of nnkBracket:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    var elems: seq[CirExpr] = @[]
    for el in n:
      let v = ctx.lowerExpr(el)
      elems.add(v)
    return aconstrE(typ, elems)
  of nnkPar:
    if n.len == 1:
      return ctx.lowerExpr(n[0])
    else:
      raiseAssert "parenthesized expression with " & $n.len & " children"
  of nnkStmtListExpr:
    var stmts: seq[CirStmt] = @[]
    var value: CirExpr = nil
    for i, c in n:
      if c.kind == nnkEmpty:
        continue
      if i == n.len - 1:
        value = ctx.lowerExpr(c)
      else:
        stmts.add(ctx.lowerSplicedStmt(c))
    if stmts.len == 0:
      return value
    return stmtsE(stmts, value)
  of nnkBlockExpr, nnkBlockStmt:
    let inner = n[^1]
    if inner.kind != nnkStmtList:
      # a single-statement body lowers unwrapped and the block value is the expression itself,
      # matching the shape `getProcBody` boxes for proc bodies
      return ctx.lowerExpr(inner)
    var stmts: seq[CirStmt] = @[]
    var value: CirExpr = nil
    for i, c in inner:
      if i == inner.len - 1 and c.kind notin {nnkAsgn, nnkIfStmt, nnkForStmt,
          nnkWhileStmt, nnkDiscardStmt, nnkReturnStmt}:
        value = ctx.lowerExpr(c)
      else:
        stmts.add(ctx.lowerSplicedStmt(c))
    if value.isNil:
      raiseAssert "block expression in expression position ends in a statement," &
        " a semchecked block expression ends in an expression"
    if stmts.len == 0:
      return value
    return stmtsE(stmts, value)
  of nnkIfExpr:
    let resType = ctx.resolver.resolveCirType(n.getTypeInst())
    var branches: seq[tuple[cond: NimNode, body: NimNode]] = @[]
    var elseBody: NimNode = nil
    for c in n:
      case c.kind
      of nnkElifBranch, nnkElifExpr:
        branches.add((cond: c[0], body: c[1]))
      of nnkElse, nnkElseExpr:
        elseBody = c[0]
      else:
        raiseAssert "if expression: unexpected branch " & $c.kind
    return lowerNestedTernary(ctx, branches, elseBody, 0, resType)
  of nnkStrLit, nnkRStrLit, nnkTripleStrLit:
    # a string literal is C string text: nnkStrLit escapes it, the raw string
    # forms carry their text verbatim, the printers add the quotes
    let text = if n.kind == nnkStrLit: n.strVal.escape("", "") else: n.strVal
    return strLit(text)
  else:
    raiseAssert("expression construct not supported: " & $n.kind & " " & n.repr)

# ═══════════════════════════════════════════════════════════════════════
#  Statements
# ═══════════════════════════════════════════════════════════════════════

func lowerVoidCall(ctx: var CirFrontend, n: NimNode): seq[CirStmt] =
  ## A call in statement position: a barrier, a kernel call, or a discardable
  ## non-void call whose value the statement discards.
  let callee = n[0]
  # a generic type instantiation left in statement position appears
  # as the `[]` choice with the type as its first operand, and a type has
  # no runtime code
  if callee.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and callee.len > 0 and
      callee[0].strVal == "[]" and n.len >= 2 and
      n[1].kind == nnkSym and n[1].symKind == nskType:
    return @[]
  if callee.kind notin {nnkSym, nnkOpenSymChoice, nnkClosedSymChoice}:
    raiseAssert("call callee " & callee.repr & " is not a proc symbol")
  let calleeSym = resolveCallCallee(callee)
  let calleeExpr = ctx.harvestedCallee(calleeSym)
  let args = ctx.lowerCallArgs(calleeSym, n)
  return @[callS(calleeExpr, args)]

func collectAddressSpace(ctx: var CirFrontend, nameNode: NimNode): AddressSpace =
  result = caThread
  if nameNode.kind != nnkPragmaExpr:
    return
  var seen = false
  template setSpace(spc: AddressSpace) =
    if seen:
      raiseAssert("multiple address-addressSpace pragmas on one local")
    seen = true
    result = spc
  for p in nameNode[1]:
    case p.strVal.toLowerAscii
    of "smem": setSpace(caThreadgroup)
    of "const_mem": setSpace(caConstant)
    of "rmem": setSpace(caThread)
    of "noinit", "inject", "gensym":
      discard
    else:
      raiseAssert("unexpected pragma " & p.repr & " on a local declaration")

func lowerForStmt(ctx: var CirFrontend, n: NimNode): seq[CirStmt]
func lowerAsmStmt(ctx: var CirFrontend, n: NimNode): seq[CirStmt]

func lowerStmt(ctx: var CirFrontend, n: NimNode): seq[CirStmt] =
  ## Unsupported constructs refuse immediately.
  if n.isNil:
    return @[]
  case n.kind
  of nnkCommentStmt, nnkEmpty:
    # an expansion hole or a comment, no code either way
    return @[]
  of nnkStmtList:
    for c in n:
      result.add(ctx.lowerStmt(c))
  of nnkLetSection, nnkVarSection:
    for defs in n:
      let typeNode = defs[^2]
      let initNode = defs[^1]
      for j in 0 ..< defs.len - 2:
        let nameNode = defs[j]
        let spc = ctx.collectAddressSpace(nameNode)
        let nimName = if nameNode.kind == nnkPragmaExpr: nameNode[0].strVal
                      else: nameNode.strVal
        var typ: CirType = nil
        if typeNode.kind != nnkEmpty:
          typ = ctx.resolver.resolveCirType(typeNode)
        else:
          typ = ctx.resolver.resolveCirType(initNode.getTypeInst())
        var value: CirExpr = nil
        if initNode.kind != nnkEmpty:
          value = ctx.lowerExpr(initNode)
          if value.kind == eOconstr and typ != nil and typ.kind == ctStruct and
              value.ocType.kind == ctStruct:
            # a constructor value is typed by the struct it instantiates,
            # its declaration uses that type, and the field address spaces
            # come from the value's type
            typ = value.ocType
          elif typ != nil and typ.kind in {ctPtr, ctAptr} and
              getExprType(value) != nil and getExprType(value).kind == typ.kind and
              getExprType(value).addressSpace != typ.addressSpace:
            # a pointer cast preserves the operand's address space:
            # - the declaration takes the value's space
            # - `let q = cast[ptr T](addr scratch[0])` is a threadgroup pointer
            typ = getExprType(value)
        let cirName = ctx.mintUniqueName(nimName)
        let lookupKey = if nameNode.kind == nnkPragmaExpr: nameNode[0].repr
                        else: nameNode.repr
        ctx.declareLocal(lookupKey, sym(cirName, typ), spc)
        if value.isNil:
          result.add(varD(cirName, typ, nil, spc))
        else:
          result.add(varD(cirName, typ, value, spc))
  of nnkConstSection:
    # a const section emits no runtime code: the resolver registers the value's struct types
    for defs in n:
      let typeNode = defs[1]
      let valueNode = defs[2]
      if typeNode.kind != nnkEmpty:
        discard ctx.resolver.resolveCirType(typeNode)
      else:
        discard ctx.resolver.resolveCirType(valueNode.getTypeInst())
  of nnkAsgn:
    let lhs = ctx.lowerExpr(n[0])
    let rhs = ctx.lowerExpr(n[1])
    return @[asgnS(lhs, rhs)]
  of nnkInfix:
    # compound assignment lowers in statement position as an infix:
    # a user-defined `op=` lowers as the proc's var-target call,
    # a system compound form lowers as primitive scalar arithmetic
    let op = n[0].strVal
    if op in ["+=", "-=", "*=", "/="]:
      # user operator vs system form, read off the callee definition:
      # system compound ops are Inc/Dec magics or generic procs,
      # a user overload is an ordinary proc of the module
      let impl = resolveCallCallee(n[0]).getImpl()
      if impl.kind in {nnkProcDef, nnkFuncDef} and not isGenericDef(impl):
        return ctx.lowerVoidCall(n)
      # a generic callee may still be a user operator over an aggregate:
      # operand kinds decide, a non-scalar target lowers as the call
      let lt = ctx.resolver.resolveCirType(n[1].getTypeInst())
      let rt = ctx.resolver.resolveCirType(n[2].getTypeInst())
      let lp = if lt != nil and lt.kind == ctPtr: lt.pElem else: lt
      if isScalarType(lp) and isScalarType(rt):
        var lhs = ctx.lowerExpr(n[1])
        # this lowering dereferences a var-returning `[]` call's element address, and the assignment target is the location behind it
        if lhs.kind == eAddr:
          lhs = lhs.addrOperand
        elif lhs.kind == eSym and getExprType(lhs) != nil and
            getExprType(lhs).kind == ctPtr:
          # a var parameter is a pointer to its location and the compound assignment operates on the pointee
          lhs = derefE(lhs)
        let rhs = ctx.lowerExpr(unwrapConv(n[2]))
        let opType = getExprType(lhs)
        # a desugared `op=` left-hand side evaluates exactly once in Nim
        # temp-binding normalization splits the reads of a call-bearing target
        let arith = arithExpr(op[0 ..< op.len - 1], opType, lhs, rhs)
        return @[asgnS(lhs, arith)]
    return ctx.lowerVoidCall(n)
  of nnkIfStmt:
    var branches: seq[tuple[cond, body: NimNode]] = @[]
    var elseBody: NimNode = nil
    for c in n:
      case c.kind
      of nnkElifBranch:
        branches.add((cond: c[0], body: c[1]))
      of nnkElse:
        elseBody = c[0]
      else:
        raiseAssert "if statement: unexpected child " & $c.kind
    return ctx.lowerIfForm(branches, elseBody)
  of nnkWhileStmt:
    # rewriteStmtsExprs places a condition's statements before every test
    let cond = ctx.lowerExpr(n[0])
    ctx.pushScope()
    let body = ctx.lowerStmt(n[1])
    ctx.popScope()
    return @[whileS(cond, body)]
  of nnkForStmt:
    return ctx.lowerForStmt(n)
  of nnkAsmStmt:
    return ctx.lowerAsmStmt(n)
  of nnkBlockStmt:
    # a `break label` inside the body is refused at the break statement.
    # A label that no break targets is dead and lowers as its body
    ctx.pushScope()
    result = ctx.lowerStmt(n[1])
    ctx.popScope()
  of nnkBreakStmt:
    raiseAssert("break: unsupported in device code, restructure the loop condition")
  of nnkContinueStmt:
    raiseAssert("continue: unsupported in device code, restructure the loop condition")
  of nnkCaseStmt:
    raiseAssert("case statement: outside the supported set")
  of nnkTypeSection:
    ctx.resolveTypeSection(n)
    return @[]
  of nnkPragma:
    var markUnroll = false
    var emitArgs: seq[NimNode] = @[]
    for p in n:
      if p.kind in {nnkIdent, nnkSym} and p.strVal == "unroll":
        # Unroll marking consumes this marker statement: the loop whose body
        # contains it gets the unroll flag
        markUnroll = true
      elif p.kind in {nnkExprColonExpr, nnkCall} and p.len >= 2 and
          p[0].kind in {nnkIdent, nnkSym} and p[0].strVal == "emit":
        emitArgs.add(p[1])
    var emitted: seq[CirStmt] = @[]
    if markUnroll:
      emitted.add(emitS(@["{.unroll.}"], @[]))
    if emitted.len == 0 and emitArgs.len == 0:
      return @[]
    for emitArg in emitArgs:
      # literal runs alternate with spliced expressions: a computed splice
      # stores through a fresh temp that the emitted text references by name
      if emitArg.kind in {nnkStrLit, nnkRStrLit, nnkTripleStrLit}:
        emitted.add(emitS(@[emitArg.strVal], @[]))
        continue
      var parts: seq[string] = @[""]
      var operands: seq[CirExpr] = @[]
      for part in emitArg:
        case part.kind
        of nnkStrLit, nnkRStrLit, nnkTripleStrLit:
          parts[^1].add(part.strVal)
        else:
          let e = ctx.lowerExpr(part)
          var op = e
          if e.kind != eSym:
            # printers splice symbol names, a computed value needs a temp
            let tmpName = ctx.mintUniqueName("emit_tmp")
            let tmpType = getExprType(e)
            op = sym(tmpName, tmpType)
            emitted.add(@[varD(tmpName, tmpType), asgnS(op, e)])
            ctx.declareLocal(tmpName, op)
          parts.add("")
          operands.add(op)
      emitted.add(emitS(parts, operands))
    return emitted
  of nnkTemplateDef, nnkMacroDef, nnkIteratorDef:
    # a definition left in a typed body is a compile-time artifact:
    # templates and macros execute at expansion, iterators inline into
    # their call sites, an unused definition has no runtime code
    return @[]
  of nnkProcDef, nnkFuncDef, nnkConverterDef, nnkMethodDef:
    raiseAssert("nested proc definition " & n.repr)
  of nnkObjConstr, nnkTupleConstr, nnkBracketExpr, nnkPar:
    # a discarded construction step of a macro expansion drops its value:
    # this lowering keeps the side effects only, as lowerSplicedStmt does for the spliced statements of an expression
    let v = ctx.lowerExpr(n)
    if v.kind == eStmts:
      return v.stmts
    return @[]
  of nnkExprColonExpr:
    # a discarded named-field step of a macro expansion whose value is never read
    let v = ctx.lowerExpr(n[1])
    if v.kind == eStmts:
      return v.stmts
    return @[]
  of nnkTryStmt, nnkRaiseStmt:
    raiseAssert($n.kind & ": exceptions are outside the supported set")
  of nnkBindStmt, nnkMixinStmt:
    # symbol-binding directives, Nim resolves them at sem time, no trace remains in the lowered code
    return @[]
  of nnkReturnStmt:
    var valueNode = n[0]
    if valueNode.kind == nnkAsgn:
      valueNode = valueNode[1]
    if valueNode.kind == nnkEmpty:
      if ctx.locals.retType.kind != ctVoid:
        return @[retS(sym("result", ctx.locals.retType))]
      return @[retS(nil)]
    let value = ctx.lowerExpr(valueNode)
    return @[retS(value)]
  of nnkDiscardStmt:
    if n[0].kind == nnkEmpty:
      return @[discardS(nil)]
    if n[0].kind in {nnkCall, nnkCommand}:
      return ctx.lowerVoidCall(n[0])
    let value = ctx.lowerExpr(n[0])
    return @[discardS(value)]
  of nnkCall, nnkCommand:
    return ctx.lowerVoidCall(n)

  else:
    raiseAssert("statement construct not supported: " & $n.kind & " " & n.repr)

func lowerForStmt(ctx: var CirFrontend, n: NimNode): seq[CirStmt] =
  ## Lowers a for statement over a literal range or `countup`.
  let vars = n[0]
  if vars.kind notin {nnkSym, nnkIdent, nnkIdentDefs}:
    raiseAssert("tuple unpacking in a for loop")
  let iter = unwrapConv(n[1])
  var startNode, stopNode, stepNode: NimNode = nil
  var inclusive = true
  case iter.kind
  of nnkInfix:
    let op = iter[0].strVal
    if op == "..":
      inclusive = true
    elif op == "..<":
      inclusive = false
    else:
      raiseAssert("for loop over the iterator " & op & " is outside the supported set")
    startNode = iter[1]
    stopNode = iter[2]
  of nnkCall:
    let name = iter[0].strVal
    if name in ["..", "..<"]:
      inclusive = name == ".."
      startNode = iter[1]
      stopNode = iter[2]
    elif name == "items" and iter.len == 2 and iter[1].kind == nnkObjConstr and
        iter[1].getTypeInst().kind == nnkBracketExpr and
        $iter[1].getTypeInst()[0] == "Slice":
      # `..<` over a static bound produces a `Slice[int](a, b)`.
      # `items` iterates it inclusive, giving the equivalent range.
      inclusive = true
      for f in iter[1]:
        if f.kind == nnkExprColonExpr:
          if $f[0] == "a":
            startNode = f[1]
          elif $f[0] == "b":
            stopNode = f[1]
    elif name == "countup" or name == "countdown":
      if name == "countdown":
        raiseAssert("countdown loop: negative steps are outside the supported set")
      startNode = iter[1]
      stopNode = iter[2]
      if iter.len > 3:
        stepNode = iter[3]
    else:
      raiseAssert("for loop over the iterator " & name & " is unsupported, source: " & n.repr)
  else:
    raiseAssert("for loop over " & $iter.kind & " is outside the supported set")
  let varName = ctx.mintUniqueName($vars)
  let varType = ctx.resolver.resolveCirType(vars.getTypeInst())
  let startV = ctx.lowerExpr(unwrapConv(startNode))
  let stopV = ctx.lowerExpr(unwrapConv(stopNode))
  # Nim evaluates a for loop's stop bound once.
  # Bounds that contain statements re-run on every loop test.
  var boundV = stopV
  var boundPre: seq[CirStmt] = @[]
  if stopV.containsStmtsDeep():
    let boundName = ctx.mintUniqueName("stop_bound")
    let boundType = getExprType(stopV)
    let boundSym = sym(boundName, boundType)
    boundPre.add(varD(boundName, boundType))
    boundPre.add(asgnS(boundSym, stopV))
    boundV = boundSym
  var stepV: CirExpr = nil
  var stepIsConst = true
  if stepNode.isNil:
    stepV = intLit(1, varType)
  else:
    let unwrapped = unwrapConv(stepNode)
    if unwrapped.kind in nnkIntLit .. nnkUInt64Lit:
      if unwrapped.intVal <= 0:
        raiseAssert("for loop step " & $unwrapped.intVal & ": only positive steps are supported")
      stepV = ctx.lowerExpr(unwrapped)
    else:
      # no entry lowering: a step's calls must not run once before the loop
      # body ever executes, per-iteration lowering belongs to the incr branch
      stepIsConst = false
  # a loop variable's Nim scope is the for statement, so the declaration
  # and the body share one scope
  ctx.pushScope()
  ctx.declareLocal(vars.repr, sym(varName, varType))
  let varSym = sym(varName, varType)
  let body = ctx.lowerStmt(n[^1])
  ctx.popScope()
  let cmp = if inclusive:
    leE(varSym, boundV)
  else:
    ltE(varSym, boundV)
  var incr: seq[CirStmt] = @[]
  if stepIsConst:
    incr.add(asgnS(varSym, addE(varType, varSym, stepV)))
  else:
    # A step evaluated once at entry yields wrong iteration counts.
    let stepNow = ctx.lowerExpr(unwrapConv(stepNode))
    incr.add(asgnS(varSym, addE(varType, varSym, stepNow)))
  var preamble = @[varD(varName, varType)]
  preamble.add(asgnS(varSym, startV))
  result = preamble & boundPre & @[whileS(cmp, body & incr)]

func lowerAsmStmt(ctx: var CirFrontend, n: NimNode): seq[CirStmt] =
  # a Nim `asm` statement: verbatim template text alternating with backtick
  # operand symbols. The target printer splices the CIRv2 symbol names.
  var parts: seq[string]
  var operands: seq[CirExpr]
  parts.add("")
  for c in n:
    case c.kind
    of nnkEmpty, nnkCommentStmt:
      continue
    of nnkStrLit, nnkRStrLit, nnkTripleStrLit:
      parts[^1].add(c.strVal)
    of nnkSym:
      let op = ctx.lowerExpr(c)
      parts.add("")
      operands.add(op)
    else:
      raiseAssert("asm statement operand of kind " & $c.kind & " is outside the supported set")
  return @[asmS(parts, operands)]


# ═══════════════════════════════════════════════════════════════════════
#  Proc translation
# ═══════════════════════════════════════════════════════════════════════

func lowerProcBody(ctx: var CirFrontend, params: var seq[CirParam], formals: NimNode, body: NimNode): seq[CirStmt] =
  ctx.pushScope()
  for i in 1 ..< formals.len:
    let formal = formals[i]
    formal.expectKind(nnkIdentDefs)
    let ftype = formal[formal.len - 2]
    if isCompileTimeFormal(ftype) or ftype.repr == "typedesc":
      continue
    let typ = ctx.resolver.resolveCirType(ftype)
    for j in 0 ..< formal.len - 2:
      let pname = ctx.mintUniqueName(formal[j].strVal)
      let psym = sym(pname, typ)
      params.add(CirParam(name: pname, typ: typ,
        varParam: ftype.kind == nnkVarTy,
        passByRef: isLargeStruct(typ)))
      ctx.declareLocal(formal[j].repr, psym)
  let ret = ctx.locals.retType
  if ret.kind != ctVoid:
    # Nim lets a non-void proc declare its own `result`, and the uniquer suffixes that local past it
    ctx.locals.variablesInScope.incl "result"
    let rsym = sym("result", ret)
    ctx.declareLocal("result", rsym)
    result.add(varD("result", ret))
  result.add(ctx.lowerStmt(body))
  ctx.popScope()
  if result.len > 0 and result[^1].kind == sRet:
    discard
  elif ret.kind != ctVoid:
    result.add(retS(sym("result", ret)))
  else:
    result.add(retS(nil))

func enterProcBody(ctx: var CirFrontend, cirName: string, formals: NimNode): LocalScope =
  result = ctx.locals
  ctx.locals = LocalScope(layers: @[ctx.module.vars], fnName: cirName)
  # a local never takes a program-scope name, a claimed proc name,
  # a claim made while this body lowers, as the uniquer skips all three
  for g in ctx.module.vars.keys:
    ctx.locals.variablesInScope.incl g
  for n in ctx.module.procNameMap.keys:
    ctx.locals.variablesInScope.incl n
  ctx.locals.retType = ctx.resolver.resolveCirType(formals[0])

func exitProcBody(ctx: var CirFrontend, save: LocalScope) =
  ctx.locals = save

func lowerProcDef(ctx: var CirFrontend, fn: NimNode): CirProc =
  ## Generic definitions return nil here, their instantiations harvest at the calls that use them.
  if isGenericDef(fn):
    return nil
  let formals = getFormalParams(fn)
  let id = getProcIdentity(fn[0])
  # a kernel's own name is preserved, launchKernel is ABI on it
  let base = if getPragmaItem(fn, "global") != nil: sanitizeIdent(fn[0].strVal)
             else: mangleOperatorName(fn[0].strVal)
  let cirName = ctx.claimProcName(id, base)
  if id notin ctx.module.procRegistry:
    ctx.module.procRegistry[id] = ProcEntry(name: cirName)
  ctx.module.procRegistry[id].hasBody = true
  let save = ctx.enterProcBody(cirName, formals)
  var params: seq[CirParam] = @[]
  let fnRet = ctx.locals.retType
  let fnName = ctx.locals.fnName
  let body = ctx.lowerProcBody(params, formals, getProcBody(fn))
  result = procD(fnName, params, fnRet, body)
  var paramTypes: seq[CirType]
  for p in params:
    paramTypes.add(p.typ)
  ctx.module.procRegistry[id].fnType = tProcT(paramTypes, fnRet)
  let wgErr = fn.applyProcPragmas(result)
  if wgErr.len > 0:
    raiseAssert(wgErr)
  ctx.exitProcBody(save)
  ctx.module.procs.add(result)

func harvestProc(ctx: var CirFrontend, callee: NimNode, cirName: string) =
  ## `hasBody` is set before the body lowers, so a recursive call inside it
  ## finds the entry marked and never re-harvests.
  let id = getProcIdentity(callee)
  ctx.module.procRegistry[id].hasBody = true
  let impl = callee.getImpl()
  let formals = getFormalParams(callee.getTypeInst())
  let save = ctx.enterProcBody(cirName, formals)
  var params: seq[CirParam] = @[]
  let fnRet = ctx.locals.retType
  let body = ctx.lowerProcBody(params, formals, getProcBody(impl))
  ctx.exitProcBody(save)
  let harvested = procD(cirName, params, fnRet, body)
  let wgErr = impl.applyProcPragmas(harvested)
  if wgErr.len > 0:
    raiseAssert(wgErr)
  ctx.module.procs.add(harvested)

# ═══════════════════════════════════════════════════════════════════════
#  Entry points
# ═══════════════════════════════════════════════════════════════════════

proc lowerCirv2*(bodyIn: NimNode): CirModule =
  ## Lowers the typed macro body to CIRv2.
  ##
  ## `bodyIn` is a block of kernel proc defs, or the single `{.global.}` wrapper
  ## proc of a launchDeviceFn call. A refusal raises immediately.
  var ctx = CirFrontend()
  ctx.pushScope()
  var body = bodyIn
  block moduleWalk:
    if body.kind == nnkStmtListExpr and body.len == 1:
      # typed evaluator wraps a single-expression body in a one-child list
      body = body[0]
    if body.kind == nnkProcDef or body.kind == nnkFuncDef:
      discard ctx.lowerProcDef(body)
      break moduleWalk
    if body.kind == nnkSym:
      let impl = body.getImpl()
      if impl.kind notin {nnkProcDef, nnkFuncDef}:
        raiseAssert(body.strVal & " does not name a proc")
      discard ctx.lowerProcDef(impl)
      break moduleWalk
    if body.kind != nnkStmtList:
      raiseAssert("expected kernel proc definitions, got " & $body.kind)
    # kernels are ABI: every {.global.} def's own name is claimed before any body lowers,
    # a same-named device proc takes the suffix
    for c in body:
      if c.kind in {nnkProcDef, nnkFuncDef} and not isGenericDef(c) and
          getPragmaItem(c, "global") != nil:
        discard ctx.claimProcName(getProcIdentity(c[0]), sanitizeIdent(c[0].strVal))
    for c in body:
      case c.kind
      of nnkProcDef, nnkFuncDef:
        discard ctx.lowerProcDef(c)
      of nnkTypeSection:
        ctx.resolveTypeSection(c)
      of nnkVarSection, nnkConstSection:
        block topLevelSection:
          # Module-level variables: {.const_mem.} or {.smem.} picks the address space, an unannotated variable is constant storage
          # TODO: unannotated variables are read-only once thread-local globals
          # have a device representation
          for def in c:
            let nameN = def[0]
            let bareName = if nameN.kind == nnkPragmaExpr: nameN[0] else: nameN
            let name = bareName.repr
            let addressSpace =
              block:
                var spc = ctx.collectAddressSpace(nameN)
                if spc == caThread:
                  spc = caConstant
                spc
            let valN = def[2]
            let typN = if def[1].kind == nnkEmpty: valN.getTypeInst() else: def[1]
            let t = ctx.resolver.resolveCirType(typN, addressSpace)
            var init: CirExpr = nil
            if valN.kind != nnkEmpty:
              init = ctx.lowerExpr(valN)
            let g = varD(name, t, init, addressSpace)
            ctx.module.decls.add(g)
            ctx.module.vars[name] = (sym(name, t), addressSpace)
      of nnkCommentStmt:
        discard
      of nnkBindStmt, nnkMixinStmt:
        discard
      else:
        raiseAssert("top-level construct not supported: " & $c.kind & " " & c.repr)
  result = moduleD("kernel", ctx.resolver.resolvedStructs, ctx.module.procs)
  result.globals = ctx.module.decls

