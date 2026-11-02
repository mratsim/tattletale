# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../ir/cirv2_lang
import ../ir/cirv2_build
import nim_type_resolvers
import ../../builtins/builtins_catalog
import ../../builtins/builtins_magic_catalog
import ../ir/cirv2_pretty
import std/[macros, tables, strutils, sets]

# ############################################################
#
#            Nim frontend to CIRv2
#
# ############################################################

type
  ProcEntry = object
    ## Registry entry of one proc identity.
    name: string
    fnType: CirType
    hasBody: bool

  FrontendCtx = object
    resolver: CirTypeResolver
    fnName: string
    ## Name of the proc under translation, prefixed onto every refusal message.
    retType: CirType
    procs: seq[CirProc]
    procRegistry: Table[uint64, ProcEntry]
    ## Keyed by proc identity hash, one entry per distinct proc.
    claimedNames: Table[string, uint64]
    ## Emitted proc name to the owning identity: definitions and harvests share one name space, a second claimant of a base suffixes.
    scopes: seq[Table[string, CirExpr]]
    localSpaces: seq[Table[string, AddressSpace]]
    ## Nim symbol name to CIRv2 symbol, innermost scope last.
    moduleScope: Table[string, CirExpr]
    moduleSpaces: Table[string, AddressSpace]
    ## Program-scope declarations, visible from every proc body, a fresh proc's scope tables start from these.
    moduleGlobals: seq[CirStmt]
    ## Program-scope declarations in encounter order, `m.globals` of the module.
    nameCounts: HashSet[string]
    ## Emitted names already issued in the proc under translation:
    ## program-scope names, claimed proc names, a non-void proc's `result`.
    ## The uniquer suffixes past every entry, live claims included.
    hoisted: seq[CirStmt]
    ## Statements hoisted in front of the statement being lowered.

proc fail(ctx: var FrontendCtx, detail: string) =
  ## First refusal wins through the resolver's channel. The detail is prefixed
  ## with the proc under translation.
  ctx.resolver.fail("at " & ctx.fnName & ": " & detail)

proc drainHoisted(ctx: var FrontendCtx): seq[CirStmt] =
  result = ctx.hoisted
  ctx.hoisted = @[]

# ═══════════════════════════════════════════════════════════════════════
#  Names and scopes
# ═══════════════════════════════════════════════════════════════════════

proc mintUniqueName(ctx: var FrontendCtx, base: string): string =
  ## Per-proc unique CIRv2 name: the sanitized base, suffixed past every name
  ## already issued in the proc, user-declared or synthetic alike, so an issued
  ## name never doubles as a later declaration's emitted name.
  let clean = sanitizeIdent(base)
  var candidate = clean
  var suffix = 0
  while candidate in ctx.nameCounts or ctx.claimedNames.hasKey(candidate):
    inc suffix
    candidate = clean & "_" & $suffix
  ctx.nameCounts.incl candidate
  result = candidate

proc pushScope(ctx: var FrontendCtx) =
  ctx.scopes.add(initTable[string, CirExpr]())
  ctx.localSpaces.add(initTable[string, AddressSpace]())

proc popScope(ctx: var FrontendCtx) =
  ctx.scopes.del(ctx.scopes.high)
  ctx.localSpaces.del(ctx.localSpaces.high)

proc declareLocal(ctx: var FrontendCtx, key: string, sym: CirExpr, spc = caThread) =
  ## Registers a local under the caller's Nim lookup key in the innermost scope.
  ## Used for frontend-internal temporaries and for `result`.
  ##
  ## The key is the symbol's `repr`, pragma wrapper included:
  ## macro-generated locals share one `strVal` across expansion scopes,
  ## a gensym id inside `repr` tells them apart.
  ctx.scopes[^1][key] = sym
  ctx.localSpaces[^1][key] = spc

proc getPragmaItem(n: NimNode, name: string): NimNode =
  ## Pragma item of `name` in the node's pragma list or in the pragma list
  ## of the definition behind a symbol:
  ## - `{.builtin.}` of the builtins catalog
  ## - `{.magic: "..." .}` of the system operators
  ## - with or without a payload, nil when absent
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

proc hasPragmaNim(n: NimNode, name: string): bool =
  ## True when `name` appears in the node's pragma list or in the pragma list
  ## of the definition behind a symbol, bare or colon form.
  getPragmaItem(n, name) != nil

proc builtinKindPayload(n: NimNode): NimNode =
  ## Kind payload of a symbol's `{.builtin.}` pragma: the family enum
  ## member from `builtins_catalog.nim`. A payload-less `{.builtin.}`
  ## yields the nil literal, an absent pragma yields nil.
  let item = getPragmaItem(n, "builtin")
  if item != nil and item.kind in {nnkExprColonExpr, nnkCall} and item.len >= 2:
    return item[1]
  return nil

proc containsCall(n: NimNode): bool =
  ## True when a call occurs anywhere in the node, `StmtListExpr`-wrapped operands included.
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
  ## Wrappers: `HiddenStdConv` and the `StmtListExpr(Empty, x)` short-circuit operand form.
  result = n
  while true:
    case result.kind
    of nnkHiddenStdConv:
      if result.len >= 2:
        result = result[1]
      else:
        return
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

proc lowerExpr(ctx: var FrontendCtx, n: NimNode): CirExpr

proc resolveTypeSection(ctx: var FrontendCtx, n: NimNode) =
  ## Registers the section's concrete type definitions with the resolver.
  ## A generic definition registers nothing at declaration time,
  ## resolution of its instantiations happens at the use sites.
  for typeDef in n:
    typeDef.expectKind(nnkTypeDef)
    let head = typeDef[0]
    if head.kind == nnkBracketExpr or head.typeKind == ntyGenericBody:
      continue
    # a pragma-wrapped type name keeps the type node under the pragma
    let typeNode = if head.kind == nnkPragmaExpr: head[0] else: head
    discard ctx.resolver.resolveCirType(typeNode)

proc lowerStmt(ctx: var FrontendCtx, n: NimNode): seq[CirStmt]

proc builtinSymExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## A `{.builtin.}` value reference: the canonical coordinate vectors and the flat thread index,
  ## emitted as the symbol itself. The printers map the kind onto the target vocabulary.
  let name = sanitizeIdent(n.strVal)
  let typ = ctx.resolver.resolveCirType(n.getTypeInst())
  if ctx.resolver.err.len > 0:
    return nil
  result = sym(name, typ, builtin = true)
  result.coordBuiltin = coordBuiltinKindOf(builtinKindPayload(n))

# ═══════════════════════════════════════════════════════════════════════
#  Call translation
# ═══════════════════════════════════════════════════════════════════════

proc getFormalParams(fn: NimNode): NimNode =
  for i, c in fn:
    if c.kind == nnkFormalParams:
      return c
  return nil

proc resolveCallCallee(callee: NimNode): NimNode =
  ## Sem's pick: overload resolution ran before the macro, so a choice
  ## node's first candidate is the resolved symbol.
  if callee.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and callee.len > 0:
    return callee[0]
  return callee

proc getProcBody(fn: NimNode): NimNode =
  ## Statement list of a proc definition, boxing a single-statement body.
  ## Skips the trailing `result` symbol child value-returning procs carry.
  var i = fn.len - 1
  while i >= 0 and fn[i].kind == nnkSym:
    i.dec
  if i >= 0:
    if fn[i].kind == nnkStmtList:
      return fn[i]
    if fn[i].kind != nnkEmpty:
      return nnkStmtList.newTree(fn[i])
  return nil

proc isGenericDef(fn: NimNode): bool =
  for c in fn:
    if c.kind == nnkGenericParams:
      return true
    if c.kind == nnkBracket and c.len > 0 and c[0].kind == nnkEmpty:
      for g in c:
        if g.kind == nnkGenericParams:
          return true
  return false

func getProcIdentity(callee: NimNode): uint64 =
  ## Identity hash of a callee: the full 64-bit hash of the signature key,
  ## the repr and the signature hash. A generic adds the body hash,
  ## its instantiations share one signature.
  let impl = callee.getImpl()
  let generic = impl.kind in [nnkProcDef, nnkFuncDef] and
    (impl[2].kind == nnkGenericParams or
      (impl.len > 5 and impl[5].kind == nnkBracket and
        impl[5].len > 1 and impl[5][1].kind == nnkGenericParams))
  let key = callee.repr & "_" & callee.signatureHash() &
    (if generic: "_" & callee.symBodyHash() else: "")
  typeIdentityHash(key)

proc claimProcName(ctx: var FrontendCtx, identity: uint64, base: string): string =
  ## Emitted name of a proc identity: `base` on the first claim.
  ## A second claimant of the same base takes a base58 suffix of the identity hash.
  ## A second claimant rechecks the suffix by appending underscores until the identifier is unclaimed.
  ##
  ## A claim by the owning identity returns the base: the module reservation
  ## pass reserves kernel names, so a kernel's own def re-claims its name.
  if ctx.claimedNames.getOrDefault(base, identity) == identity:
    ctx.claimedNames[base] = identity
    return base
  result = base & "_" & base58Short(identity)
  while ctx.claimedNames.getOrDefault(result, identity) != identity:
    result.add("_")
  ctx.claimedNames[result] = identity

proc registerOrGet(ctx: var FrontendCtx, calleeSym: NimNode): ProcEntry =
  ## Registry entry of a call's callee.
  ## First registration derives the proc type once and claims the emitted name.
  ## Later calls read the entry.
  ##
  ## A `{.builtin.}` callee maps to its canonical native name at the call site. The entry stores its type only.
  let id = getProcIdentity(calleeSym)
  if ctx.procRegistry.hasKey(id):
    result = ctx.procRegistry[id]
    if result.fnType.isNil:
      # a definition registers before its body lowers, so the type is read
      # once at the first call inside it
      result.fnType = ctx.resolver.resolveCirType(calleeSym.getTypeInst())
      if ctx.resolver.err.len > 0:
        return
      ctx.procRegistry[id].fnType = result.fnType
    return
  let fnType = ctx.resolver.resolveCirType(calleeSym.getTypeInst())
  if ctx.resolver.err.len > 0:
    return
  let name = ctx.claimProcName(id,
    mangleOperatorName(calleeSym.strVal) & "_" & base58Short(id))
  result = ProcEntry(name: name, fnType: fnType)
  ctx.procRegistry[id] = result

proc harvestProc(ctx: var FrontendCtx, callee: NimNode, cirName: string)

proc lowerCallArgs(ctx: var FrontendCtx, callee: NimNode, call: NimNode): seq[CirExpr] =
  ## Lowers the runtime arguments of a call, erasing the compile-time parameters'
  ## arguments (their values live in the callee's types).
  ## Compile-time formals and call arguments align positionally.
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
    if ctx.resolver.err.len > 0:
      return @[]
    result.add(arg)

proc getNimMagicName(n: NimNode): string =
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
  ## True when the body renders as trivia only, a body with no runtime content.
  ## Trivia renders as `CommentStmt` and `Empty` children.
  for c in body:
    if c.kind notin {nnkCommentStmt, nnkEmpty}:
      return false
  return true

proc harvestedCallee(ctx: var FrontendCtx, calleeSym: NimNode): CirExpr =
  ## Callee symbol of a call to a harvested or builtin proc:
  ## - the registry entry registers or reads
  ## - a body-less callee harvests
  ## - a builtin maps to its canonical native name
  ##
  ## An instantiation hash would name a proc the backend never defines.
  ## A builtin callee reads its family kinds off the `{.builtin.}` pragma
  ## payload of its symbol at the ingest boundary:
  ## - every printer and pass dispatches on those kinds
  ## - an untagged builtin resolves every family at none
  ## - the conv and math vocabulary dispatches on the canonical name
  ##   through the registry tables
  let calleeBuiltin = hasPragmaNim(calleeSym, "builtin")
  let entry = ctx.registerOrGet(calleeSym)
  if ctx.resolver.err.len > 0:
    return nil
  if not calleeBuiltin and not entry.hasBody:
    # a call lowers 1:1 into the callee's body. A callee without a lowerable
    # body has no runtime content, a bodiless magic's documentation stub
    # renders exactly this way, so refuse and name the construct
    let body = getProcBody(calleeSym.getImpl())
    if body.isNil or isTriviaBody(body):
      ctx.fail("call to " & calleeSym.repr & " has no body to lower")
      return nil
    ctx.harvestProc(calleeSym, entry.name)
    if ctx.resolver.err.len > 0:
      return nil
  let calleeName = if calleeBuiltin: calleeSym.strVal
                   else: entry.name
  result = sym(calleeName, entry.fnType, calleeBuiltin)
  if calleeBuiltin:
    let payload = builtinKindPayload(calleeSym)
    result.synchroBuiltin = synchroBuiltinKindOf(payload)
    result.reductionBuiltin = reductionBuiltinKindOf(payload)
    result.shuffleBuiltin = shuffleBuiltinKindOf(payload)
    result.atomicsBuiltin = atomicsBuiltinKindOf(payload)
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

proc lowerCallValue(ctx: var FrontendCtx, callee: NimNode, call: NimNode): CirExpr =
  ## A value-producing call as a call expression: the use site nests the call
  ## into its operator tree.
  let magic = getNimMagicName(callee)
  # toOpenArray builds a span and len reads a span's length: the span split
  # pass consumes both forms, no proc harvests, no call emits
  let targetName = if callee.kind == nnkSym: callee.strVal
                   elif callee.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and callee.len > 0:
                     callee[0].strVal
                   else: ""
  if magic == "Slice" and targetName == "toOpenArray":
    # a `toOpenArray(base, first, last)` call lowers into the span construction
    # for the span split pass, the span type comes from the base pointer's
    # element type and space. The base must be a pointer: the split's pointer
    # side offsets the base by `first`, and CIRv2 declares pointer arithmetic over pointers
    let base = ctx.lowerExpr(call[1])
    if ctx.resolver.err.len > 0:
      return nil
    let first = ctx.lowerExpr(unwrapConv(call[2]))
    let last = ctx.lowerExpr(unwrapConv(call[3]))
    if ctx.resolver.err.len > 0:
      return nil
    let baseTyp = getExprType(base)
    if baseTyp.isNil or baseTyp.kind notin {ctPtr, ctAptr}:
      ctx.fail("toOpenArray over a " & (if baseTyp.isNil: "base with no derivable type" else: $baseTyp.kind) & " is outside the supported set, pass a pointer to the storage")
      return nil
    let firstTyp = getExprType(first)
    let lastTyp = getExprType(last)
    let spanTyp = tSpan(baseTyp.space, baseTyp.pElem)
    let callTyp = tProcT(baseTyp, firstTyp, lastTyp, spanTyp)
    return callE(sym("toOpenArray", callTyp), base, first, last)
  if magic == "LengthOpenArray":
    # a `len(s)` read over an openArray lowers to the `len` call
    # that the span split pass folds to the span's split length binding
    let arg = ctx.lowerExpr(call[1])
    if ctx.resolver.err.len > 0:
      return nil
    let argTyp = getExprType(arg)
    let readTyp = tProcT(argTyp, tInt(32))
    return callE(sym("len", readTyp), arg)
  # a catalog magic lowers to the primitive operation, nothing harvests
  if magic in NimGpuNumericBuiltinsOps:
    let args = ctx.lowerCallArgs(callee, call)
    if ctx.resolver.err.len > 0:
      return nil
    let opType = getExprType(args[0])
    return arithExpr(NimGpuNumericBuiltinsOps[magic], opType, args[0], args[1])
  let calleeSym = resolveCallCallee(callee)
  # `abs` over scalars lowers as a builtin call, printers use the native name.
  # Only `abs` intercepts: system `abs` over floats has a refused body.
  # `min`/`max` harvest into a ternary idiom.
  # An aggregate `abs` proc takes the harvest path below.
  if calleeSym.strVal == "abs":
    let fnType = ctx.resolver.resolveCirType(calleeSym.getTypeInst())
    if ctx.resolver.err.len > 0:
      return nil
    var scalar = true
    for p in fnType.params:
      if not isScalarType(p):
        scalar = false
        break
    if scalar:
      let args = ctx.lowerCallArgs(calleeSym, call)
      if ctx.resolver.err.len > 0:
        return nil
      return callE(sym(calleeSym.strVal, fnType, true), args)
  # a bodiless magic took the primitive path above
  # every other callee harvests, a magic with a Nim body (like `max`) lowers through it
  let calleeExpr = ctx.harvestedCallee(calleeSym)
  if ctx.resolver.err.len > 0:
    return nil
  let args = ctx.lowerCallArgs(calleeSym, call)
  if ctx.resolver.err.len > 0:
    return nil
  return callE(calleeExpr, args)

# ═══════════════════════════════════════════════════════════════════════
#  Expression translation
# ═══════════════════════════════════════════════════════════════════════

proc isBuiltinScalar(ctx: var FrontendCtx, n: NimNode): bool =
  let t = ctx.resolver.resolveCirType(n.getTypeInst())
  if ctx.resolver.err.len > 0:
    return false
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

proc lowerNestedTernary(ctx: var FrontendCtx,
    branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode, idx: int,
    resType: CirType): CirExpr =
  ## Branch `idx` of an if-expression as a ternary whose else variant
  ## contains the next branch, first match wins.
  ## `elseBody` is the innermost else variant.
  let cond = ctx.lowerExpr(branches[idx].cond)
  if ctx.resolver.err.len > 0:
    return nil
  let thenV = ctx.lowerExpr(getBranchValueExpr(branches[idx].body))
  if ctx.resolver.err.len > 0:
    return nil
  let elseV = if idx + 1 < branches.len:
                 lowerNestedTernary(ctx, branches, elseBody, idx + 1, resType)
               else:
                 ctx.lowerExpr(getBranchValueExpr(elseBody))
  if ctx.resolver.err.len > 0:
    return nil
  return ternaryE(resType, cond, thenV, elseV)

proc buildBranchBody(ctx: var FrontendCtx, bodyNode: NimNode, dest: CirExpr): seq[CirStmt] =
  ## One branch body: simple statements, or the expression form assigning
  ## `dest` from the branch's value.
  ## A branch body nests one scope: a branch-local shadows an outer local only
  ## inside the branch.
  ctx.pushScope()
  if dest.isNil:
    result = ctx.lowerStmt(bodyNode)
    ctx.popScope()
    return
  var body: NimNode = bodyNode
  if body.kind != nnkStmtList:
    body = nnkStmtList.newTree(bodyNode)
  var stmts: seq[CirStmt] = @[]
  var value: CirExpr = nil
  for i, c in body:
    if i == body.len - 1:
      value = ctx.lowerExpr(c)
    else:
      stmts.add(ctx.lowerStmt(c))
  if ctx.resolver.err.len > 0:
    ctx.popScope()
    return @[]
  ctx.popScope()
  return stmts & ctx.drainHoisted() & @[asgnS(dest, value)]

proc lowerIfForm(ctx: var FrontendCtx, branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode, dest: CirExpr): seq[CirStmt] =
  ## An elif chain lowers to one `ite` per branch, first match wins.
  ## Else statements sit innermost.
  ## Every branch body of the expression form assigns `dest`.
  var tail: seq[CirStmt] = @[]
  if not elseBody.isNil:
    tail = ctx.buildBranchBody(elseBody, dest)
    if ctx.resolver.err.len > 0:
      return @[]
  for i in countdown(branches.len - 1, 0):
    let cond = ctx.lowerExpr(branches[i].cond)
    if ctx.resolver.err.len > 0:
      return @[]
    # condition splices emit before the branch test, inside the else body
    # of the previous branch, so a splice runs only when the test runs
    let condHoists = ctx.drainHoisted()
    let body = ctx.buildBranchBody(branches[i].body, dest)
    if ctx.resolver.err.len > 0:
      return @[]
    tail = condHoists & @[iteS(cond, body, tail)]
  result = tail

proc lowerSplicedStmt(ctx: var FrontendCtx, c: NimNode): seq[CirStmt] =
  ## One non-final child of an expression block, a statement of the spliced expansion,
  ## hoisted in front of the enclosing statement.
  ## An object or tuple construction with no call inside has no runtime code
  ## and is dropped.
  if c.kind == nnkStmtList:
    for child in c:
      result.add(ctx.lowerSplicedStmt(child))
      if ctx.resolver.err.len > 0:
        return @[]
    return result
  if c.kind in {nnkObjConstr, nnkTupleConstr, nnkExprColonExpr, nnkBracketExpr,
                nnkPar, nnkIntLit .. nnkUInt64Lit, nnkFloatLit, nnkStrLit} and
      not containsCall(c):
    return @[]
  return ctx.lowerStmt(c)

proc lowerExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## Translates one expression node: the statements spliced by the expression
  ## sit in front of the enclosing statement.
  if ctx.resolver.err.len > 0:
    return nil
  if n.kind == nnkNilLit:
    # a nil literal: the literal node reports isNil too, this case runs
    # before the missing-node check
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.resolver.err.len > 0:
      return nil
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
      if hasPragmaNim(n, "builtin"):
        return ctx.builtinSymExpr(n)
      # a local resolves from the innermost scope outward
      var found: CirExpr = nil
      for i in countdown(ctx.scopes.high, 0):
        if ctx.scopes[i].hasKey(n.repr):
          found = ctx.scopes[i][n.repr]
          break
      if found.isNil:
        ctx.fail("unknown local symbol " & n.repr)
      return found
    of nskConst:
      if hasPragmaNim(n, "builtin"):
        return ctx.builtinSymExpr(n)
      let impl = n.getImpl()
      if impl.kind == nnkConstDef:
        return ctx.lowerExpr(impl[2])
    else:
      ctx.fail("symbol " & name & " (" & $n.symKind & ") is not a value")
      return nil
  of nnkIntLit .. nnkUInt64Lit:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.resolver.err.len > 0:
      return nil
    return intLit(n.intVal, typ)
  of nnkFloat32Lit, nnkFloat64Lit, nnkFloatLit:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.resolver.err.len > 0:
      return nil
    return fltLit(n.floatVal, typ)
  of nnkHiddenStdConv:
    # int/int32 share one bit width and the standard conversion is the identity
    # for values. A float literal sits at its default width: the conv node
    # reaches the expression-form pass, which re-types it to the target
    let inner = n[1]
    if inner.kind in {nnkFloat32Lit, nnkFloat64Lit, nnkFloatLit}:
      let typ = ctx.resolver.resolveCirType(n.getTypeInst())
      if ctx.resolver.err.len > 0:
        return nil
      return convE(typ, ctx.lowerExpr(inner))
    return ctx.lowerExpr(inner)
  of nnkHiddenSubConv, nnkConv:
    var typ = if n[0].kind == nnkEmpty:
      ctx.resolver.resolveCirType(n.getTypeInst())
    else:
      ctx.resolver.resolveCirType(n[0])
    if ctx.resolver.err.len > 0:
      return nil
    let operand = ctx.lowerExpr(n[1])
    if ctx.resolver.err.len > 0:
      return nil
    let ot = getExprType(operand)
    if ot != nil and sameType(ot, typ):
      # an identity conversion: the layout algebra's static wrappers stack up and the value already has the target type
      return operand
    if operand.kind in {eIntLit, eFloatLit}:
      # a literal conversion is exact, the expression-form pass re-types it
      # to the conversion target
      return convE(typ, operand)
    if typ.kind in {ctPtr, ctAptr}:
      # a pointer conversion preserves the operand's storage space and the cast target supplies the pointee type
      typ = typ.withAddressSpace(resolveAddressSpace(operand))
    return convE(typ, operand)
  of nnkCast:
    var typ = ctx.resolver.resolveCirType(n[0])
    if ctx.resolver.err.len > 0:
      return nil
    let operand = ctx.lowerExpr(n[1])
    if ctx.resolver.err.len > 0:
      return nil
    if typ.kind in {ctPtr, ctAptr}:
      # a pointer cast preserves the operand's storage space: the cast target
      # supplies the pointee type and the space resolves from the operand
      typ = typ.withAddressSpace(resolveAddressSpace(operand))
    return castE(typ, operand)
  of nnkHiddenDeref, nnkDerefExpr:
    let base = ctx.lowerExpr(n[0])
    if ctx.resolver.err.len > 0:
      return nil
    let bt = getExprType(base)
    if bt != nil and bt.kind == ctSpan:
      # a sem deref of a span lowers to the span itself, an index over it selects one element
      return base
    return derefE(base)
  of nnkHiddenAddr, nnkAddr:
    let base = ctx.lowerExpr(n[0])
    if ctx.resolver.err.len > 0:
      return nil
    let bt = getExprType(base)
    # address space: the memory where the base value lives
    # device storage yields a device pointer, a local a thread pointer
    var space = caThread
    var walk = base
    while true:
      case walk.kind
      of ePat, eAt:
        let next = if walk.kind == ePat: walk.patBase else: walk.atBase
        let ct = getExprType(next)
        if ct != nil and ct.kind in {ctPtr, ctAptr}:
          space = ct.space
          break
        # an array or struct base: the storage is the base's
        walk = next
      of eDeref:
        let ct = getExprType(walk.derefOperand)
        if ct != nil and ct.kind in {ctPtr, ctAptr}:
          space = ct.space
        break
      of eDot:
        walk = walk.dotBase
      of eSym:
        # a local's storage class comes from its declaration: a `{.smem.}` local
        # is threadgroup storage, an unannotated local is a register
        for i in countdown(ctx.localSpaces.high, 0):
          if ctx.localSpaces[i].hasKey(walk.name):
            space = ctx.localSpaces[i][walk.name]
            break
        break
      else:
        break
    return addrE(tPtr(space, bt), base)
  of nnkBracketExpr:
    let baseNode = n[0]
    var loweredBase: CirExpr = nil
    if baseNode.kind == nnkHiddenDeref:
      loweredBase = ctx.lowerExpr(baseNode[0])
    else:
      loweredBase = ctx.lowerExpr(baseNode)
    if ctx.resolver.err.len > 0:
      return nil
    let idx = ctx.lowerExpr(unwrapConv(n[1]))
    if ctx.resolver.err.len > 0:
      return nil
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
      # tuple-style `[i]` access over a named struct: the field of that index
      return dotE(loweredBase, bt.fields[idx.value].name)
    else:
      ctx.fail("indexing over type " & $(bt.kind) & " is outside the supported set")
      return nil
  of nnkDotExpr:
    let base = ctx.lowerExpr(n[0])
    if ctx.resolver.err.len > 0:
      return nil
    return dotE(base, sanitizeIdent(n[1].strVal))
  of nnkInfix:
    # operators: arithmetic and bitwise lower directly. Comparisons normalize into four kinds, `>` and `>=` swap operands
    let op = n[0].strVal
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
    if ctx.resolver.err.len > 0:
      return nil
    let b = ctx.lowerExpr(unwrapConv(n[2]))
    if ctx.resolver.err.len > 0:
      return nil
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
      ctx.fail("operator " & op & " is outside the supported set")
      return nil
  of nnkPrefix:
    let op = n[0].strVal
    let operand = ctx.lowerExpr(unwrapConv(n[1]))
    if ctx.resolver.err.len > 0:
      return nil
    case op
    of "-":
      return negE(getExprType(operand), operand)
    of "not":
      return notE(operand)
    else:
      ctx.fail("prefix operator " & op & " is outside the supported set")
      return nil
  of nnkCall, nnkCommand:
    let callee = n[0]
    if callee.kind notin {nnkSym, nnkOpenSymChoice, nnkClosedSymChoice}:
      ctx.fail("value call callee " & callee.repr & " is not a proc symbol")
      return nil
    return ctx.lowerCallValue(callee, n)
  of nnkObjConstr:
    var fields: seq[CirKv] = @[]
    var valueSpaces = initTable[string, AddressSpace]()
    for i in 1 ..< n.len:
      n[i].expectKind(nnkExprColonExpr)
      if n[i][1].kind == nnkEmpty:
        # semantic analysis yields an Empty node for a nil field value, the zero-fill pass supplies the typed nil
        continue
      let v = ctx.lowerExpr(n[i][1])
      if ctx.resolver.err.len > 0:
        return nil
      let fieldName = sanitizeIdent(n[i][0].strVal)
      # a pointer-typed field takes the address space its value proves
      valueSpaces[fieldName] = resolveAddressSpace(v)
      fields.add(kv(fieldName, v))
    let typ = ctx.resolver.resolveStructType(n.getTypeInst(), valueSpaces)
    if ctx.resolver.err.len > 0:
      return nil
    # Nim binds constructor values by name while the target syntax binds positionally.
    # Zero-fill reorders every constructor to the type's declaration order, omitted fields fill
    return oconstrE(typ, fields)
  of nnkExprColonExpr:
    # a named field outside a constructor: the identifier is compile-time
    # vocabulary of the expansion and the value is the runtime content
    return ctx.lowerExpr(n[1])
  of nnkTupleConstr:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.resolver.err.len > 0:
      return nil
    var fields: seq[CirKv] = @[]
    for i in 0 ..< n.len:
      let v = ctx.lowerExpr(n[i])
      if ctx.resolver.err.len > 0:
        return nil
      fields.add(kv("F" & $i, v))
    return oconstrE(typ, fields)
  of nnkBracket:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.resolver.err.len > 0:
      return nil
    var elems: seq[CirExpr] = @[]
    for el in n:
      let v = ctx.lowerExpr(el)
      if ctx.resolver.err.len > 0:
        return nil
      elems.add(v)
    return aconstrE(typ, elems)
  of nnkPar:
    if n.len == 1:
      return ctx.lowerExpr(n[0])
  of nnkStmtListExpr:
    var value: CirExpr = nil
    for i, c in n:
      if c.kind == nnkEmpty:
        continue
      if i == n.len - 1:
        value = ctx.lowerExpr(c)
      else:
        ctx.hoisted.add(ctx.lowerSplicedStmt(c))
        if ctx.resolver.err.len > 0:
          return nil
    return value
  of nnkBlockExpr, nnkBlockStmt:
    # a block expression splices its statements in front of the enclosing statement,
    # its value being the last expression.
    let inner = n[^1]
    if inner.kind != nnkStmtList:
      # a single-statement body lowers unwrapped and the block value is the expression itself,
      # matching the shape `getProcBody` boxes for proc bodies
      return ctx.lowerExpr(inner)
    var value: CirExpr = nil
    for i, c in inner:
      if i == inner.len - 1 and c.kind notin {nnkAsgn, nnkIfStmt, nnkForStmt,
          nnkWhileStmt, nnkDiscardStmt, nnkReturnStmt}:
        value = ctx.lowerExpr(c)
      else:
        ctx.hoisted.add(ctx.lowerSplicedStmt(c))
        if ctx.resolver.err.len > 0:
          return nil
    return value
  of nnkIfExpr:
    # an if-expression: single-expression branches nest into conditional values,
    # statement-bearing branches hoist a temp that the statement form assigns
    let resType = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.resolver.err.len > 0:
      return nil
    var branches: seq[tuple[cond: NimNode, body: NimNode]] = @[]
    var elseBody: NimNode = nil
    for c in n:
      case c.kind
      of nnkElifBranch, nnkElifExpr:
        # statement-shaped elif branches reach the expression form
        # when the branch values sit in value position
        branches.add((cond: c[0], body: c[1]))
      of nnkElse, nnkElseExpr:
        elseBody = c[0]
      else:
        ctx.fail("if expression: unexpected branch " & $c.kind)
        return nil
    # a branch nests into a ternary value when it is a single expression,
    # statement-bearing branches take the statement form
    var simple = true
    for br in branches:
      if isStmtBody(getBranchValueExpr(br.body)):
        simple = false
        break
    if simple and not isStmtBody(getBranchValueExpr(elseBody)):
      return lowerNestedTernary(ctx, branches, elseBody, 0, resType)
    let tmpName = ctx.mintUniqueName("iftmp")
    ctx.declareLocal(tmpName, sym(tmpName, resType))
    let stmts = ctx.lowerIfForm(branches, elseBody, sym(tmpName, resType))
    if ctx.resolver.err.len > 0:
      return nil
    ctx.hoisted.add(@[varD(tmpName, resType)] & stmts)
    return sym(tmpName, resType)
  of nnkStrLit, nnkRStrLit, nnkTripleStrLit:
    # a string literal is C string text: nnkStrLit escapes it, the raw string
    # forms carry their text verbatim, the printers add the quotes
    let text = if n.kind == nnkStrLit: n.strVal.escape("", "") else: n.strVal
    return strLit(text)
  else:
    ctx.fail("expression construct not supported: " & $n.kind & " " & n.repr)
    return nil

# ═══════════════════════════════════════════════════════════════════════
#  Statements
# ═══════════════════════════════════════════════════════════════════════

proc lowerVoidCall(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
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
    ctx.fail("call callee " & callee.repr & " is not a proc symbol")
    return @[]
  let calleeSym = resolveCallCallee(callee)
  let calleeExpr = ctx.harvestedCallee(calleeSym)
  if ctx.resolver.err.len > 0:
    return @[]
  let args = ctx.lowerCallArgs(calleeSym, n)
  if ctx.resolver.err.len > 0:
    return @[]
  return ctx.drainHoisted() & @[callS(calleeExpr, args)]

proc collectAddressSpace(ctx: var FrontendCtx, nameNode: NimNode): AddressSpace =
  ## Memory class of a local declaration, decided by the pragma list:
  ## `smem` means threadgroup, `const_mem` constant, `rmem` or unannotated registers.
  result = caThread
  if nameNode.kind != nnkPragmaExpr:
    return
  var seen = false
  template setSpace(spc: AddressSpace) =
    if seen:
      ctx.fail("multiple address-space pragmas on one local")
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
      ctx.fail("unexpected pragma " & p.repr & " on a local declaration")

proc lowerStmtBody(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## Lowers one Nim statement body to CIRv2 statements in source order.
  ## Hoisted statements drained from the context prepend to the body's own statements.
  if ctx.resolver.err.len > 0:
    return @[]
  if n.isNil:
    return @[]
  case n.kind
  of nnkCommentStmt, nnkEmpty:
    # an expansion hole or a comment, no code either way
    return @[]
  of nnkStmtList:
    for c in n:
      result.add(ctx.lowerStmt(c))
      if ctx.resolver.err.len > 0:
        return @[]
  of nnkLetSection, nnkVarSection:
    # a `let`/`var` section: one typed declaration per name, a call-valued initializer lowers into an `asgn` after the declaration.
    # memory class comes from the declaration's pragma list.
    for defs in n:
      let typeNode = defs[^2]
      let initNode = defs[^1]
      for j in 0 ..< defs.len - 2:
        let nameNode = defs[j]
        let spc = ctx.collectAddressSpace(nameNode)
        if ctx.resolver.err.len > 0:
          return @[]
        let nimName = if nameNode.kind == nnkPragmaExpr: nameNode[0].strVal
                      else: nameNode.strVal
        var typ: CirType = nil
        if typeNode.kind != nnkEmpty:
          typ = ctx.resolver.resolveCirType(typeNode)
        else:
          typ = ctx.resolver.resolveCirType(initNode.getTypeInst())
        if ctx.resolver.err.len > 0:
          return @[]
        var value: CirExpr = nil
        if initNode.kind != nnkEmpty:
          value = ctx.lowerExpr(initNode)
          if ctx.resolver.err.len > 0:
            return @[]
          if value.kind == eOconstr and typ != nil and typ.kind == ctStruct and
              value.ocType.kind == ctStruct:
            # a constructor value is typed by the struct it instantiates,
            # its declaration uses that type, and the field address spaces
            # come from the value's type
            typ = value.ocType
          elif typ != nil and typ.kind in {ctPtr, ctAptr} and
              getExprType(value) != nil and getExprType(value).kind == typ.kind and
              getExprType(value).space != typ.space:
            # a pointer cast preserves the operand's address space:
            # - the declaration takes the value's space
            # - `let q = cast[ptr T](addr scratch[0])` is a threadgroup pointer
            typ = getExprType(value)
        let hoists = ctx.drainHoisted()
        let cirName = ctx.mintUniqueName(nimName)
        # lookup key: `repr` of the declaration, pragma wrapper included
        let lookupKey = if nameNode.kind == nnkPragmaExpr: nameNode[0].repr
                        else: nameNode.repr
        ctx.declareLocal(lookupKey, sym(cirName, typ), spc)
        if value.isNil:
          result.add(hoists & @[varD(cirName, typ, nil, spc)])
        else:
          result.add(hoists & @[varD(cirName, typ, value, spc)])
  of nnkConstSection:
    # a const section emits no runtime code: the resolver registers the value's struct types
    for defs in n:
      defs.expectKind(nnkConstDef)
      let typeNode = defs[1]
      let valueNode = defs[2]
      if typeNode.kind != nnkEmpty:
        discard ctx.resolver.resolveCirType(typeNode)
      else:
        discard ctx.resolver.resolveCirType(valueNode.getTypeInst())
      if ctx.resolver.err.len > 0:
        return @[]
  of nnkAsgn:
    let lhs = ctx.lowerExpr(n[0])
    if ctx.resolver.err.len > 0:
      return @[]
    let rhs = ctx.lowerExpr(n[1])
    if ctx.resolver.err.len > 0:
      return @[]
    return ctx.drainHoisted() & @[asgnS(lhs, rhs)]
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
      if ctx.resolver.err.len > 0:
        # operand type resolution failed, return with the pending failure,
        # a statement-position call path needs a clean channel
        return @[]
      let lp = if lt != nil and lt.kind == ctPtr: lt.pElem else: lt
      if isScalarType(lp) and isScalarType(rt):
        var lhs = ctx.lowerExpr(n[1])
        if ctx.resolver.err.len > 0:
          return @[]
        # this lowering dereferences a var-returning `[]` call's element address, and the assignment target is the location behind it
        if lhs.kind == eAddr:
          lhs = lhs.addrOperand
        elif lhs.kind == eSym and getExprType(lhs) != nil and
            getExprType(lhs).kind == ctPtr:
          # a var parameter's location lives in a pointer and the compound assignment operates on the pointee
          lhs = derefE(lhs)
        let rhs = ctx.lowerExpr(unwrapConv(n[2]))
        if ctx.resolver.err.len > 0:
          return @[]
        let opType = getExprType(lhs)
        # target and arithmetic share one tree reference
        # a desugared `op=` left-hand side evaluates exactly once in Nim
        # temp-binding normalization splits the reads of a call-bearing target
        let arith = arithExpr(op[0 ..< op.len - 1], opType, lhs, rhs)
        return ctx.drainHoisted() & @[asgnS(lhs, arith)]
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
        ctx.fail("if statement: unexpected child " & $c.kind)
        return @[]
    return ctx.lowerIfForm(branches, elseBody, nil)
  of nnkWhileStmt:
    # statements spliced by the condition drain to the front of the loop body,
    # a splice runs once per iteration after the test that admits it
    let cond = ctx.lowerExpr(n[0])
    if ctx.resolver.err.len > 0:
      return @[]
    ctx.pushScope()
    let body = ctx.lowerStmt(n[1])
    ctx.popScope()
    if ctx.resolver.err.len > 0:
      return @[]
    return @[whileS(cond, body)]
  of nnkForStmt:
    # a `for` over a literal range or `countup`: the loop variable is declared up front, bounds evaluate once at entry and the increment runs last
    let vars = n[0]
    if vars.kind notin {nnkSym, nnkIdent, nnkIdentDefs}:
      ctx.fail("tuple unpacking in a for loop")
      return @[]
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
        ctx.fail("for loop over the iterator " & op & " is outside the supported set")
        return @[]
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
          ctx.fail("countdown loop: negative steps are outside the supported set")
          return @[]
        startNode = iter[1]
        stopNode = iter[2]
        if iter.len > 3:
          stepNode = iter[3]
      else:
        ctx.fail("for loop over the iterator " & name & " is unsupported, source: " & n.repr)
        return @[]
    else:
      ctx.fail("for loop over " & $iter.kind & " is outside the supported set")
      return @[]
    let varName = ctx.mintUniqueName($vars)
    let varType = ctx.resolver.resolveCirType(vars.getTypeInst())
    if ctx.resolver.err.len > 0:
      return @[]
    let startV = ctx.lowerExpr(unwrapConv(startNode))
    if ctx.resolver.err.len > 0:
      return @[]
    let stopV = ctx.lowerExpr(unwrapConv(stopNode))
    if ctx.resolver.err.len > 0:
      return @[]
    var stepV: CirExpr = nil
    var stepIsConst = true
    if stepNode.isNil:
      stepV = intLit(1, varType)
    else:
      let unwrapped = unwrapConv(stepNode)
      if unwrapped.kind in nnkIntLit .. nnkUInt64Lit:
        if unwrapped.intVal <= 0:
          ctx.fail("for loop step " & $unwrapped.intVal & ": only positive steps are supported")
          return @[]
        stepV = ctx.lowerExpr(unwrapped)
      else:
        # no entry lowering: a step's calls must not run once before the loop
        # body ever executes, per-iteration lowering belongs to the incr branch
        stepIsConst = false
    if ctx.resolver.err.len > 0:
      return @[]
    # a loop variable's Nim scope is the for statement, so the declaration
    # and the body share one scope
    ctx.pushScope()
    ctx.declareLocal(vars.repr, sym(varName, varType))
    let varSym = sym(varName, varType)
    # start/stop/step hoists queued so far belong to the preamble, while the body's own hoists drain per statement inside it
    let startHoists = ctx.drainHoisted()
    let body = ctx.lowerStmtBody(n[^1])
    if ctx.resolver.err.len > 0:
      ctx.popScope()
      return @[]
    let bodyHoists = ctx.drainHoisted()
    ctx.popScope()
    let cmp = if inclusive:
      leE(varSym, stopV)
    else:
      ltE(varSym, stopV)
    var incr: seq[CirStmt] = @[]
    if stepIsConst:
      incr.add(asgnS(varSym, addE(varType, varSym, stepV)))
    else:
      # loop-variant step expressions lower inside the loop,
      # they re-evaluate at the continue point.
      # A step evaluated once at entry yields wrong iteration counts.
      let stepNow = ctx.lowerExpr(unwrapConv(stepNode))
      if ctx.resolver.err.len > 0:
        return @[]
      incr.add(ctx.drainHoisted() & @[asgnS(varSym, addE(varType, varSym, stepNow))])
    var preamble = startHoists
    preamble.add(varD(varName, varType))
    preamble.add(asgnS(varSym, startV))
    result = preamble & @[whileS(cmp, bodyHoists & body & incr)]
  of nnkReturnStmt:
    var valueNode = n[0]
    if valueNode.kind == nnkAsgn:
      valueNode = valueNode[1]
    if valueNode.kind == nnkEmpty:
      if ctx.retType.kind != ctVoid:
        return @[retS(sym("result", ctx.retType))]
      return @[retS(nil)]
    let value = ctx.lowerExpr(valueNode)
    if ctx.resolver.err.len > 0:
      return @[]
    return ctx.drainHoisted() & @[retS(value)]
  of nnkDiscardStmt:
    if n[0].kind == nnkEmpty:
      return @[discardS(nil)]
    if n[0].kind in {nnkCall, nnkCommand}:
      return ctx.lowerVoidCall(n[0])
    let value = ctx.lowerExpr(n[0])
    if ctx.resolver.err.len > 0:
      return @[]
    return ctx.drainHoisted() & @[discardS(value)]
  of nnkCall, nnkCommand:
    return ctx.lowerVoidCall(n)
  of nnkAsmStmt:
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
        if ctx.resolver.err.len > 0:
          return @[]
        parts.add("")
        operands.add(op)
      else:
        ctx.fail("asm statement operand of kind " & $c.kind & " is outside the supported set")
        return @[]
    return ctx.drainHoisted() & @[asmS(parts, operands)]
  of nnkBlockStmt:
    # a `break label` inside the body is refused at the break statement.
    # A label no break targets is dead and lowers as its body
    ctx.pushScope()
    result = ctx.lowerStmt(n[1])
    ctx.popScope()
  of nnkBreakStmt:
    ctx.fail("break: unsupported in device code, restructure the loop condition")
    return @[]
  of nnkContinueStmt:
    ctx.fail("continue: unsupported in device code, restructure the loop condition")
    return @[]
  of nnkCaseStmt:
    ctx.fail("case statement: outside the supported set")
    return @[]
  of nnkTypeSection:
    ctx.resolveTypeSection(n)
    return @[]
  of nnkPragma:
    var markUnroll = false
    var emitArgs: seq[NimNode] = @[]
    for p in n:
      if p.kind in {nnkIdent, nnkSym} and p.strVal == "unroll":
        # Unroll marking consumes this marker statement: the loop whose body
        # states it gets the unroll flag
        markUnroll = true
      elif p.kind in {nnkExprColonExpr, nnkCall} and p.len >= 2 and
          p[0].kind in {nnkIdent, nnkSym} and p[0].strVal == "emit":
        emitArgs.add(p[1])
    var emitted: seq[CirStmt] = @[]
    if markUnroll:
      emitted.add(emitS(@["{.unroll.}"], @[]))
    if emitted.len == 0 and emitArgs.len == 0:
      # a non-emit statement pragma is a compile-time annotation, nothing emits
      return @[]
    for emitArg in emitArgs:
      # literal runs alternate with spliced expressions: a computed splice
      # stores through a fresh temp that the emitted text references by name
      if emitArg.kind in {nnkStrLit, nnkRStrLit, nnkTripleStrLit}:
        emitted.add(ctx.drainHoisted() & @[emitS(@[emitArg.strVal], @[])])
        continue
      var parts: seq[string] = @[""]
      var operands: seq[CirExpr] = @[]
      for part in emitArg:
        case part.kind
        of nnkStrLit, nnkRStrLit, nnkTripleStrLit:
          parts[^1].add(part.strVal)
        else:
          let e = ctx.lowerExpr(part)
          if ctx.resolver.err.len > 0:
            return @[]
          var op = e
          if e.kind != eSym:
            # printers splice symbol names, a computed value needs a temp
            let tmpName = ctx.mintUniqueName("emit_tmp")
            let tmpType = getExprType(e)
            op = sym(tmpName, tmpType)
            ctx.hoisted.add(varD(tmpName, tmpType))
            ctx.hoisted.add(asgnS(op, e))
            ctx.declareLocal(tmpName, op)
          parts.add("")
          operands.add(op)
      emitted.add(ctx.drainHoisted() & @[emitS(parts, operands)])
    return emitted
  of nnkTemplateDef, nnkMacroDef, nnkIteratorDef:
    # a definition left in a typed body is a compile-time artifact:
    # templates and macros execute at expansion, iterators inline into
    # their call sites, an unused definition has no runtime code
    return @[]
  of nnkProcDef, nnkFuncDef, nnkConverterDef, nnkMethodDef:
    ctx.fail("nested proc definition " & n.repr)
    return @[]
  of nnkObjConstr, nnkTupleConstr, nnkBracketExpr, nnkPar:
    # a discarded construction step of a macro expansion drops its value:
    # this lowering keeps the side effects only, as lowerSplicedStmt does for the spliced statements of an expression
    discard ctx.lowerExpr(n)
    if ctx.resolver.err.len > 0:
      return @[]
    return ctx.drainHoisted()
  of nnkExprColonExpr:
    # a discarded named-field step of a macro expansion whose value is never read
    discard ctx.lowerExpr(n[1])
    if ctx.resolver.err.len > 0:
      return @[]
    return ctx.drainHoisted()
  of nnkTryStmt, nnkRaiseStmt:
    ctx.fail($n.kind & ": exceptions are outside the supported set")
    return @[]
  of nnkBindStmt, nnkMixinStmt:
    # symbol-binding directives, Nim resolves them at sem time, no trace remains in the lowered code
    return @[]
  else:
    ctx.fail("statement construct not supported: " & $n.kind & " " & n.repr)
    return @[]

proc lowerStmt(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  let body = ctx.lowerStmtBody(n)
  if ctx.resolver.err.len > 0:
    return @[]
  return ctx.drainHoisted() & body

# ═══════════════════════════════════════════════════════════════════════
#  Proc translation
# ═══════════════════════════════════════════════════════════════════════

proc lowerProcBody(ctx: var FrontendCtx, fn: NimNode, formals: NimNode, body: NimNode, params: var seq[CirParam]): seq[CirStmt] =
  ## Shared proc body lowering:
  ## - the formal parameters are declared into the proc's root scope
  ## - the `result` variable exists for a non-void proc
  ## - an implicit `ret` closes a fall-through body
  ctx.pushScope()
  for i in 1 ..< formals.len:
    let formal = formals[i]
    formal.expectKind(nnkIdentDefs)
    let ftype = formal[formal.len - 2]
    if isCompileTimeFormal(ftype) or ftype.repr == "typedesc":
      continue
    let typ = ctx.resolver.resolveCirType(ftype)
    if ctx.resolver.err.len > 0:
      ctx.popScope()
      return @[]
    for j in 0 ..< formal.len - 2:
      let pname = ctx.mintUniqueName(formal[j].strVal)
      let psym = sym(pname, typ)
      params.add(CirParam(name: pname, typ: typ,
        varParam: ftype.kind == nnkVarTy,
        passByRef: isLargeStruct(typ)))
      ctx.declareLocal(formal[j].repr, psym)
  let ret = ctx.retType
  if ret.kind != ctVoid:
    # Nim lets a non-void proc declare its own `result`, and the uniquer suffixes that local past it
    ctx.nameCounts.incl "result"
    let rsym = sym("result", ret)
    ctx.declareLocal("result", rsym)
    result.add(varD("result", ret))
  result.add(ctx.lowerStmt(body))
  ctx.popScope()
  if ctx.resolver.err.len > 0:
    return @[]
  # a fall-through body takes the fallback return, an explicit trailing
  # return already ends the proc
  if result.len > 0 and result[^1].kind == sRet:
    discard
  elif ret.kind != ctVoid:
    result.add(retS(sym("result", ret)))
  else:
    result.add(retS(nil))

type ProcCtxSave = object
  ## Snapshot of the enclosing proc context that a body lowering replaces
  ## and restores on exit.
  name: string
  ret: CirType
  counts: HashSet[string]
  scopes: seq[Table[string, CirExpr]]
  spaces: seq[Table[string, AddressSpace]]

proc enterProcBody(ctx: var FrontendCtx, cirName: string, formals: NimNode): ProcCtxSave =
  ## Seeds a body's state: a fresh issued-name set covering program-scope
  ## names and claimed proc names, a root scope over the module scope,
  ## and the return type resolved from the formals.
  result = ProcCtxSave(name: ctx.fnName, ret: ctx.retType,
    counts: ctx.nameCounts, scopes: ctx.scopes, spaces: ctx.localSpaces)
  ctx.fnName = cirName
  ctx.nameCounts = initHashSet[string]()
  # a local never takes a program-scope name, a claimed proc name,
  # a claim made while this body lowers, as the uniquer skips all three
  for g in ctx.moduleScope.keys:
    ctx.nameCounts.incl g
  for n in ctx.claimedNames.keys:
    ctx.nameCounts.incl n
  ctx.scopes = @[ctx.moduleScope]
  ctx.localSpaces = @[ctx.moduleSpaces]
  ctx.retType = ctx.resolver.resolveCirType(formals[0])

proc exitProcBody(ctx: var FrontendCtx, save: ProcCtxSave) =
  ctx.fnName = save.name
  ctx.retType = save.ret
  ctx.nameCounts = save.counts
  ctx.scopes = save.scopes
  ctx.localSpaces = save.spaces

proc lowerProcDef(ctx: var FrontendCtx, fn: NimNode): CirProc =
  ## One proc definition: parameters typed off Nim once and the body lowered
  ## node for node.
  ##
  ## Generic definitions return nil here, their instantiations harvest at the calls that use them.
  ## A def emits under its own name: kernels keep their own name,
  ## other defs mangle the operator name, a claim conflict suffixes.
  ## A call-site harvest never claims an undecorated name.
  if ctx.resolver.err.len > 0:
    return nil
  if isGenericDef(fn):
    return nil
  let formals = getFormalParams(fn)
  let id = getProcIdentity(fn[0])
  # a kernel keeps its own name, launchKernel is ABI on it
  let base = if hasPragmaNim(fn, "global"): sanitizeIdent(fn[0].strVal)
             else: mangleOperatorName(fn[0].strVal)
  let cirName = ctx.claimProcName(id, base)
  if id notin ctx.procRegistry:
    ctx.procRegistry[id] = ProcEntry(name: cirName)
  ctx.procRegistry[id].hasBody = true
  let save = ctx.enterProcBody(cirName, formals)
  if ctx.resolver.err.len > 0:
    return nil
  var params: seq[CirParam] = @[]
  let fnRet = ctx.retType
  let fnName = ctx.fnName
  let body = ctx.lowerProcBody(fn, formals, getProcBody(fn), params)
  result = procD(fnName, params, fnRet, body)
  var paramTypes: seq[CirType]
  for p in params:
    paramTypes.add(p.typ)
  ctx.procRegistry[id].fnType = tProcT(paramTypes, fnRet)
  let wgErr = fn.applyProcPragmas(result)
  if wgErr.len > 0:
    ctx.fail(wgErr)
  ctx.exitProcBody(save)
  if ctx.resolver.err.len > 0:
    return nil
  ctx.procs.add(result)

proc harvestProc(ctx: var FrontendCtx, callee: NimNode, cirName: string) =
  ## Lowers the body of a callee a call registered.
  ## The registry entry gives the emitted name and the proc type, compile-time parameters erased.
  ## `hasBody` set before the body lowers guards the recursion, callers harvest only body-less entries.
  let id = getProcIdentity(callee)
  ctx.procRegistry[id].hasBody = true
  let impl = callee.getImpl()
  let formals = getFormalParams(callee.getTypeInst())
  let savedProcsLen = ctx.procs.len
  let savedHoisted = ctx.hoisted
  # a callee body is lowered under its own hoist queue: the caller's pending hoists belong to the caller's statement, never to the callee's body
  ctx.hoisted = @[]
  let save = ctx.enterProcBody(cirName, formals)
  if ctx.resolver.err.len > 0:
    return
  var params: seq[CirParam] = @[]
  let fnRet = ctx.retType
  let body = ctx.lowerProcBody(impl, formals, getProcBody(impl), params)
  ctx.exitProcBody(save)
  # pending hoists of the callee append to its body first and the caller's own queue resumes after
  let calleeHoists = ctx.drainHoisted()
  ctx.hoisted = savedHoisted
  if ctx.resolver.err.len > 0:
    ctx.procs.setLen(savedProcsLen)
    return
  let harvested = procD(cirName, params, fnRet, calleeHoists & body)
  let wgErr = impl.applyProcPragmas(harvested)
  if wgErr.len > 0:
    ctx.fail(wgErr)
    # nested harvests inside the callee body roll back with it
    ctx.procs.setLen(savedProcsLen)
    return
  ctx.procs.add(harvested)

# ═══════════════════════════════════════════════════════════════════════
#  Entry points
# ═══════════════════════════════════════════════════════════════════════

proc lowerCirv2*(bodyIn: NimNode): tuple[m: CirModule, err: string] =
  ## Lowers the typed macro body to CIRv2.
  ##
  ## Returns:
  ##   the module and "" when clean, else the first refusal message.
  ##
  ## `bodyIn` is a block of kernel proc defs, or the single `{.global.}` wrapper
  ## proc of a launchDeviceFn call.
  var ctx = FrontendCtx()
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
        ctx.fail(body.strVal & " does not name a proc")
        break moduleWalk
      discard ctx.lowerProcDef(impl)
      break moduleWalk
    if body.kind != nnkStmtList:
      ctx.fail("expected kernel proc definitions, got " & $body.kind)
      break moduleWalk
    # kernels are ABI: every {.global.} def's own name is claimed before any body lowers,
    # a same-named device proc takes the suffix
    for c in body:
      if c.kind in {nnkProcDef, nnkFuncDef} and not isGenericDef(c) and
          hasPragmaNim(c, "global"):
        discard ctx.claimProcName(getProcIdentity(c[0]), sanitizeIdent(c[0].strVal))
    for c in body:
      case c.kind
      of nnkProcDef, nnkFuncDef:
        discard ctx.lowerProcDef(c)
      of nnkTypeSection:
        ctx.resolveTypeSection(c)
      of nnkVarSection, nnkConstSection:
        # program-scope storage whose pragma list picks the memory class,
        # unannotated is constant. Kernels must appear after the declarations they read
        block topLevelSection:
          for def in c:
            let nameN = def[0]
            let bareName = if nameN.kind == nnkPragmaExpr: nameN[0] else: nameN
            let name = bareName.repr
            let space =
              block:
                var spc = ctx.collectAddressSpace(nameN)
                if spc == caThread:
                  # program scope has no per-thread storage, unannotated is constant
                  spc = caConstant
                spc
            let valN = def[2]
            let typN = if def[1].kind == nnkEmpty: valN.getTypeInst() else: def[1]
            let t = ctx.resolver.resolveCirType(typN, space)
            if ctx.resolver.err.len > 0:
              break topLevelSection
            var init: CirExpr = nil
            if valN.kind != nnkEmpty:
              init = ctx.lowerExpr(valN)
              if ctx.resolver.err.len > 0:
                break topLevelSection
            let g = varD(name, t, init, space)
            ctx.moduleGlobals.add(g)
            let symE = sym(name, t)
            ctx.moduleScope[name] = symE
            ctx.moduleSpaces[name] = space
      of nnkCommentStmt:
        discard
      of nnkBindStmt, nnkMixinStmt:
        # symbol-binding directives, Nim resolves them at sem time
        discard
      else:
        ctx.fail("top-level construct not supported: " & $c.kind & " " & c.repr)
        break moduleWalk
  let m = moduleD("kernel", ctx.resolver.structs, ctx.procs)
  m.globals = ctx.moduleGlobals
  return (m, ctx.resolver.err)

