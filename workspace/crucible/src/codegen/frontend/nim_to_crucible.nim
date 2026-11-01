# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cirv2 frontend: typed NimNode to CIRv2, normal form by construction.
##
## Entry: the typed node the `metal:`-family macros receive, a block of kernel
## proc defs or the single `{.global.}` wrapper proc of a launchDeviceFn call.
## Types translate exactly once through `nim_type_resolvers`.
##
## Procs intern through the frontend's proc registry: one entry per proc identity, its emitted name claimed at first registration.
##
## Constructs outside the supported closure refuse through the pure refusal channel,
## naming the construct: the macro wrapper turns the refusal into
## a compile-time `error`.

import ../ir/cirv2_lang
import ../ir/cirv2_build
import nim_type_resolvers
import ../ir/cirv2_pretty
import std/[macros, tables, strutils]

type
  ProcEntry = object
    ## Registry entry of one proc identity: the emitted name is a printed string.
    ## Identity stays the registry's int key.
    name: string
    fnType: CirType
    hasBody: bool

  FrontendCtx = object
    resolver: CirTypeResolver
    fnName: string
    ## Name of the proc under translation, names every refusal.
    retType: CirType
    procs: seq[CirProc]
    procRegistry: Table[uint64, ProcEntry]
    ## Keyed by proc identity hash: emitted name, proc type, and body state.
    ## One entry per distinct proc, a call reads its name and its type from the entry.
    claimedNames: Table[string, uint64]
    ## Emitted proc name to the owning identity: definitions and harvests share one name space, a second claimant of a base suffixes.
    scopes: seq[Table[string, CirExpr]]
    localSpaces: seq[Table[string, CirAddrSpace]]
    ## Nim symbol name to CIRv2 symbol, innermost scope last.
    moduleScope: Table[string, CirExpr]
    moduleSpaces: Table[string, CirAddrSpace]
    ## Program-scope declarations, visible from every proc body, a fresh proc's scope tables seed from these.
    moduleGlobals: seq[CirStmt]
    ## Program-scope declarations in encounter order, `m.globals` of the module.
    nameCounts: Table[string, int]
    hoisted: seq[CirStmt]
    ## Statements hoisted in front of the statement being lowered.

proc failing(ctx: FrontendCtx): bool =
  ctx.resolver.err.len > 0

proc fail(ctx: var FrontendCtx, detail: string) =
  if ctx.resolver.err.len == 0:
    ctx.resolver.err = "cirv2 frontend: at " & ctx.fnName & ": " & detail

# ═══════════════════════════════════════════════════════════════════════
# Names and scopes
# ═══════════════════════════════════════════════════════════════════════

proc uniqueName(ctx: var FrontendCtx, base: string): string =
  ## Per-proc unique CIRv2 name: the sanitized base, suffixed when reused.
  let clean = sanitizeIdent(base)
  if ctx.nameCounts.hasKey(clean):
    ctx.nameCounts[clean].inc
    result = clean & "_" & $ctx.nameCounts[clean]
  else:
    ctx.nameCounts[clean] = 0
    result = clean

proc pushScope(ctx: var FrontendCtx) =
  ctx.scopes.add(initTable[string, CirExpr]())
  ctx.localSpaces.add(initTable[string, CirAddrSpace]())

proc popScope(ctx: var FrontendCtx) =
  ctx.scopes.del(ctx.scopes.high)
  ctx.localSpaces.del(ctx.localSpaces.high)

func getLocalKey(n: NimNode): string =
  ## Lookup key of a Nim local in the scope tables: the symbol's `repr`.
  ## Macro-generated locals share one `strVal` across expansion scopes.
  ## The gensym id inside `repr` tells them apart.
  if n.kind == nnkPragmaExpr:
    result = n[0].repr
  else:
    result = n.repr

proc declareLocal(ctx: var FrontendCtx, nimName: string, sym: CirExpr, spc = caThread) =
  ## Registers a local under an already-computed key in the innermost scope.
  ## Used for frontend-internal temporaries and for `result`.
  ctx.scopes[^1][nimName] = sym
  ctx.localSpaces[^1][nimName] = spc

proc declareLocal(ctx: var FrontendCtx, n: NimNode, sym: CirExpr, spc = caThread) =
  let key = getLocalKey(n)
  ctx.scopes[^1][key] = sym
  ctx.localSpaces[^1][key] = spc

func localSpace(ctx: FrontendCtx, name: string): CirAddrSpace =
  ## Memory class a declared local lives in, `caThread` when unannotated.
  for i in countdown(ctx.localSpaces.high, 0):
    if ctx.localSpaces[i].hasKey(name):
      return ctx.localSpaces[i][name]
  result = caThread

proc resolveLocal(ctx: FrontendCtx, nimName: string): CirExpr =
  ## Looks a local up from the innermost scope outward, nil when unknown.
  for i in countdown(ctx.scopes.high, 0):
    if ctx.scopes[i].hasKey(nimName):
      return ctx.scopes[i][nimName]
  return nil

proc hasPragmaNim(n: NimNode, name: string): bool =
  ## True when `name` appears in the node's pragma list or in the pragma list
  ## of the definition behind a symbol: `{.builtin.}` of the builtins catalog,
  ## `{.magic: "..." .}` of the system operators, bare or colon form.
  case n.kind
  of nnkProcDef, nnkFuncDef:
    for p in n.pragma:
      if p.kind in {nnkIdent, nnkSym} and p.strVal == name:
        return true
      if p.kind in {nnkExprColonExpr, nnkCall} and p.len >= 1 and
          p[0].kind in {nnkIdent, nnkSym} and p[0].strVal == name:
        return true
  of nnkSym:
    let impl = n.getImpl()
    case impl.kind
    of nnkProcDef, nnkFuncDef:
      return hasPragmaNim(impl, name)
    of nnkIdentDefs, nnkConstDef:
      if impl.len > 0 and impl[0].kind == nnkPragmaExpr:
        for p in impl[0][1]:
          if p.kind in {nnkIdent, nnkSym} and p.strVal == name:
            return true
    else:
      discard
  else:
    discard
  return false

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
# Symbol translation
# ═══════════════════════════════════════════════════════════════════════

proc lowerExpr(ctx: var FrontendCtx, n: NimNode): CirExpr

proc lowerOperandExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## Lowers one operator operand: a literal under a sem-introduced conversion
  ## re-types to the conversion's target, a literal conversion is exact.
  let bare = unwrapConv(n)
  if bare != n and bare.kind in {nnkIntLit .. nnkUInt64Lit,
      nnkFloat32Lit, nnkFloat64Lit, nnkFloatLit}:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.failing():
      return nil
    if bare.kind in {nnkFloat32Lit, nnkFloat64Lit, nnkFloatLit}:
      return fltLit(bare.floatVal, typ)
    return intLit(bare.intVal, typ)
  return ctx.lowerExpr(bare)

proc operandLiteralTypedAs(e: CirExpr, opType: CirType): CirExpr =
  ## Returns a literal operand typed as the operation's operand type.
  ## Nim types a literal at its default width and converts at the use site,
  ## and the typed AST does not always carry the conversion around the literal.
  ## Non-literals and kind-mismatched pairs return unchanged.
  if e.isNil or opType.isNil:
    return e
  case e.kind
  of eFloatLit:
    if opType.kind == ctFloat and not sameType(e.fLitType, opType):
      fltLit(e.fValue, opType)
    else:
      e
  of eIntLit:
    if opType.kind in {ctInt, ctUint} and not sameType(e.litType, opType):
      intLit(e.value, opType)
    else:
      e
  else:
    e

func zeroValue(t: CirType): CirExpr =
  ## Zero value of a CIRv2 type, an omitted object-constructor field takes this zero value.
  case t.kind
  of ctInt, ctUint:
    result = intLit(0, t)
  of ctFloat:
    result = fltLit(0.0, t)
  of ctBool:
    result = falseE()
  of ctPtr, ctAptr:
    result = nilE(t)
  of ctArray:
    var elems: seq[CirExpr] = @[]
    for i in 0 ..< t.aN:
      elems.add(zeroValue(t.aElem))
    result = aconstrE(t, elems)
  of ctStruct:
    var fields: seq[CirKv] = @[]
    for f in t.fields:
      fields.add(kv(f.name, zeroValue(f.typ)))
    result = oconstrE(t, fields)
  of ctVoid, ctVector, ctProc, ctSpan:
    result = nil

proc lowerEmitArg(ctx: var FrontendCtx, emitArg: NimNode): CirStmt =
  ## Literal runs alternate with spliced expressions.
  ## A computed splice stores through a fresh temp that the emitted text
  ## references by name.
  if emitArg.kind in {nnkStrLit, nnkRStrLit, nnkTripleStrLit}:
    return emitS(@[emitArg.strVal], @[])
  var parts: seq[string] = @[""]
  var operands: seq[CirExpr] = @[]
  for part in emitArg:
    case part.kind
    of nnkStrLit, nnkRStrLit, nnkTripleStrLit:
      parts[^1].add(part.strVal)
    else:
      let e = ctx.lowerExpr(part)
      if ctx.failing():
        return nil
      var op = e
      if e.kind != eSym:
        # printers splice symbol names, a computed value needs a temp
        let tmpName = ctx.uniqueName("emit_tmp")
        let tmpType = getExprType(e)
        op = sym(tmpName, tmpType)
        ctx.hoisted.add(varD(tmpName, tmpType))
        ctx.hoisted.add(asgnS(op, e))
        ctx.declareLocal(tmpName, op)
      parts.add("")
      operands.add(op)
  result = emitS(parts, operands)

proc resolveTypeSection(ctx: var FrontendCtx, n: NimNode) =
  ## Registers the section's concrete type definitions with the resolver.
  ## A generic definition registers nothing at declaration time,
  ## resolution of its instantiations happens at the use sites.
  for typeDef in n:
    typeDef.expectKind(nnkTypeDef)
    let head = typeDef[0]
    if head.kind == nnkPragmaExpr:
      # a pragma-wrapped type name keeps the type node under the pragma
      discard ctx.resolver.resolveCirType(head[0])
    elif head.kind == nnkBracketExpr or head.typeKind == ntyGenericBody:
      continue
    else:
      discard ctx.resolver.resolveCirType(head)

proc lowerStmt(ctx: var FrontendCtx, n: NimNode): seq[CirStmt]


proc lowerStmtBody(ctx: var FrontendCtx, n: NimNode): seq[CirStmt]

proc drainHoisted(ctx: var FrontendCtx): seq[CirStmt]

proc resolveSymExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  result = ctx.resolveLocal(n.repr)
  if result.isNil:
    ctx.fail("unknown local symbol " & n.repr)
  return result

func hasLabeledBreak(n: NimNode, label: string): bool =
  ## True when a `break label` statement inside `n` targets `label`.
  ## Sym children are not descended into, a label never nests a break.
  if n.kind == nnkBreakStmt and n.len > 0 and n[0].kind in {nnkSym, nnkIdent} and
      n[0].strVal == label:
    return true
  if n.kind in {nnkSym, nnkIdent, nnkEmpty, nnkCommentStmt}:
    return false
  for c in n:
    if hasLabeledBreak(c, label):
      return true
  return false

proc builtinSymExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## A `{.builtin.}` value reference: the canonical coordinate vectors and the flat thread index,
  ## emitted as a plain symbol. The printers map the name onto the target vocabulary.
  let name = sanitizeIdent(n.strVal)
  let typ = ctx.resolver.resolveCirType(n.getTypeInst())
  if ctx.failing():
    return nil
  result = sym(name, typ, builtin = true)

proc lowerSym(ctx: var FrontendCtx, n: NimNode): CirExpr =
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
    return ctx.resolveSymExpr(n)
  of nskConst:
    if hasPragmaNim(n, "builtin"):
      return ctx.builtinSymExpr(n)
    let impl = n.getImpl()
    if impl.kind == nnkConstDef:
      return ctx.lowerExpr(impl[2])
  else:
    ctx.fail("symbol " & name & " (" & $n.symKind & ") is not a value")
    return nil

# ═══════════════════════════════════════════════════════════════════════
# Call translation
# ═══════════════════════════════════════════════════════════════════════

proc getFormalParams(fn: NimNode): NimNode

proc resolveCallCallee(callee: NimNode): NimNode =
  ## Callee symbol of a sem-checked call. Overload resolution runs before
  ## the macro sees the call. A resolved call names the symbol sem picked,
  ## a surviving choice node names sem's pick as its first candidate.
  if callee.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and callee.len > 0:
    return callee[0]
  return callee

proc getFormalParams(fn: NimNode): NimNode =
  for i, c in fn:
    if c.kind == nnkFormalParams:
      return c
  return nil

proc getProcBody(fn: NimNode): NimNode =
  ## Body of a proc definition node: the trailing statement list when present,
  ## otherwise the trailing statement boxed into one, a single-statement body
  ## has no statement-list wrapper.
  ##
  ## A value-returning proc appends its `result` symbol as an extra trailing
  ## child and the skip loop removes it.
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

func isGenericProc(node: NimNode): bool =
  ## True for a generic proc or an instantiation of one, whose params sit in an `nnkBracket` at `impl[5]`.
  let impl = node.getImpl()
  if impl.kind notin [nnkProcDef, nnkFuncDef]:
    return false
  if impl[2].kind == nnkGenericParams:
    return true
  result = impl.len > 5 and impl[5].kind == nnkBracket and
    impl[5].len > 1 and impl[5][1].kind == nnkGenericParams

proc calleeSigKey(callee: NimNode): string =
  ## Identity key of a callee symbol: the repr and the signature hash.
  ## A generic adds the body hash (instantiations share one signature).
  result = callee.repr & "_" & callee.signatureHash()
  if callee.symKind in {nskProc, nskFunc} and isGenericProc(callee):
    result &= "_" & callee.symBodyHash()

proc resolveCalleeProcType(ctx: var FrontendCtx, callee: NimNode): CirType

func procIdentity(callee: NimNode): uint64 =
  ## Identity hash of a callee: the full 64-bit hash of the signature key.
  ## Distinct instantiations of one generic carry distinct identities through
  ## the body hash inside the key.
  typeIdentityHash(calleeSigKey(callee))

proc claimProcName(ctx: var FrontendCtx, identity: uint64, base: string): string =
  ## Emitted name of a proc identity: `base` on the first claim.
  ## A second claimant of the same base takes a base58 suffix of the identity hash.
  ## The suffix rechecks by appending underscores until the name is unclaimed.
  ##
  ## A claim by the owning identity returns the base: the module reservation
  ## pass reserves kernel plain names, so a kernel's own def re-claims its name.
  if ctx.claimedNames.getOrDefault(base, identity) == identity:
    ctx.claimedNames[base] = identity
    return base
  result = base & "_" & base58Short(identity)
  while ctx.claimedNames.getOrDefault(result, identity) != identity:
    result.add("_")
  ctx.claimedNames[result] = identity

proc registerOrGet(ctx: var FrontendCtx, calleeSym: NimNode): ProcEntry =
  ## Registry entry of a call's callee.
  ## The first registration derives the proc type once and claims the emitted name under the harvest contract: base plus identity-hash suffix.
  ## Later calls read the entry.
  ##
  ## A `{.builtin.}` callee keeps its canonical native name at the call site. The entry stores its type only.
  let id = procIdentity(calleeSym)
  if ctx.procRegistry.hasKey(id):
    result = ctx.procRegistry[id]
    if result.fnType.isNil:
      # a definition registers before its body lowers, so the type is read
      # once at the first call inside it
      result.fnType = ctx.resolveCalleeProcType(calleeSym)
      if ctx.failing():
        return
      ctx.procRegistry[id].fnType = result.fnType
    return
  let fnType = ctx.resolveCalleeProcType(calleeSym)
  if ctx.failing():
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
    if ctx.failing():
      return @[]
    # a literal argument re-types to the formal, the conversion is exact
    let ftyp = ctx.resolver.resolveCirType(ftype)
    if ctx.failing():
      return @[]
    result.add(operandLiteralTypedAs(arg, ftyp))

proc resolveCalleeProcType(ctx: var FrontendCtx, callee: NimNode): CirType =
  ## Proc type of a callee symbol. An overload choice resolves to its first candidate,
  ## which sem picked during overload resolution.
  let calleeSym =
    if callee.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and callee.len > 0:
      callee[0]
    else:
      callee
  result = ctx.resolver.resolveCirType(calleeSym.getTypeInst())
  if ctx.failing():
    return nil

proc nimMagicName(n: NimNode): string =
  ## Value of the definition's `magic: "..."` pragma, "" when absent.
  ## The system integer and float operators are magics:
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

proc hasBodyStatements(impl: NimNode): bool =
  ## True when the definition's body has a real statement, a bodiless magic renders as a `CommentStmt` and an `Empty`.
  let body = getProcBody(impl)
  if body.isNil:
    return false
  for c in body:
    if c.kind notin {nnkCommentStmt, nnkEmpty}:
      return true
  return false

proc isBodilessMagic(callee: NimNode): bool =
  ## A magic whose effect has no Nim body: a call lowers to the primitive operation, nothing harvests.
  nimMagicName(callee).len > 0 and not hasBodyStatements(callee.getImpl())

proc lowerMagicPrimitive(ctx: var FrontendCtx, callee: NimNode, call: NimNode, magic: string): CirExpr =
  ## Lowers a call of an arithmetic magic into the matching primitive CIRv2 operation,
  ## nil when the magic has no primitive counterpart. Only bodiless magics route here,
  ## a magic with a Nim body (like `max`) harvests through the ordinary proc path.
  case magic
  of "AddI", "AddU", "AddF32", "AddF64",
     "SubI", "SubU", "SubF32", "SubF64",
     "MulI", "MulU", "MulF32", "MulF64",
     "DivI", "DivU", "DivF32", "DivF64",
     "ModI", "ModU", "ShlI", "ShrI":
    let args = ctx.lowerCallArgs(callee, call)
    if ctx.failing():
      return nil
    let opType = getExprType(args[0])
    let rhs = operandLiteralTypedAs(args[1], opType)
    case magic
    of "AddI", "AddU", "AddF32", "AddF64":
      return addE(opType, args[0], rhs)
    of "SubI", "SubU", "SubF32", "SubF64":
      return subE(opType, args[0], rhs)
    of "MulI", "MulU", "MulF32", "MulF64":
      return mulE(opType, args[0], rhs)
    of "DivI", "DivU", "DivF32", "DivF64":
      return divE(opType, args[0], rhs)
    of "ModI", "ModU":
      return modE(opType, args[0], rhs)
    of "ShlI":
      return shlE(opType, args[0], rhs)
    else:
      return shrE(opType, args[0], rhs)
  else:
    return nil

proc absBuiltinCall(ctx: var FrontendCtx, calleeSym: NimNode, call: NimNode): CirExpr =
  ## Lowers an `abs` call over scalar operands as a builtin call, no body harvest: the printers use the per-type native name.
  ##
  ## Only `abs` intercepts:
  ## - system `abs` over a float has a `when nimvm` body. The frontend refuses it, so no harvest path exists
  ## - `min`/`max` harvest into a ternary idiom, folded to the native call in the optimization stage
  ## - a user proc named `abs` over aggregates takes the ordinary harvest path. The targets have no builtin over aggregates
  if calleeSym.strVal != "abs":
    return nil
  let fnType = ctx.resolveCalleeProcType(calleeSym)
  if ctx.failing():
    return nil
  for p in fnType.params:
    if p.kind notin {ctBool, ctInt, ctUint, ctFloat}:
      return nil
  let args = ctx.lowerCallArgs(calleeSym, call)
  if ctx.failing():
    return nil
  return callE(sym(calleeSym.strVal, fnType, true), args)

proc callTargetName(n: NimNode): string =
  if n.kind == nnkSym:
    return n.strVal
  if n.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and n.len > 0:
    return n[0].strVal
  return ""

proc lowerToOpenArrayCall(ctx: var FrontendCtx, call: NimNode): CirExpr =
  ## Lowers a `toOpenArray(base, first, last)` call into the span construction
  ## the span split pass consumes.
  ## The span type comes from the base pointer's element type and space.
  ##
  ## The base must be a pointer: the split's pointer side offsets the base by `first`,
  ## and CIRv2 states pointer arithmetic over pointers.
  let base = ctx.lowerExpr(call[1])
  if ctx.failing():
    return nil
  let first = ctx.lowerOperandExpr(call[2])
  let last = ctx.lowerOperandExpr(call[3])
  if ctx.failing():
    return nil
  let baseTyp = getExprType(base)
  if baseTyp.isNil or baseTyp.kind notin {ctPtr, ctAptr}:
    ctx.fail("toOpenArray over a " &
      (if baseTyp.isNil: "base with no derivable type" else: $baseTyp.kind) &
      " is outside the supported set, pass a pointer to the storage")
    return nil
  let firstTyp = getExprType(first)
  let lastTyp = getExprType(last)
  let spanTyp = tSpan(baseTyp.space, baseTyp.pElem)
  let callTyp = tProcT(baseTyp, firstTyp, lastTyp, spanTyp)
  return callE(sym("toOpenArray", callTyp), base, first, last)

proc lowerSpanLenCall(ctx: var FrontendCtx, call: NimNode): CirExpr =
  ## Lowers a `len(s)` read over an openArray into the `len` call that the span split pass folds to the span's split length binding.
  let arg = ctx.lowerExpr(call[1])
  if ctx.failing():
    return nil
  let argTyp = getExprType(arg)
  let readTyp = tProcT(argTyp, tInt(32))
  return callE(sym("len", readTyp), arg)

proc lowerCallValue(ctx: var FrontendCtx, callee: NimNode, call: NimNode): CirExpr =
  ## A value-producing call as a call expression: the use site nests the call
  ## into its operator tree.
  ## A magic callee lowers to the primitive operation directly,
  ## no proc harvests, no call emits.
  let magic = nimMagicName(callee)
  # toOpenArray builds a span and len reads a span's length: the span split
  # pass consumes both forms, no proc harvests, no call emits
  if magic == "Slice" and callTargetName(callee) == "toOpenArray":
    return ctx.lowerToOpenArrayCall(call)
  if magic == "LengthOpenArray":
    return ctx.lowerSpanLenCall(call)
  if isBodilessMagic(callee):
    let arith = ctx.lowerMagicPrimitive(callee, call, magic)
    if ctx.failing():
      return nil
    if arith.isNil:
      ctx.fail("magic call " & callee.repr & " (" & magic & ") is outside the supported set")
      return nil
    return arith
  let calleeSym = resolveCallCallee(callee)
  if ctx.failing():
    return nil
  let absCall = ctx.absBuiltinCall(calleeSym, call)
  if ctx.failing():
    return nil
  if absCall != nil:
    return absCall
  let entry = ctx.registerOrGet(calleeSym)
  if ctx.failing():
    return nil
  # a bodiless magic took the primitive path above
  # every other callee harvests, a magic with a Nim body (like `max`) lowers through it
  let calleeBuiltin = hasPragmaNim(calleeSym, "builtin")
  if not calleeBuiltin and not entry.hasBody:
    ctx.harvestProc(calleeSym, entry.name)
    if ctx.failing():
      return nil
  let args = ctx.lowerCallArgs(calleeSym, call)
  if ctx.failing():
    return nil
  # a builtin callee emits the canonical native name, an instantiation hash would
  # name a proc the backend never defines. Builtin-marking the callee symbol
  # lets the alias canonicalization pass kind-assign from the canonical tables
  let calleeName = if calleeBuiltin: calleeSym.strVal
                   else: entry.name
  return callE(sym(calleeName, entry.fnType, calleeBuiltin), args)

# ═══════════════════════════════════════════════════════════════════════
# Expression translation
# ═══════════════════════════════════════════════════════════════════════

proc isLatticeScalar(ctx: var FrontendCtx, n: NimNode): bool =
  ## True when the expression's type resolves to a primitive scalar kind,
  ## the kind the primitive arithmetic covers.
  ## An overloaded operand (a register tile, an element-wise `-.`/`*.` proc)
  ## lowers as the proc's call instead.
  let t = ctx.resolver.resolveCirType(n.getTypeInst())
  if ctx.failing():
    return false
  return t != nil and t.kind in {ctBool, ctInt, ctUint, ctFloat}

proc lowerInfix(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## Operators: arithmetic and bitwise directly, comparisons normalized to
  ## the four primitive comparison kinds, `>` and `>=` swap operands.
  let op = n[0].strVal
  case op
  of "+", "-", "*", "div", "mod", "shl", "shr", "/", "and", "or", "xor":
    if not (ctx.isLatticeScalar(n[1]) and ctx.isLatticeScalar(n[2])):
      # overloaded operator over non-primitive-scalar operands: the check fails,
      # overload resolution picks the proc, lowered here as a call.
      # Example: `m_i - m_ij` over (MT x 1) tiles. Each operand lowers once,
      # inside the call, since this check runs before operand lowering.
      return ctx.lowerCallValue(n[0], n)
  else:
    discard
  let a = ctx.lowerOperandExpr(n[1])
  if ctx.failing():
    return nil
  case op
  of "==", "!=", "<", "<=", ">", ">=":
    let b = ctx.lowerOperandExpr(n[2])
    if ctx.failing():
      return nil
    let at = getExprType(a)
    let bCmp = operandLiteralTypedAs(b, at)
    case op
    of "==":
      result = eqE(a, bCmp)
    of "!=":
      result = neqE(a, bCmp)
    of "<":
      result = ltE(a, bCmp)
    of "<=":
      result = leE(a, bCmp)
    of ">":
      result = ltE(b, a)
    else:
      result = leE(b, a)
    return result
  of "and", "or", "xor":
    # bool domain: boolean operators emit directly over their operands
    # (`xor` evaluates both sides and compares). Text targets apply
    # short-circuit natively, SPIR-V's OpLogicalAnd/OpLogicalOr do not.
    # int domain: bitwise operators.
    let at = getExprType(a)
    if at != nil and at.kind == ctBool:
      let b = ctx.lowerOperandExpr(n[2])
      if ctx.failing():
        return nil
      case op
      of "and":
        return andE(a, b)
      of "or":
        return orE(a, b)
      else:
        return neqE(a, b)
    let b = ctx.lowerOperandExpr(n[2])
    if ctx.failing():
      return nil
    case op
    of "and":
      result = bitAndE(at, a, b)
    of "or":
      result = bitOrE(at, a, b)
    else:
      result = bitXorE(at, a, b)
    return result
  of "+", "-", "*", "div", "mod", "shl", "shr":
    let b = ctx.lowerOperandExpr(n[2])
    if ctx.failing():
      return nil
    let opType = getExprType(a)
    let bOp = operandLiteralTypedAs(b, opType)
    case op
    of "+":
      result = addE(opType, a, bOp)
    of "-":
      result = subE(opType, a, bOp)
    of "*":
      result = mulE(opType, a, bOp)
    of "div":
      result = divE(opType, a, bOp)
    of "mod":
      result = modE(opType, a, bOp)
    of "shl":
      result = shlE(opType, a, bOp)
    else:
      result = shrE(opType, a, bOp)
    return result
  of "/":
    # Nim `/` is float division, `div` is the integer one
    let b = ctx.lowerOperandExpr(n[2])
    if ctx.failing():
      return nil
    let opType = getExprType(a)
    result = divE(opType, a, operandLiteralTypedAs(b, opType))
    return result
  else:
    ctx.fail("operator " & op & " is outside the supported set")
    return nil

proc buildBranchBody(ctx: var FrontendCtx, bodyNode: NimNode, dest: CirExpr): seq[CirStmt]

proc lowerIfForm(ctx: var FrontendCtx, branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode, dest: CirExpr): seq[CirStmt]

func branchValueExpr(n: NimNode): NimNode =
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

func allBranchesSimple(branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode): bool =
  ## True when every branch body and the else body hold a call-free value expression,
  ## that shape nests into ternary values with no hoisted temp.
  ## - A call in a branch value binds a temp and the binding must stay inside the taken branch.
  ## - Call-bearing branches take the statement form.
  for br in branches:
    if isStmtBody(branchValueExpr(br.body)) or containsCall(branchValueExpr(br.body)):
      return false
  return not isStmtBody(branchValueExpr(elseBody)) and
      not containsCall(branchValueExpr(elseBody))

proc lowerNestedTernary(ctx: var FrontendCtx,
    branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode, idx: int,
    resType: CirType): CirExpr =
  ## Branch `idx` of an if-expression as a ternary whose else variant
  ## contains the next branch, first match wins.
  ## The innermost else variant is `elseBody`.
  let cond = ctx.lowerExpr(branches[idx].cond)
  if ctx.failing():
    return nil
  let thenV = ctx.lowerExpr(branchValueExpr(branches[idx].body))
  if ctx.failing():
    return nil
  let elseV = if idx + 1 < branches.len:
                 lowerNestedTernary(ctx, branches, elseBody, idx + 1, resType)
               else:
                 ctx.lowerExpr(branchValueExpr(elseBody))
  if ctx.failing():
    return nil
  return ternaryE(resType, cond, thenV, elseV)

proc lowerIfExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## An if-expression: single-expression branches nest into conditional values,
  ## statement-bearing branches hoist a temp that the statement form assigns.
  let resType = ctx.resolver.resolveCirType(n.getTypeInst())
  if ctx.failing():
    return nil
  var branches: seq[tuple[cond: NimNode, body: NimNode]] = @[]
  var elseBody: NimNode = nil
  for c in n:
    case c.kind
    of nnkElifBranch:
      # statement shape of an elif chain reaches the expression form
      # when the branch values sit in value position
      branches.add((cond: c[0], body: c[1]))
    of nnkElse:
      elseBody = c[0]
    of nnkElifExpr:
      branches.add((cond: c[0], body: c[1]))
    of nnkElseExpr:
      elseBody = c[0]
    else:
      ctx.fail("if expression: unexpected branch " & $c.kind)
      return nil
  if allBranchesSimple(branches, elseBody):
    return lowerNestedTernary(ctx, branches, elseBody, 0, resType)
  let tmpName = ctx.uniqueName("iftmp")
  ctx.declareLocal(tmpName, sym(tmpName, resType))
  let stmts = ctx.lowerIfForm(branches, elseBody, sym(tmpName, resType))
  if ctx.failing():
    return nil
  ctx.hoisted.add(@[varD(tmpName, resType)] & stmts)
  return sym(tmpName, resType)

proc lowerSplicedStmt(ctx: var FrontendCtx, c: NimNode): seq[CirStmt] =
  ## One non-final child of an expression block, a statement of the spliced expansion,
  ## hoisted in front of the enclosing statement.
  ## A discarded construction step of the expansion drops, such a step has no
  ## runtime code (an object or tuple with no call inside).
  if c.kind == nnkStmtList:
    for child in c:
      result.add(ctx.lowerSplicedStmt(child))
      if ctx.failing():
        return @[]
    return result
  if c.kind in {nnkObjConstr, nnkTupleConstr, nnkExprColonExpr, nnkBracketExpr,
                nnkPar, nnkIntLit .. nnkUInt64Lit, nnkFloatLit, nnkStrLit} and
      not containsCall(c):
    return @[]
  return ctx.lowerStmt(c)

proc retypedLiteral(e: CirExpr, typ: CirType): CirExpr =
  case e.kind
  of eIntLit:
    intLit(e.value, typ)
  of eFloatLit:
    fltLit(e.fValue, typ)
  else:
    e


proc lowerExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## Translates one expression node: the statements spliced by the expression
  ## sit in front of the enclosing statement.
  if ctx.failing():
    return nil
  if n.kind == nnkNilLit:
    # a nil literal: the literal node reports isNil too, this case runs
    # before the missing-node check
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.failing():
      return nil
    return nilE(typ)
  case n.kind
  of nnkSym:
    return ctx.lowerSym(n)
  of nnkIntLit .. nnkUInt64Lit:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.failing():
      return nil
    return intLit(n.intVal, typ)
  of nnkFloat32Lit, nnkFloat64Lit, nnkFloatLit:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.failing():
      return nil
    return fltLit(n.floatVal, typ)
  of nnkHiddenStdConv:
    # int/int32 share one bit width, the standard conversion is the identity.
    # A float literal sems at its default width and converts to the context's
    # float width here: a literal conversion is exact, the literal re-types
    # to the conversion's target
    let inner = n[1]
    if inner.kind in {nnkFloat32Lit, nnkFloat64Lit, nnkFloatLit}:
      let typ = ctx.resolver.resolveCirType(n.getTypeInst())
      if ctx.failing():
        return nil
      return fltLit(inner.floatVal, typ)
    return ctx.lowerExpr(inner)
  of nnkHiddenSubConv, nnkConv:
    var typ = if n[0].kind == nnkEmpty:
      ctx.resolver.resolveCirType(n.getTypeInst())
    else:
      ctx.resolver.resolveCirType(n[0])
    if ctx.failing():
      return nil
    let operand = ctx.lowerExpr(n[1])
    if ctx.failing():
      return nil
    let ot = getExprType(operand)
    if ot != nil and sameType(ot, typ):
      # an identity conversion: the layout algebra's static wrappers stack up and the value already has the target type
      return operand
    if operand.kind in {eIntLit, eFloatLit}:
      # a literal conversion is exact: the literal re-types to the cast's target,
      # no runtime conversion node materializes.
      # An int literal converting to a float type becomes that float literal.
      if typ.kind == ctFloat and operand.kind == eIntLit:
        return fltLit(operand.value.float, typ)
      return retypedLiteral(operand, typ)
    if typ.kind in {ctPtr, ctAptr}:
      # a pointer conversion keeps its operand's storage space, the target type names the pointee
      typ = typ.withSpace(provenanceSpace(operand))
    return convE(typ, operand)
  of nnkCast:
    var typ = ctx.resolver.resolveCirType(n[0])
    if ctx.failing():
      return nil
    let operand = ctx.lowerExpr(n[1])
    if ctx.failing():
      return nil
    if typ.kind in {ctPtr, ctAptr}:
      # a pointer cast keeps its operand's storage space: the target type
      # names the pointee, the space derives from the value's provenance
      typ = typ.withSpace(provenanceSpace(operand))
    return castE(typ, operand)
  of nnkHiddenDeref, nnkDerefExpr:
    let base = ctx.lowerExpr(n[0])
    if ctx.failing():
      return nil
    let bt = getExprType(base)
    if bt != nil and bt.kind == ctSpan:
      # a sem deref of a span names the span, an index over it selects one element
      return base
    return derefE(base)
  of nnkHiddenAddr, nnkAddr:
    let base = ctx.lowerExpr(n[0])
    if ctx.failing():
      return nil
    let bt = getExprType(base)
    # address space: the memory where the base value lives
    # an element of device memory gets a device pointer, a local a thread pointer
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
        # a local base lives where its declaration puts it: `{.smem.}` locals are threadgroup storage, unannotated locals are registers
        space = ctx.localSpace(walk.name)
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
    if ctx.failing():
      return nil
    let idx = ctx.lowerOperandExpr(n[1])
    if ctx.failing():
      return nil
    let bt = getExprType(loweredBase)
    case bt.kind
    of ctAptr:
      return patE(loweredBase, idx)
    of ctArray:
      return atE(loweredBase, idx)
    of ctPtr:
      # a var parameter of array type names the array through a pointer:
      # deref names that array, the index selects one element
      return atE(derefE(loweredBase), idx)
    of ctSpan:
      # a span element read, the span split pass retypes the base symbol
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
    if ctx.failing():
      return nil
    if base.kind == eOconstr:
      # a field read of a construction is the field's value, no runtime temporary materializes
      for f in base.kvFields:
        if f.name == sanitizeIdent(n[1].strVal):
          return f.value
    return dotE(base, sanitizeIdent(n[1].strVal))
  of nnkInfix:
    return ctx.lowerInfix(n)
  of nnkPrefix:
    let op = n[0].strVal
    let operand = ctx.lowerOperandExpr(n[1])
    if ctx.failing():
      return nil
    case op
    of "-":
      return negE(getExprType(operand), operand)
    of "not":
      return notE(operand)
    else:
      # dispatch on a string, an unknown prefix operator would otherwise
      # fall through uninitialized
      ctx.fail("prefix operator " & op & " is outside the supported set")
      return nil
  of nnkCall, nnkCommand:
    let callee = n[0]
    if callee.kind notin {nnkSym, nnkOpenSymChoice, nnkClosedSymChoice}:
      ctx.fail("call callee " & callee.repr & " is not a proc symbol")
      return nil
    return ctx.lowerCallValue(callee, n)
  of nnkObjConstr:
    var fields: seq[CirKv] = @[]
    var valueSpaces = initTable[string, CirAddrSpace]()
    for i in 1 ..< n.len:
      n[i].expectKind(nnkExprColonExpr)
      if n[i][1].kind == nnkEmpty:
        # a nil field value sems to an Empty node, the omitted-field zero
        # fill below supplies the typed nil
        continue
      let v = ctx.lowerExpr(n[i][1])
      if ctx.failing():
        return nil
      let fieldName = sanitizeIdent(n[i][0].strVal)
      # a pointer-typed field instantiates under the space its value proves
      valueSpaces[fieldName] = provenanceSpace(v)
      fields.add(kv(fieldName, v))
    let typ = ctx.resolver.resolveStructType(n.getTypeInst(), valueSpaces)
    if ctx.failing():
      return nil
    # Nim binds constructor values by name, the target syntax binds positionally:
    # normalize every constructor to the type's declaration order, omitted fields zero-initialize,
    # an empty initializer list is invalid C
    var ordered: seq[CirKv] = @[]
    for f in typ.fields:
      var value: CirExpr = nil
      for kvf in fields:
        if kvf.name == f.name:
          value = kvf.value
          break
      if value != nil:
        ordered.add(kv(f.name, value))
      else:
        ordered.add(kv(f.name, zeroValue(f.typ)))
    fields = ordered
    return oconstrE(typ, fields)
  of nnkExprColonExpr:
    # a named field outside a constructor: the name is compile-time
    # vocabulary of the expansion, the value is the runtime content
    return ctx.lowerExpr(n[1])
  of nnkTupleConstr:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.failing():
      return nil
    var fields: seq[CirKv] = @[]
    for i in 0 ..< n.len:
      let v = ctx.lowerExpr(n[i])
      if ctx.failing():
        return nil
      fields.add(kv("F" & $i, v))
    return oconstrE(typ, fields)
  of nnkBracket:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.failing():
      return nil
    var elems: seq[CirExpr] = @[]
    for el in n:
      let v = ctx.lowerExpr(el)
      if ctx.failing():
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
        if ctx.failing():
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
        if ctx.failing():
          return nil
    return value
  of nnkIfExpr:
    return ctx.lowerIfExpr(n)
  of nnkStrLit, nnkRStrLit, nnkTripleStrLit:
    ctx.fail("string literal: GPU kernels carry no strings")
    return nil
  of nnkCharLit:
    ctx.fail("char literal: outside the supported set")
    return nil
  else:
    ctx.fail("expression construct not supported: " & $n.kind & " " & n.repr)
    return nil

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc drainHoisted(ctx: var FrontendCtx): seq[CirStmt] =
  result = ctx.hoisted
  ctx.hoisted = @[]

proc buildBranchBody(ctx: var FrontendCtx, bodyNode: NimNode, dest: CirExpr): seq[CirStmt] =
  ## One branch body: plain statements, or the expression form assigning
  ## `dest` from the branch's value.
  if dest.isNil:
    return ctx.lowerStmt(bodyNode)
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
  if ctx.failing():
    return @[]
  return stmts & ctx.drainHoisted() & @[asgnS(dest, value)]

proc lowerIfForm(ctx: var FrontendCtx, branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode, dest: CirExpr): seq[CirStmt] =
  ## An elif chain nests one `ite` per branch, first match wins.
  ## The else statements sit innermost.
  ## The expression form assigns `dest` in every branch body.
  var tail: seq[CirStmt] = @[]
  if not elseBody.isNil:
    tail = ctx.buildBranchBody(elseBody, dest)
    if ctx.failing():
      return @[]
  for i in countdown(branches.len - 1, 0):
    let cond = ctx.lowerExpr(branches[i].cond)
    if ctx.failing():
      return @[]
    # condition splices emit before the branch test, inside the else body
    # of the previous branch, so a splice runs only when the test runs
    let condHoists = ctx.drainHoisted()
    let body = ctx.buildBranchBody(branches[i].body, dest)
    if ctx.failing():
      return @[]
    tail = condHoists & @[iteS(cond, body, tail)]
  result = tail

proc lowerIfStmt(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
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

proc lowerWhile(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## Statements spliced by the condition drain to the front of the loop body,
  ## a splice runs once per iteration after the test that admits it.
  let cond = ctx.lowerExpr(n[0])
  if ctx.failing():
    return @[]
  let body = ctx.lowerStmt(n[1])
  if ctx.failing():
    return @[]
  return @[whileS(cond, body)]

proc lowerFor(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## A `for` over a literal range or `countup`.
  ## The loop variable declares up front, bounds evaluate once at entry,
  ## the increment runs last.
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
      # `items` walks it inclusive, the loop is the equivalent range.
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
      ctx.fail("for loop over the iterator " & name & " is outside the supported set: " & n.repr)
      return @[]
  else:
    ctx.fail("for loop over " & $iter.kind & " is outside the supported set")
    return @[]
  let varName = ctx.uniqueName($vars)
  let varType = ctx.resolver.resolveCirType(vars.getTypeInst())
  if ctx.failing():
    return @[]
  let startV = ctx.lowerExpr(unwrapConv(startNode))
  if ctx.failing():
    return @[]
  let stopV = ctx.lowerExpr(unwrapConv(stopNode))
  if ctx.failing():
    return @[]
  # bounds and step re-type to the loop variable. A literal conversion is exact,
  # comparisons and increment then use one operand type
  let startT = operandLiteralTypedAs(startV, varType)
  let stopT = operandLiteralTypedAs(stopV, varType)
  var stepV: CirExpr = nil
  var stepT: CirExpr = nil
  var stepIsConst = true
  if stepNode.isNil:
    stepV = intLit(1, varType)
  else:
    let unwrapped = unwrapConv(stepNode)
    if unwrapped.kind in nnkIntLit .. nnkUInt64Lit:
      if unwrapped.intVal <= 0:
        ctx.fail("for loop step " & $unwrapped.intVal &
          ": only positive steps are supported")
        return @[]
      stepV = ctx.lowerExpr(unwrapped)
    else:
      # no entry lowering: a step's calls must not run once before the loop
      # body ever executes, per-iteration lowering belongs to the incr branch
      stepIsConst = false
  if ctx.failing():
    return @[]
  if stepIsConst:
    stepT = operandLiteralTypedAs(stepV, varType)
  ctx.declareLocal(vars, sym(varName, varType))
  let varSym = sym(varName, varType)
  # start/stop/step hoists queued so far belong to the preamble, the body's own hoists drain per statement inside it
  let startHoists = ctx.drainHoisted()
  let body = ctx.lowerStmtBody(n[^1])
  if ctx.failing():
    return @[]
  let bodyHoists = ctx.drainHoisted()
  let cmp = if inclusive:
    leE(varSym, stopT)
  else:
    ltE(varSym, stopT)
  var incr: seq[CirStmt] = @[]
  if stepIsConst:
    incr.add(asgnS(varSym, addE(varType, varSym, stepT)))
  else:
    # loop-variant step expressions lower inside the loop,
    # they re-evaluate at the continue point.
    # A step frozen at entry counts wrong iterations.
    let stepNow = ctx.lowerExpr(unwrapConv(stepNode))
    if ctx.failing():
      return @[]
    let stepNowT = operandLiteralTypedAs(stepNow, varType)
    incr.add(ctx.drainHoisted() & @[asgnS(varSym, addE(varType, varSym, stepNowT))])
  var preamble = startHoists
  preamble.add(varD(varName, varType))
  preamble.add(asgnS(varSym, startT))
  result = preamble & @[whileS(cmp, bodyHoists & body & incr)]

proc lowerReturn(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  var valueNode = n[0]
  if valueNode.kind == nnkAsgn:
    valueNode = valueNode[1]
  if valueNode.kind == nnkEmpty:
    if ctx.retType.kind != ctVoid:
      return @[retS(sym("result", ctx.retType))]
    return @[retS(nil)]
  let value = ctx.lowerExpr(valueNode)
  if ctx.failing():
    return @[]
  return ctx.drainHoisted() & @[retS(value)]

proc lowerVoidCall(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## A call in statement position: a barrier, a kernel call, or a discardable
  ## non-void call whose value drops.
  let callee = n[0]
  # a generic type instantiation left in statement position appears
  # as the `[]` choice, the first operand is the type
  # a type has no runtime code
  if callee.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and callee.len > 0 and
      callee[0].strVal == "[]" and n.len >= 2 and
      n[1].kind == nnkSym and n[1].symKind == nskType:
    return @[]
  if callee.kind notin {nnkSym, nnkOpenSymChoice, nnkClosedSymChoice}:
    ctx.fail("call callee " & callee.repr & " is not a proc symbol")
    return @[]
  let calleeSym = resolveCallCallee(callee)
  if ctx.failing():
    return @[]
  let calleeBuiltin = hasPragmaNim(calleeSym, "builtin")
  if isBodilessMagic(callee):
    ctx.fail("magic call " & callee.repr & " in statement position without a destination")
    return @[]
  let entry = ctx.registerOrGet(calleeSym)
  if ctx.failing():
    return @[]
  let calleeName = if calleeBuiltin: calleeSym.strVal
                   else: entry.name
  if not calleeBuiltin and not entry.hasBody:
    ctx.harvestProc(calleeSym, entry.name)
    if ctx.failing():
      return @[]
  let args = ctx.lowerCallArgs(calleeSym, n)
  if ctx.failing():
    return @[]
  return ctx.drainHoisted() & @[callS(sym(calleeName, entry.fnType, calleeBuiltin), args)]

proc resolveLocalSpace(ctx: var FrontendCtx, nameNode: NimNode): CirAddrSpace =
  ## Memory class of a local declaration: the pragma list decides.
  ## `smem` means threadgroup, `const_mem` constant, `rmem` or unannotated registers.
  result = caThread
  if nameNode.kind != nnkPragmaExpr:
    return
  var seen = false
  template setSpace(spc: CirAddrSpace) =
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

proc lowerLetSection(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## A `let`/`var` section: one typed declaration per name, a call-valued initializer
  ## binds through an `asgn` after the declaration.
  ## The memory class comes from the declaration's pragma list.
  for defs in n:
    let typeNode = defs[^2]
    let initNode = defs[^1]
    for j in 0 ..< defs.len - 2:
      let nameNode = defs[j]
      let spc = ctx.resolveLocalSpace(nameNode)
      if ctx.failing():
        return @[]
      let nimName = if nameNode.kind == nnkPragmaExpr: nameNode[0].strVal
                    else: nameNode.strVal
      var typ: CirType = nil
      if typeNode.kind != nnkEmpty:
        typ = ctx.resolver.resolveCirType(typeNode)
      else:
        typ = ctx.resolver.resolveCirType(initNode.getTypeInst())
      if ctx.failing():
        return @[]
      var value: CirExpr = nil
      if initNode.kind != nnkEmpty:
        value = ctx.lowerExpr(initNode)
        if ctx.failing():
          return @[]
        if value.kind == eOconstr and typ != nil and typ.kind == ctStruct and
            value.ocType.kind == ctStruct:
          # a construction instantiates the struct it builds, the declaration adopts the instantiation with its provenance-derived fields
          typ = value.ocType
        elif typ != nil and typ.kind in {ctPtr, ctAptr} and
            getExprType(value) != nil and getExprType(value).kind == typ.kind and
            getExprType(value).space != typ.space:
          # a pointer cast keeps the operand's address space:
          # - the declaration adopts the value's space
          # - `let q = cast[ptr T](addr scratch[0])` is a threadgroup pointer
          typ = getExprType(value)
      let hoists = ctx.drainHoisted()
      let cirName = ctx.uniqueName(nimName)
      ctx.declareLocal(nameNode, sym(cirName, typ), spc)
      if value.isNil:
        result.add(hoists & @[varD(cirName, typ, nil, spc)])
      elif value.kind == eCall:
        result.add(hoists & @[varD(cirName, typ, nil, spc),
            asgnS(sym(cirName, typ), value)])
      else:
        result.add(hoists & @[varD(cirName, typ, value, spc)])

proc lowerConstSection(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## A `const` section: the value must already be a compile-time literal.
  for defs in n:
    defs.expectKind(nnkConstDef)
    let nameNode = defs[0]
    var nimName = if nameNode.kind == nnkPragmaExpr: nameNode[0].strVal
                  else: nameNode.strVal
    let typeNode = defs[1]
    let valueNode = defs[2]
    var typ: CirType = nil
    if typeNode.kind != nnkEmpty:
      typ = ctx.resolver.resolveCirType(typeNode)
    else:
      typ = ctx.resolver.resolveCirType(valueNode.getTypeInst())
    if ctx.failing():
      return @[]
    let value = ctx.lowerExpr(valueNode)
    if ctx.failing():
      return @[]
    let hoists = ctx.drainHoisted()
    let cirName = ctx.uniqueName(nimName)
    ctx.declareLocal(nameNode, sym(cirName, typ))
    result.add(hoists & @[constD(cirName, typ, value)])

proc isScalarCompoundAssign(ctx: var FrontendCtx, n: NimNode): bool =
  ## True when a compound assign's operands are primitive scalars where the primitive arithmetic applies.
  ## An overloaded operand (a register tile, an element-wise `op=` proc) lowers
  ## as the proc's call instead.
  if n[0].kind notin {nnkSym, nnkOpenSymChoice, nnkClosedSymChoice}:
    return false
  let lt = ctx.resolver.resolveCirType(n[1].getTypeInst())
  let rt = ctx.resolver.resolveCirType(n[2].getTypeInst())
  if ctx.failing():
    return false
  let lp = if lt != nil and lt.kind == ctPtr: lt.pElem else: lt
  proc isScalar(t: CirType): bool =
    t != nil and t.kind in {ctBool, ctInt, ctUint, ctFloat}
  return isScalar(lp) and isScalar(rt)

proc lowerStmtBody(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  if ctx.failing():
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
      if ctx.failing():
        return @[]
  of nnkLetSection, nnkVarSection:
    return ctx.lowerLetSection(n)
  of nnkConstSection:
    return ctx.lowerConstSection(n)
  of nnkAsgn:
    let lhs = ctx.lowerExpr(n[0])
    if ctx.failing():
      return @[]
    let rhs = ctx.lowerExpr(n[1])
    if ctx.failing():
      return @[]
    # a literal rhs re-types to the destination, the conversion is exact
    let rhsT = operandLiteralTypedAs(rhs, getExprType(lhs))
    return ctx.drainHoisted() & @[asgnS(lhs, rhsT)]
  of nnkInfix:
    # compound assignment reaches the statement position as an infix:
    # primitive-scalar operands take the primitive arithmetic, any other `op=`
    # (`qk -.= m_ij` over register tiles) resolves to its overloaded proc
    # and lowers as a var-target call in statement position
    let op = n[0].strVal
    let scalar = ctx.isScalarCompoundAssign(n)
    if ctx.failing():
      # operand type resolution failed, return with the pending failure,
      # a statement-position call path needs a clean channel
      return @[]
    if op in ["+=", "-=", "*=", "/="] and scalar:
      var lhs = ctx.lowerExpr(n[1])
      if ctx.failing():
        return @[]
      # a var-returning `[]` hands the compound assignment the element's address
      # and the assignment target is the location behind it
      if lhs.kind == eAddr:
        lhs = lhs.addrOperand
      elif lhs.kind == eSym and getExprType(lhs) != nil and
          getExprType(lhs).kind == ctPtr:
        # a var parameter names its location through a pointer and the compound
        # assignment operates on the pointee the pointer names
        lhs = derefE(lhs)
      let rhs = ctx.lowerOperandExpr(n[2])
      if ctx.failing():
        return @[]
      let opType = getExprType(lhs)
      let rhsOp = operandLiteralTypedAs(rhs, opType)
      # target and arithmetic read share one tree reference
      # a desugared `op=` left-hand side evaluates exactly once in Nim
      # temp-binding normalization splits the reads of a call-bearing target
      let arith = case op
        of "+=": addE(opType, lhs, rhsOp)
        of "-=": subE(opType, lhs, rhsOp)
        of "*=": mulE(opType, lhs, rhsOp)
        else: divE(opType, lhs, rhsOp)
      return ctx.drainHoisted() & @[asgnS(lhs, arith)]
    return ctx.lowerVoidCall(n)
  of nnkIfStmt:
    return ctx.lowerIfStmt(n)
  of nnkWhileStmt:
    return ctx.lowerWhile(n)
  of nnkForStmt:
    return ctx.lowerFor(n)
  of nnkReturnStmt:
    return ctx.lowerReturn(n)
  of nnkDiscardStmt:
    if n[0].kind == nnkEmpty:
      return @[discardS(nil)]
    if n[0].kind in {nnkCall, nnkCommand}:
      return ctx.lowerVoidCall(n[0])
    let value = ctx.lowerExpr(n[0])
    if ctx.failing():
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
        if ctx.failing():
          return @[]
        parts.add("")
        operands.add(op)
      else:
        ctx.fail("asm statement operand of kind " & $c.kind &
          " is outside the supported set")
        return @[]
    return ctx.drainHoisted() & @[asmS(parts, operands)]
  of nnkBlockStmt:
    if n[0].kind in {nnkSym, nnkIdent} and hasLabeledBreak(n[1], n[0].strVal):
      ctx.fail("labeled block: break out of blocks is outside the supported set")
      return @[]
    # a label no `break label` targets is dead, the block lowers as its body
    return ctx.lowerStmt(n[1])
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
      let s = ctx.lowerEmitArg(emitArg)
      if ctx.failing():
        return @[]
      emitted.add(ctx.drainHoisted() & @[s])
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
    # a discarded construction step of a macro expansion
    # value is never read, the step lowers for its side effects only
    discard ctx.lowerExpr(n)
    if ctx.failing():
      return @[]
    return ctx.drainHoisted()
  of nnkExprColonExpr:
    # a discarded named-field step of a macro expansion
    # value is never read
    discard ctx.lowerExpr(n[1])
    if ctx.failing():
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
  if ctx.failing():
    return @[]
  return ctx.drainHoisted() & body

# ═══════════════════════════════════════════════════════════════════════
# Proc translation
# ═══════════════════════════════════════════════════════════════════════

proc lowerProcBody(ctx: var FrontendCtx, fn: NimNode, formals: NimNode, body: NimNode, params: var seq[CirParam]): seq[CirStmt] =
  ## Shared proc body lowering:
  ## - the formal parameters declare into the proc's root scope
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
    if ctx.failing():
      ctx.popScope()
      return @[]
    for j in 0 ..< formal.len - 2:
      let pname = ctx.uniqueName(formal[j].strVal)
      let psym = sym(pname, typ)
      params.add(CirParam(name: pname, typ: typ,
        varParam: ftype.kind == nnkVarTy))
      ctx.declareLocal(formal[j], psym)
  let ret = ctx.retType
  if ret.kind != ctVoid:
    let rsym = sym("result", ret)
    ctx.declareLocal("result", rsym)
    result.add(varD("result", ret))
  result.add(ctx.lowerStmt(body))
  ctx.popScope()
  if ctx.failing():
    return @[]
  # a fall-through body takes the fallback return, an explicit trailing
  # return already ends the proc
  if result.len > 0 and result[^1].kind == sRet:
    discard
  elif ret.kind != ctVoid:
    result.add(retS(sym("result", ret)))
  else:
    result.add(retS(nil))

proc lowerProcDef(ctx: var FrontendCtx, fn: NimNode): CirProc =
  ## One proc definition: parameters typed off Nim once and the body lowered
  ## into normal form.
  ##
  ## Generic definitions skip here, their instantiations harvest at the calls that use them.
  ## A def claims and emits under its own name: kernels keep the plain name,
  ## other defs mangle the operator name, a claim conflict suffixes.
  ## A call-site harvest never claims a plain name.
  if ctx.failing():
    return nil
  if isGenericDef(fn):
    return nil
  let formals = getFormalParams(fn)
  let id = procIdentity(fn[0])
  # a kernel keeps its plain name, launchKernel is ABI on it
  let base = if hasPragmaNim(fn, "global"): sanitizeIdent(fn[0].strVal)
             else: mangleOperatorName(fn[0].strVal)
  let cirName = ctx.claimProcName(id, base)
  if id notin ctx.procRegistry:
    ctx.procRegistry[id] = ProcEntry(name: cirName)
  ctx.procRegistry[id].hasBody = true
  let savedName = ctx.fnName
  let savedRet = ctx.retType
  let savedCounts = ctx.nameCounts
  let savedScopes = ctx.scopes
  let savedSpaces = ctx.localSpaces
  ctx.fnName = cirName
  ctx.nameCounts = initTable[string, int]()
  ctx.scopes = @[ctx.moduleScope]
  ctx.localSpaces = @[ctx.moduleSpaces]
  ctx.retType = ctx.resolver.resolveCirType(formals[0])
  if ctx.failing():
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
  ctx.fnName = savedName
  ctx.retType = savedRet
  ctx.nameCounts = savedCounts
  ctx.scopes = savedScopes
  ctx.localSpaces = savedSpaces
  if ctx.failing():
    return nil
  ctx.procs.add(result)

proc harvestProc(ctx: var FrontendCtx, callee: NimNode, cirName: string) =
  ## Lowers the body of a callee a call registered.
  ## The emitted name and the proc type come from the registry entry, compile-time parameters erased.
  ## `hasBody` set before the body lowers guards the recursion, callers harvest only body-less entries.
  let id = procIdentity(callee)
  ctx.procRegistry[id].hasBody = true
  let impl = callee.getImpl()
  let formals = getFormalParams(callee.getTypeInst())
  let savedName = ctx.fnName
  let savedRet = ctx.retType
  let savedCounts = ctx.nameCounts
  let savedScopes = ctx.scopes
  let savedSpaces = ctx.localSpaces
  let savedProcsLen = ctx.procs.len
  let savedHoisted = ctx.hoisted
  # callee lowers with its own hoist queue: the caller's pending hoists
  # belong to the caller's statement, never to the callee's body
  ctx.hoisted = @[]
  ctx.fnName = cirName
  ctx.nameCounts = initTable[string, int]()
  ctx.scopes = @[ctx.moduleScope]
  ctx.localSpaces = @[ctx.moduleSpaces]
  ctx.retType = ctx.resolver.resolveCirType(formals[0])
  if ctx.failing():
    return
  var params: seq[CirParam] = @[]
  let fnRet = ctx.retType
  let body = ctx.lowerProcBody(impl, formals, getProcBody(impl), params)
  ctx.fnName = savedName
  ctx.retType = savedRet
  ctx.nameCounts = savedCounts
  ctx.scopes = savedScopes
  ctx.localSpaces = savedSpaces
  # callee's own pending hoists close its body, the caller's queue resumes
  let calleeHoists = ctx.drainHoisted()
  ctx.hoisted = savedHoisted
  if ctx.failing():
    ctx.procs.setLen(savedProcsLen)
    return
  let harvested = procD(cirName, params, fnRet, calleeHoists & body)
  let wgErr = impl.applyProcPragmas(harvested)
  if wgErr.len > 0:
    ctx.fail(wgErr)
    ctx.fnName = savedName
    ctx.retType = savedRet
    ctx.nameCounts = savedCounts
    ctx.scopes = savedScopes
    ctx.localSpaces = savedSpaces
    ctx.procs.setLen(savedProcsLen)
    ctx.hoisted = savedHoisted
    return
  ctx.fnName = savedName
  ctx.retType = savedRet
  ctx.nameCounts = savedCounts
  ctx.scopes = savedScopes
  ctx.localSpaces = savedSpaces
  ctx.procs.add(harvested)

# ═══════════════════════════════════════════════════════════════════════
# Entry points
# ═══════════════════════════════════════════════════════════════════════

proc lowerTopLevelDecls(ctx: var FrontendCtx, n: NimNode) =
  ## A top-level `var`/`const` section: program-scope storage whose pragma list
  ## picks the memory class, unannotated reading as constant.
  ## Kernels must appear after the declarations they read.
  for def in n:
    let nameN = def[0]
    let bareName = if nameN.kind == nnkPragmaExpr: nameN[0] else: nameN
    let name = bareName.repr
    let space =
      block:
        var spc = ctx.resolveLocalSpace(nameN)
        if spc == caThread:
          # program scope has no per-thread storage, unannotated is constant
          spc = caConstant
        spc
    let valN = def[2]
    let typN = if def[1].kind == nnkEmpty: valN.getTypeInst() else: def[1]
    let t = ctx.resolver.resolveCirType(typN, space)
    if ctx.failing():
      return
    var init: CirExpr = nil
    if valN.kind != nnkEmpty:
      init = ctx.lowerExpr(valN)
      if ctx.failing():
        return
    let g = varD(name, t, init, space)
    ctx.moduleGlobals.add(g)
    let symE = sym(name, t)
    ctx.moduleScope[name] = symE
    ctx.moduleSpaces[name] = space

proc lowerModule(ctx: var FrontendCtx, bodyIn: NimNode) =
  ## Lowers the typed macro body: a block of kernel proc defs, or the single `{.global.}` wrapper proc of a launchDeviceFn call.
  var body = bodyIn
  if body.kind == nnkStmtListExpr and body.len == 1:
    # typed evaluator wraps a single-expression body in a one-child list
    body = body[0]
  if body.kind == nnkProcDef or body.kind == nnkFuncDef:
    discard ctx.lowerProcDef(body)
    return
  if body.kind == nnkSym:
    let impl = body.getImpl()
    if impl.kind notin {nnkProcDef, nnkFuncDef}:
      ctx.fail(body.strVal & " does not name a proc")
      return
    discard ctx.lowerProcDef(impl)
    return
  if body.kind != nnkStmtList:
    ctx.fail("expected a block of kernel proc definitions or a single kernel proc, got " &
      $body.kind)
    return
  # kernels are ABI: every {.global.} def claims its plain name before any body lowers,
  # a same-named device proc takes the suffix
  for c in body:
    if c.kind in {nnkProcDef, nnkFuncDef} and not isGenericDef(c) and
        hasPragmaNim(c, "global"):
      discard ctx.claimProcName(procIdentity(c[0]), sanitizeIdent(c[0].strVal))
  for c in body:
    case c.kind
    of nnkProcDef, nnkFuncDef:
      discard ctx.lowerProcDef(c)
    of nnkTypeSection:
      ctx.resolveTypeSection(c)
    of nnkVarSection, nnkConstSection:
      ctx.lowerTopLevelDecls(c)
    of nnkCommentStmt:
      discard
    of nnkBindStmt, nnkMixinStmt:
      # symbol-binding directives, Nim resolves them at sem time
      discard
    else:
      ctx.fail("top-level construct not supported: " & $c.kind & " " & c.repr)
      return

proc lowerCirv2*(body: NimNode): tuple[m: CirModule, err: string] =
  ## Lowers the typed macro body to CIRv2, returning the module and "" when clean,
  ## else the first refusal message. The constant-shell fold runs on this output.
  var ctx = FrontendCtx()
  ctx.pushScope()
  ctx.lowerModule(body)
  let m = moduleD("kernel", ctx.resolver.structs, ctx.procs)
  m.globals = ctx.moduleGlobals
  return (m, ctx.resolver.err)

proc tryLowerCirv2*(body: NimNode): tuple[m: CirModule, err: string] =
  let (m, err) = lowerCirv2(body)
  return (m, err)

proc toCirv2Module*(body: NimNode): CirModule =
  ## Lowers the typed macro body to CIRv2, a construct outside the supported
  ## closure rejects through `error`, naming the construct.
  let (m, err) = tryLowerCirv2(body)
  if err.len > 0:
    error(err)
  return m
