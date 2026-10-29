# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cirv2 frontend: typed NimNode to CIRv2, normal form by construction.
##
## Entry: the typed node the `metal:`-family macros receive:
##
## - a block of kernel proc defs
## - the single `{.global.}` wrapper proc whose body calls the kernel
##   (the launchDeviceFn shape)
##
##   typed NimNode ──lower──▶ CIRv2 normal form ──verify──▶ CirModule
##
## Types translate exactly once through `nim_type_resolvers`. Normal form produced directly:
##
## - a value-producing call is an `asgn` RHS, nested call operands hoist in front of the enclosing statement
## - `while` and `for` lower to a conditionless `loop` with a leading guard
## - `and`/`or` compile through the two-target condition compiler (Appel Cx), value position materializes into a bool temp through `ite`
## - if/elif chains lay flat over one shared merge label
##
## Constructs outside the supported closure refuse through the pure refusal
## channel (`lowerMessage`), naming the construct. The macro wrapper turns
## a refusal into a compile-time `error`.

import ../ir/cirv2_lang
import ../ir/cirv2_build
import nim_type_resolvers
import ../ir/cirv2_verify
import ../ir/cirv2_pretty
import std/[macros, tables, strutils, hashes]

type
  FrontendCtx = object
    ## Translation state for one module.
    resolver: CirTypeResolver
    fnName: string
    ## Name of the proc under translation, names every refusal.
    retType: CirType
    procs: seq[CirProc]
    procKeys: Table[string, bool]
    ## Harvested proc names, dedups instantiations and guards recursion.
    scopes: seq[Table[string, CirExpr]]
    localSpaces: seq[Table[string, CirAddrSpace]]
    ## Nim symbol name to CIRv2 symbol, innermost scope last.
    moduleScope: Table[string, CirExpr]
    moduleSpaces: Table[string, CirAddrSpace]
    ## Program-scope declarations, visible from every proc body. A fresh proc's scope tables seed from these.
    moduleGlobals: seq[CirStmt]
    ## Program-scope declarations in encounter order, `m.globals` of the module.
    nameCounts: Table[string, int]
    labels: int
    loopStack: seq[tuple[exitL, contL: string]]
    hoisted: seq[CirStmt]
    ## Statements hoisted in front of the statement being lowered.
    continueUsed: bool
    breakTargets: Table[string, string]
    ## Labeled block name to exit label, for `break label`.
    ## Empty: no labeled block is open.

proc failing(ctx: FrontendCtx): bool =
  ## True while a refusal is recorded, the walk unwinds through this guard.
  ctx.resolver.err.len > 0

proc fail(ctx: var FrontendCtx, detail: string) =
  ## Records the first refusal, naming the construct and the proc.
  if ctx.resolver.err.len == 0:
    ctx.resolver.err = "cirv2 frontend: at " & ctx.fnName & ": " & detail

# ═══════════════════════════════════════════════════════════════════════
# Names, scopes, labels
# ═══════════════════════════════════════════════════════════════════════

const ContinueLabelPrefix* = "lc"
  ## freshLabel prefix of a for-loop's continue label, the structure pass
  ## picks the loop's `continue` target by it.

proc uniqueName(ctx: var FrontendCtx, base: string): string =
  ## Per-proc unique CIRv2 name: the sanitized base, suffixed when reused.
  let clean = sanitizeIdent(base)
  if ctx.nameCounts.hasKey(clean):
    ctx.nameCounts[clean].inc
    result = clean & "_" & $ctx.nameCounts[clean]
  else:
    ctx.nameCounts[clean] = 0
    result = clean

proc freshLabel(ctx: var FrontendCtx, prefix: string): string =
  ## Proc-unique label name, forward-only lab/jmp targets derive from these.
  ctx.labels.inc
  result = prefix & $ctx.labels

proc pushScope(ctx: var FrontendCtx) =
  ctx.scopes.add(initTable[string, CirExpr]())
  ctx.localSpaces.add(initTable[string, CirAddrSpace]())

proc popScope(ctx: var FrontendCtx) =
  ctx.scopes.del(ctx.scopes.high)
  ctx.localSpaces.del(ctx.localSpaces.high)

func getLocalKey(n: NimNode): string =
  ## Lookup key of a Nim local in the scope tables: the symbol's `repr`.
  ##
  ## Macro-generated locals share one `strVal` across their expansion scopes,
  ## for example every `composeMode0` of the layout algebra. The gensym id
  ## inside `repr` is what tells them apart.
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
  ## Registers a local under its symbol key in the innermost scope.
  let key = getLocalKey(n)
  ctx.scopes[^1][key] = sym
  ctx.localSpaces[^1][key] = spc

func localSpace(ctx: FrontendCtx, name: string): CirAddrSpace =
  ## Memory class a declared local lives in, `caThread` when unannotated
  ## (`smem` locals are threadgroup, `const_mem` locals constant).
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
  ## True when `name` appears in the node's pragma list or in the pragma
  ## list of the definition behind a symbol.
  ##
  ## Detected markers:
  ## - `{.builtin.}` of the builtins catalog
  ## - `{.magic: "..." .}` of the system operators
  ##
  ## Both markers match in the bare and in the colon pragma form.
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
  ## True when a call occurs anywhere in the node.
  ## `StmtListExpr`-wrapped operands count too.
  ## A call-bearing condition needs the two-target compiler.
  ## Short-circuit must not evaluate the call side.
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
  ## Peels the sem-introduced wrappers off a node.
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
  ## Forward declaration, expressions and statements recurse into each other.

proc lowerOperandExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## Lowers one operator operand. A literal under a sem-introduced conversion
  ## re-types to the conversion's target, a literal conversion is exact.
  ## Every other node lowers through `lowerExpr` with the wrappers peeled.
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
  ##
  ## Nim types a literal at its default width and converts at the use site.
  ## The typed AST does not always carry the conversion around the literal:
  ## the operation's operand type is the inference result, and the literal
  ## re-types to that operand type.
  ##
  ## - a literal conversion is exact
  ## - non-literals and kind-mismatched pairs return unchanged
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

func provenanceSpace(e: CirExpr): CirAddrSpace =
  ## Address space of the storage an expression's pointer provenance names:
  ## the innermost pointer type along the access chain.
  ##
  ## - provenance survives value casts and integer arithmetic over cast pointers
  ## - `caDevice` reads as no provenance
  result = caDevice
  case e.kind
  of eSym:
    let t = e.typ
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.space
  of eDeref, eAt, ePat:
    # element reads live where the base's storage lives
    let base = case e.kind
      of eDeref: e.derefOperand
      of eAt: e.atBase
      else: e.patBase
    if base != nil:
      result = provenanceSpace(base)
  of eDot:
    let t = getExprType(e)
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.space
    elif e.dotBase != nil:
      result = provenanceSpace(e.dotBase)
  of eCast, eConv:
    let t = e.castType
    if t != nil and t.kind in {ctPtr, ctAptr}:
      result = t.space
    elif e.castOperand != nil:
      result = provenanceSpace(e.castOperand)
  of eAddr:
    if e.addrType != nil and e.addrType.kind in {ctPtr, ctAptr}:
      result = e.addrType.space
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    result = provenanceSpace(e.left)
    if result == caDevice:
      result = provenanceSpace(e.right)
  else:
    discard

func zeroValue(t: CirType): CirExpr =
  ## Zero value of a lattice type:
  ##
  ## - 0 for numerics, false for bool, nil for pointers
  ## - element-wise zeros for arrays, all-zero fields for structs
  ##
  ## An omitted object-constructor field initializes to this, v1's rule,
  ## an empty initializer list is never valid C.
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
  of ctVoid, ctVector, ctProc:
    result = nil

proc lowerEmitArg(ctx: var FrontendCtx, emitArg: NimNode): CirStmt =
  ## One `{.emit.}` argument, lowered to an `sEmit` statement.
  ##
  ## - a plain string literal emits verbatim
  ## - an argument list alternates literal runs with spliced expressions,
  ##   Nim hands backticked identifiers in the string over as symbols
  ## - a compound expression splices through a fresh temp: the hoisted
  ##   assignment computes the value, the text references the temp's name
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
        let tmpName = ctx.uniqueName("emit_tmp")
        op = sym(tmpName, getExprType(e))
        ctx.hoisted.add(asgnS(op, e))
        ctx.declareLocal(tmpName, op)
      parts.add("")
      operands.add(op)
  result = emitS(parts, operands)

proc resolveTypeSection(ctx: var FrontendCtx, n: NimNode) =
  ## Registers the section's concrete type definitions with the resolver.
  ##
  ## - a generic type definition registers nothing at declaration time
  ## - instantiations resolve at the use sites (`Tile[2, 3](...)`)
  ## - the generic head (`ntyGenericBody`) has no runtime representation
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
  ## Forward declaration, expressions and statements recurse into each other.


proc lowerStmtBody(ctx: var FrontendCtx, n: NimNode): seq[CirStmt]
  ## Forward declaration, the statement walk proper.

proc drainHoisted(ctx: var FrontendCtx): seq[CirStmt]
  ## Forward declaration, hoist placement is needed before the walk.

proc resolveSymExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## A local symbol reference (param, let, var, for var, result, temp).
  result = ctx.resolveLocal(n.repr)
  if result.isNil:
    ctx.fail("unknown local symbol " & n.repr)
  return result

func hasLabeledBreak(n: NimNode, label: string): bool =
  ## True when a `break label` statement inside `n` targets `label`.
  ## Sym children are not descended into, a label name never nests a break.
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
  ## A `{.builtin.}` value reference:
  ## - the canonical coordinate vectors and the flat thread index
  ## - the CIRv2 expression is a plain symbol, the printers map the name
  ##   onto the target vocabulary
  let name = sanitizeIdent(n.strVal)
  let typ = ctx.resolver.resolveCirType(n.getTypeInst())
  if ctx.failing():
    return nil
  result = sym(name, typ, builtin = true)

proc lowerSym(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## Symbol reference in expression position: locals, builtins, consts,
  ## boolean literals. Boolean literals are `true`/`false` symbols.
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
    ctx.fail("const symbol " & name & " without a definition")
    return nil
  else:
    ctx.fail("symbol " & name & " (" & $n.symKind & ") is not a value")
    return nil

# ═══════════════════════════════════════════════════════════════════════
# Call translation
# ═══════════════════════════════════════════════════════════════════════

proc getFormalParams(fn: NimNode): NimNode
  ## Forward declaration, the overload matcher reads the formals.

func getHeadName(n: NimNode): string =
  ## Head name of a type node: the nominal or generic name the type instantiates.
  ## `var`, `static`, `ptr`, and generic-argument wrappers unwrap to the head.
  case n.kind
  of nnkVarTy, nnkStaticTy, nnkPtrTy, nnkRefTy, nnkBracketExpr, nnkCall,
      nnkHiddenStdConv, nnkHiddenSubConv:
    if n.len > 0:
      result = getHeadName(n[0])
  of nnkSym, nnkIdent:
    result = n.strVal
  else:
    result = ""

proc resolveCallCallee(ctx: var FrontendCtx, callee: NimNode, call: NimNode): NimNode =
  ## An overload choice resolves to the candidate whose formals fit the call's
  ## argument types: head name per parameter against head name per argument.
  ## The first candidate is only correct when every overload agrees on all parameters.
  if callee.kind notin {nnkOpenSymChoice, nnkClosedSymChoice}:
    return callee
  for cand in callee:
    if cand.kind != nnkSym:
      continue
    let formals = getFormalParams(cand.getTypeInst())
    if formals.isNil or formals.len - 1 < call.len - 1:
      continue
    let impl = cand.getImpl()
    var generics: seq[string] = @[]
    if impl.kind in {nnkProcDef, nnkFuncDef, nnkConverterDef, nnkMethodDef} and
        impl[1].kind == nnkGenericParams:
      for gp in impl[1]:
        if gp.kind == nnkIdentDefs:
          for j in 0 ..< gp.len - 2:
            generics.add(gp[j].strVal)
    var fits = true
    for i in 1 ..< call.len:
      let formal = formals[i]
      if formal.kind != nnkIdentDefs or formal.len < 2:
        fits = false
        break
      let fhead = getHeadName(formal[formal.len - 2])
      let ahead = getHeadName(call[i].getTypeInst())
      if ahead != "" and fhead != ahead and fhead notin generics:
        fits = false
        break
    if fits:
      return cand
  ctx.fail("no overload of " & callee.repr & " matches the call's argument types")
  return nil

proc getFormalParams(fn: NimNode): NimNode =
  ## Returns the `FormalParams` child of a proc definition or proc type node.
  for i, c in fn:
    if c.kind == nnkFormalParams:
      return c
  return nil

proc getProcBody(fn: NimNode): NimNode =
  ## Returns the body of a proc definition node:
  ## - the trailing statement list when present
  ## - otherwise the trailing statement boxed into one, a single-statement
  ##   body like `proc k() = f(x)` has no statement-list wrapper
  ## - a value-returning proc appends its `result` symbol as an extra trailing child, skipped here
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
  ## True when a proc definition declares generic parameters: `GenericParams` or the instantiation `Bracket` form.
  for c in fn:
    if c.kind == nnkGenericParams:
      return true
    if c.kind == nnkBracket and c.len > 0 and c[0].kind == nnkEmpty:
      for g in c:
        if g.kind == nnkGenericParams:
          return true
  return false

proc isCompileTimeFormal(t: NimNode): bool =
  ## `static T` (either node layout) or `typedesc`: the argument is
  ## compile-time only, erased from the closure.
  ##
  ## A `static` formal bound to a value shows that value as its instantiated
  ## type node: a literal (an enum member's ordinal for a `static ReduceOp`-
  ## like formal) or the enum field's symbol. The argument never reaches runtime.
  if t.kind in nnkIntLit .. nnkUInt64Lit:
    return true
  if t.kind == nnkSym and t.symKind == nskEnumField:
    return true
  if t.kind in {nnkStaticTy, nnkBracketExpr, nnkCommand}:
    if t.len > 0 and t[0].repr == "static":
      return true
  if t.kind in {nnkIdent, nnkSym} and t.strVal == "typedesc":
    return true
  if t.kind == nnkBracketExpr and t.len > 0 and t[0].strVal == "typeDesc":
    # an instantiated `typeDesc[T]` formal of a generic callee
    return true
  return false

func isGenericProc(node: NimNode): bool =
  ## True for a generic proc or an instantiation of one, whose params sit
  ## in an `nnkBracket` at `impl[5]`.
  let impl = node.getImpl()
  if impl.kind notin [nnkProcDef, nnkFuncDef]:
    return false
  if impl[2].kind == nnkGenericParams:
    return true
  result = impl.len > 5 and impl[5].kind == nnkBracket and
    impl[5].len > 1 and impl[5][1].kind == nnkGenericParams

proc computeInstProcName(ctx: var FrontendCtx, callee: NimNode): string =
  ## Stable per-callee-instantiation name: the sanitized proc name plus
  ## a base58 hash over the callee's identity.
  ##
  ## Hash input:
  ##
  ## - the instantiated signature
  ## - the callee's name
  ## - the callee's definition site
  ##
  ## One CIRv2 proc exists per distinct callee instantiation:
  ##
  ## - the signature alone cannot separate callees: the `ProcTy` repr
  ##   erases bound static formals
  ## - the base maps operator callees to word names: `+` becomes `plus`.
  ##   An operator is not a legal identifier in the targets.
  ##   The hash folds the callee's name and definition site,
  ##   separating overloads that share one erased signature
  ## - two call sites of one proc instantiation produce the same name,
  ##   the harvest dedup keeps one body per instantiation
  let base = mangleOperatorName(callee.strVal)
  let formals = getFormalParams(callee.getTypeInst())
  var sigRepr = callee.getTypeInst().repr
  if not formals.isNil:
    # each formal's instantiated type node in raw form, `getTypeInst` of a bound
    # static formal collapses to the value's type so distinct enum-member
    # instantiations would share one signature and collide

    sigRepr = ""
    for i in 0 ..< formals.len:
      let f = formals[i]
      if f.kind == nnkIdentDefs and f.len >= 2:
        let ft = f[f.len - 2]
        if isCompileTimeFormal(ft):
          # a bound static formal hashes its value, the type node of a bound
          # static collapses to the value's type and enum-member instantiations
          # would share one signature and collide
          sigRepr.add(ft.repr)
        else:
          sigRepr.add(ft.getTypeInst().repr)
      else:
        sigRepr.add(f.repr)
  let impl = callee.getImpl()
  let calleeSite = if impl.kind == nnkEmpty:
    ""
  else:
    impl.lineInfo
  # an instantiated signature alone cannot separate instantiations: Nim's
  # `signatureHash` erases the bound static formals, the body hash of a generic
  # instantiation includes them. Plain text of both goes into the hash.
  var sigKey = callee.repr & "_" & callee.signatureHash()
  if callee.symKind in {nskProc, nskFunc} and isGenericProc(callee):
    sigKey &= "_" & callee.symBodyHash()
  var h = hash(sigKey & "\x1f" & sigRepr & "\x1f" & callee.strVal & "\x1f" & calleeSite)
  h = !$h
  var x = uint64(h) and 0xFFFFFFFFFFFFFFFF'u64
  const alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  var suffix = ""
  if x == 0:
    suffix = "1"
  while x > 0:
    suffix = alphabet[int(x mod 58)] & suffix
    x = x div 58
  result = base & "_" & suffix

proc harvestProc(ctx: var FrontendCtx, callee: NimNode, cirName: string)

proc lowerCallArgs(ctx: var FrontendCtx, callee: NimNode, call: NimNode): seq[CirExpr] =
  ## Lowers the runtime arguments of a call, erasing the compile-time
  ## parameters' arguments (their values live in the callee's types).
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
  if argIdx + 1 < call.len:
    ctx.fail("call has more arguments than the callee has parameters")
    return @[]
  if formalIdx + 1 < formals.len and not isCompileTimeFormal(
      formals[formalIdx + 1][formals[formalIdx + 1].len - 2]):
    ctx.fail("call has fewer arguments than the callee has runtime parameters")
    return @[]

proc resolveCalleeProcType(ctx: var FrontendCtx, callee: NimNode): CirType =
  ## Returns the proc type of a callee symbol: runtime parameters in order, then the return type.
  ## An overload choice resolves to its first candidate, the one sem picked.
  let calleeSym =
    if callee.kind in {nnkOpenSymChoice, nnkClosedSymChoice} and callee.len > 0:
      callee[0]
    else:
      callee
  if calleeSym.kind != nnkSym:
    ctx.fail("callee " & callee.repr & " does not name a proc")
    return nil
  let procTy = calleeSym.getTypeInst()
  if procTy.kind != nnkProcTy:
    ctx.fail("callee " & callee.repr & " does not name a proc")
    return nil
  result = ctx.resolver.resolveCirType(procTy)
  if ctx.failing():
    return nil

proc nimMagicName(n: NimNode): string =
  ## Returns the value of the definition's `magic: "..."` pragma.
  ## Returns "" when the definition has no magic pragma.
  ##
  ## The system integer and float operators are magics.
  ## A magic's effect is the primitive operation.
  ## Its Nim body is a documentation stub.
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
  ## True when the definition's body has a real statement.
  ## Nim renders a bodiless magic as a `CommentStmt` and an `Empty`.
  ## The magic's effect lives in the compiler backend.
  let body = getProcBody(impl)
  if body.isNil:
    return false
  for c in body:
    if c.kind notin {nnkCommentStmt, nnkEmpty}:
      return true
  return false

proc isBodilessMagic(callee: NimNode): bool =
  ## A magic whose effect has no Nim body.
  ## The system integer and float operators are the cases in point.
  ## A call lowers to the primitive operation, nothing harvests.
  nimMagicName(callee).len > 0 and not hasBodyStatements(callee.getImpl())

proc lowerMagicPrimitive(ctx: var FrontendCtx, callee: NimNode, call: NimNode, magic: string): CirExpr =
  ## Lowers a call of an arithmetic magic into the matching primitive
  ## CIRv2 operation, nil when the magic has no primitive counterpart.
  ## Only bodiless magics route here: a magic with a Nim body (like `max`)
  ## harvests through the ordinary proc path.
  case magic
  of "AddI", "AddF32", "AddF64",
     "SubI", "SubF32", "SubF64",
     "MulI", "MulF32", "MulF64",
     "DivI", "DivF32", "DivF64",
     "ModI", "ShlI", "ShrI":
    let args = ctx.lowerCallArgs(callee, call)
    if ctx.failing():
      return nil
    if args.len != 2:
      ctx.fail("magic " & magic & " call with " & $args.len & " arguments, expected 2")
      return nil
    let opType = getExprType(args[0])
    if opType.isNil:
      ctx.fail("magic " & magic & " call with an operand of no derivable type")
      return nil
    let rhs = operandLiteralTypedAs(args[1], opType)
    case magic
    of "AddI", "AddF32", "AddF64":
      return addE(opType, args[0], rhs)
    of "SubI", "SubF32", "SubF64":
      return subE(opType, args[0], rhs)
    of "MulI", "MulF32", "MulF64":
      return mulE(opType, args[0], rhs)
    of "DivI", "DivF32", "DivF64":
      return divE(opType, args[0], rhs)
    of "ModI":
      return modE(opType, args[0], rhs)
    of "ShlI":
      return shlE(opType, args[0], rhs)
    else:
      return shrE(opType, args[0], rhs)
  else:
    return nil

proc numericBuiltinCall(ctx: var FrontendCtx, calleeSym: NimNode, call: NimNode): CirExpr =
  ## Lowers a `min`/`max`/`abs` call as a builtin call, no body harvest.
  ## The printers use the per-type native name, v1's contract:
  ## `fmax`/`fmin`/`fabs` on OpenCL floats. Nil for any other callee.
  if calleeSym.strVal notin ["min", "max", "abs"]:
    return nil
  let fnType = ctx.resolveCalleeProcType(calleeSym)
  if ctx.failing():
    return nil
  let args = ctx.lowerCallArgs(calleeSym, call)
  if ctx.failing():
    return nil
  let tmpName = ctx.uniqueName("tmp")
  let ret = fnType.ret
  ctx.hoisted.add(varD(tmpName, ret))
  ctx.hoisted.add(asgnS(sym(tmpName, ret),
    callE(sym(calleeSym.strVal, fnType, true), args)))
  ctx.declareLocal(tmpName, sym(tmpName, ret))
  return sym(tmpName, ret)

proc lowerCallValue(ctx: var FrontendCtx, callee: NimNode, call: NimNode): CirExpr =
  ## A value-producing call binds to a fresh temp: the `asgn` RHS hoists
  ## in front of the enclosing statement. Returns the temp symbol.
  ## A magic callee (the system operators) lowers to the primitive
  ## operation directly. No proc harvests, no call emits.
  let magic = nimMagicName(callee)
  if isBodilessMagic(callee):
    let arith = ctx.lowerMagicPrimitive(callee, call, magic)
    if ctx.failing():
      return nil
    if arith.isNil:
      ctx.fail("magic call " & callee.repr & " (" & magic & ") is outside the supported set")
      return nil
    let tmpName = ctx.uniqueName("tmp")
    let ret = getExprType(arith)
    ctx.hoisted.add(varD(tmpName, ret))
    ctx.hoisted.add(asgnS(sym(tmpName, ret), arith))
    ctx.declareLocal(tmpName, sym(tmpName, ret))
    return sym(tmpName, ret)
  let calleeSym = ctx.resolveCallCallee(callee, call)
  if ctx.failing():
    return nil
  let numeric = ctx.numericBuiltinCall(calleeSym, call)
  if ctx.failing():
    return nil
  if numeric != nil:
    return numeric
  let cirName = ctx.computeInstProcName(calleeSym)
  # a bodiless magic took the primitive path above
  # every other callee harvests, a magic with a Nim body (like `max`) lowers through it
  if not hasPragmaNim(calleeSym, "builtin"):
    ctx.harvestProc(calleeSym, cirName)
    if ctx.failing():
      return nil
  let fnType = ctx.resolveCalleeProcType(calleeSym)
  if ctx.failing():
    return nil
  let args = ctx.lowerCallArgs(calleeSym, call)
  if ctx.failing():
    return nil
  # a builtin callee emits the canonical native name, an instantiation hash would
  # name a proc the backend never defines. Builtin-marking the callee symbol
  # lets the alias canonicalization pass kind-assign from the canonical tables
  let calleeBuiltin = hasPragmaNim(calleeSym, "builtin")
  let calleeName = if calleeBuiltin: calleeSym.strVal
                   else: cirName
  let tmpName = ctx.uniqueName("tmp")
  let ret = fnType.ret
  ctx.hoisted.add(varD(tmpName, ret))
  ctx.hoisted.add(asgnS(sym(tmpName, ret),
    callE(sym(calleeName, fnType, calleeBuiltin), args)))
  ctx.declareLocal(tmpName, sym(tmpName, ret))
  return sym(tmpName, ret)

# ═══════════════════════════════════════════════════════════════════════
# Expression translation
# ═══════════════════════════════════════════════════════════════════════

proc isLatticeScalar(ctx: var FrontendCtx, n: NimNode): bool =
  ## True when the expression's type resolves to a lattice scalar: bool, int, uint, float.
  ## The primitive arithmetic applies to those.
  ##
  ## An overloaded operand (a register tile, an element-wise `-.`/`*.` proc)
  ## lowers as the proc's call instead.
  let t = ctx.resolver.resolveCirType(n.getTypeInst())
  if ctx.failing():
    return false
  return t != nil and t.kind in {ctBool, ctInt, ctUint, ctFloat}

proc lowerInfix(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## Operators: arithmetic and bitwise directly, comparisons normalized to
  ## the lattice's four kinds (`>` and `>=` swap operands), boolean
  ## `and`/`or` materialize into a temp through `ite` statements.
  let op = n[0].strVal
  case op
  of "+", "-", "*", "div", "mod", "shl", "shr", "/", "and", "or", "xor":
    if not (ctx.isLatticeScalar(n[1]) and ctx.isLatticeScalar(n[2])):
      # overloaded operator over non-scalar operands: the lattice resolution
      # picks the proc, lowered here as the call it is.
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
    # bool domain: short-circuit `and`/`or` materialize into a temp
    # through ite statements. `xor` evaluates both sides and compares.
    # int domain: the bitwise operators.
    let at = getExprType(a)
    if at != nil and at.kind == ctBool:
      let tmpName = ctx.uniqueName("condtmp")
      let boolT = tBool()
      let (bVal, bHoists) = block:
        # right operand's own hoists belong inside the branch evaluating it, short-circuit
        let saved = ctx.hoisted
        ctx.hoisted = @[]
        let lowered = ctx.lowerOperandExpr(n[2])
        let inner = ctx.hoisted
        ctx.hoisted = saved
        (lowered, inner)
      if ctx.failing():
        return nil
      let setTrue = asgnS(sym(tmpName, boolT), trueE())
      let setFalse = asgnS(sym(tmpName, boolT), falseE())
      if op == "xor":
        let b = bVal
        result = neqE(a, b)
        return result
      if op == "and":
        ctx.hoisted.add(varD(tmpName, boolT, falseE()))
        ctx.hoisted.add(iteS(a, bHoists & iteS(bVal, @[setTrue]), @[]))
      else:
        ctx.hoisted.add(varD(tmpName, boolT, trueE()))
        ctx.hoisted.add(iteS(a, @[], bHoists & iteS(bVal, @[setTrue], @[setFalse])))
      ctx.declareLocal(tmpName, sym(tmpName, boolT))
      return sym(tmpName, boolT)
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
    if opType.isNil:
      ctx.fail("operator " & op & " has no derivable operand type")
      return nil
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
    if opType.isNil or opType.kind != ctFloat:
      ctx.fail("operator / on a non-float operand type: use div for integers")
      return nil
    result = divE(opType, a, operandLiteralTypedAs(b, opType))
    return result
  else:
    ctx.fail("operator " & op & " is outside the supported set")
    return nil

proc buildBranchBody(ctx: var FrontendCtx, bodyNode: NimNode, dest: CirExpr): seq[CirStmt]
  ## Forward declaration, the if forms share the branch builder.

proc lowerIfForm(ctx: var FrontendCtx, branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode, dest: CirExpr): seq[CirStmt]
  ## Forward declaration, the if expression reuses the statement forms.

proc lowerIfExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## An if-expression:
  ## - a hoisted temp, one per expression
  ## - the statement form lowers with each branch assigning that temp
  let resType = ctx.resolver.resolveCirType(n.getTypeInst())
  if ctx.failing():
    return nil
  let tmpName = ctx.uniqueName("iftmp")
  ctx.declareLocal(tmpName, sym(tmpName, resType))
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
  let stmts = ctx.lowerIfForm(branches, elseBody, sym(tmpName, resType))
  if ctx.failing():
    return nil
  ctx.hoisted.add(@[varD(tmpName, resType)] & stmts)
  return sym(tmpName, resType)

proc lowerSplicedStmt(ctx: var FrontendCtx, c: NimNode): seq[CirStmt] =
  ## One non-final child of an expression block.
  ## The child is a statement of the spliced expansion.
  ## It hoists in front of the enclosing statement.
  ##
  ## A discarded construction step of the expansion drops:
  ## - the step shapes are an object or a tuple with no call inside
  ## - such a step has no runtime code
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
  ## Returns the literal `e` re-typed to `typ`, the value unchanged.
  ## A conversion of a literal is exact, the re-type replaces the conversion.
  case e.kind
  of eIntLit:
    intLit(e.value, typ)
  of eFloatLit:
    fltLit(e.fValue, typ)
  else:
    e


proc lowerExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## Translates one expression node.
  ## Hoisted statements (fresh temps, unnested call bindings) sit in front of the enclosing statement.
  if ctx.failing():
    return nil
  if n.isNil:
    ctx.fail("missing expression")
    return nil
  case n.kind
  of nnkSym:
    return ctx.lowerSym(n)
  of nnkIdent, nnkOpenSymChoice, nnkClosedSymChoice:
    ctx.fail("unresolved symbol " & n.repr & " in expression position")
    return nil
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
  of nnkNilLit:
    let typ = ctx.resolver.resolveCirType(n.getTypeInst())
    if ctx.failing():
      return nil
    return nilE(typ)
  of nnkHiddenStdConv:
    # int/int32 share one lattice width, the standard conversion is the identity.
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
    if bt != nil and bt.kind == ctAptr:
      ctx.fail("deref of an array pointer: index it instead")
      return nil
    return derefE(base)
  of nnkHiddenAddr, nnkAddr:
    let base = ctx.lowerExpr(n[0])
    if ctx.failing():
      return nil
    let bt = getExprType(base)
    if bt.isNil:
      ctx.fail("addr of an expression with no derivable type")
      return nil
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
    if baseNode.typeKind == ntyTypeDesc:
      ctx.fail("type-level index " & n.repr & ": compile-time layout arithmetic is folded by Nim before lowering")
      return nil
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
    if bt.isNil:
      ctx.fail("index base with no derivable type: " & n.repr)
      return nil
    case bt.kind
    of ctAptr:
      return patE(loweredBase, idx)
    of ctArray:
      return atE(loweredBase, idx)
    of ctPtr:
      # a var parameter of array type names the array through a pointer
      # index selects one element of the array that pointer names
      let pointee = bt.pElem
      if pointee.isNil or pointee.kind != ctArray:
        ctx.fail("indexing over a pointer to " & $(pointee.kind) &
          " is outside the supported set")
        return nil
      return patE(derefE(loweredBase), idx)
    of ctStruct:
      # tuple-style `[i]` access over a named struct: the field of that index
      if idx.kind != eIntLit:
        ctx.fail("tuple index must be a literal")
        return nil
      if idx.value < 0 or idx.value >= bt.fields.len:
        ctx.fail("tuple index out of range: " & $idx.value)
        return nil
      return dotE(loweredBase, bt.fields[idx.value].name)
    else:
      ctx.fail("indexing over type " & $(bt.kind) & " is outside the supported set")
      return nil
  of nnkDotExpr:
    if n[0].typeKind == ntyTypeDesc:
      ctx.fail("type-level field access " & n.repr & ": compile-time layout algebra is folded by Nim before lowering")
      return nil
    let base = ctx.lowerExpr(n[0])
    if ctx.failing():
      return nil
    if base.kind == eOconstr:
      # a field read of a construction is the field's value, no runtime temporary materializes
      for f in base.kvFields:
        if f.name == sanitizeIdent(n[1].strVal):
          return f.value
      ctx.fail("field " & n[1].strVal & " is not part of the constructed value")
      return nil
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
      let opType = getExprType(operand)
      if opType.isNil:
        ctx.fail("negation of an expression with no derivable type")
        return nil
      return negE(opType, operand)
    of "not":
      return notE(operand)
    else:
      ctx.fail("prefix operator " & op & " is outside the supported set")
      return nil
  of nnkCall, nnkCommand:
    let callee = n[0]
    if callee.kind notin {nnkSym, nnkOpenSymChoice, nnkClosedSymChoice}:
      ctx.fail("call callee " & callee.repr & " is not a proc symbol")
      return nil
    if callee.getImpl().kind in {nnkTemplateDef, nnkMacroDef}:
      ctx.fail("unexpanded template/macro call " & callee.repr &
        ": templates must resolve before the kernel lowers")
      return nil
    if callee.getImpl().kind in {nnkIteratorDef}:
      ctx.fail("iterator call " & callee.repr & " outside a for loop")
      return nil
    return ctx.lowerCallValue(callee, n)
  of nnkObjConstr:
    var fields: seq[CirKv] = @[]
    var valueSpaces = initTable[string, CirAddrSpace]()
    for i in 1 ..< n.len:
      n[i].expectKind(nnkExprColonExpr)
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
    # omitted fields zero-initialize, an empty initializer list is invalid C
    if fields.len < typ.fields.len:
      var ordered: seq[CirKv] = @[]
      for f in typ.fields:
        var value: CirExpr = nil
        for kvf in fields:
          if kvf.name == f.name:
            value = kvf.value
            break
        ordered.add(if value != nil: kv(f.name, value)
                    else: kv(f.name, zeroValue(f.typ)))
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
    if typ.kind != ctArray:
      ctx.fail("array literal over a non-array type")
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
    ctx.fail("parenthesized expression list")
    return nil
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
  of nnkCheckedFieldExpr:
    ctx.fail("case object field access: variant records have no C representation")
    return nil
  of nnkStrLit, nnkRStrLit, nnkTripleStrLit:
    ctx.fail("string literal: GPU kernels carry no strings")
    return nil
  of nnkCharLit:
    ctx.fail("char literal: outside the supported set")
    return nil
  of nnkStmtListType:
    ctx.fail("statement-list type in expression position: " & n.repr)
    return nil
  else:
    ctx.fail("expression construct not supported: " & $n.kind & " " & n.repr)
    return nil

# ═══════════════════════════════════════════════════════════════════════
# Conditions: the two-target compiler (Appel Cx)
# ═══════════════════════════════════════════════════════════════════════

proc containsShortCircuit(n: NimNode): bool =
  ## True when a boolean `and`/`or` occurs anywhere in the node,
  ## `StmtListExpr`-wrapped operands included.
  if n.isNil:
    return false
  case n.kind
  of nnkInfix:
    if n[0].strVal in ["and", "or"] and n[1].getTypeInst().typeKind == ntyBool:
      return true
  of nnkPrefix:
    if n[0].strVal == "not":
      return containsShortCircuit(n[1])
  else:
    discard
  for c in n:
    if containsShortCircuit(c):
      return true
  return false

proc genCondBoolExpr(ctx: var FrontendCtx, n: NimNode): CirExpr =
  ## Boolean expression of a call-free condition:
  ## - an `and`/`or` tree nests into `eAnd`/`eOr`, no short-circuit needed,
  ##   the operands are side-effect-free
  ## - `not` negates
  ## - any other leaf lowers as an expression (a comparison, a bool local)
  let node = unwrapConv(n)
  if node.kind == nnkInfix and node[0].strVal in ["and", "or"] and
      node[1].getTypeInst().typeKind == ntyBool:
    let a = ctx.genCondBoolExpr(node[1])
    if ctx.failing():
      return nil
    let b = ctx.genCondBoolExpr(node[2])
    if ctx.failing():
      return nil
    if node[0].strVal == "and":
      return andE(a, b)
    return orE(a, b)
  if node.kind == nnkPrefix and node[0].strVal == "not":
    let a = ctx.genCondBoolExpr(node[1])
    if ctx.failing():
      return nil
    return notE(a)
  return ctx.lowerExpr(node)

proc branchOn(ctx: var FrontendCtx, cond: CirExpr, trueL, falseL, fallL: string): seq[CirStmt] =
  ## Guarded transfer on a ready boolean expression:
  ## - jumps to `trueL`/`falseL`, one explicit `jmp` per taken side
  ## - falls through (empty branch) to the other when that one is the textual follower
  # an empty target name means fall-through into the next statement.
  # A named target always gets an explicit jump: the verifier's lab/jmp
  # model has no fall-through into a lab
  if fallL.len == 0 and trueL.len == 0:
    result.add(iteS(notE(cond), @[jmpS(falseL)], @[]))
  elif fallL.len == 0 and falseL.len == 0:
    result.add(iteS(cond, @[jmpS(trueL)], @[]))
  else:
    result.add(iteS(cond, @[jmpS(trueL)], @[jmpS(falseL)]))

proc genLeaf(ctx: var FrontendCtx, n: NimNode, trueL, falseL, fallL: string): seq[CirStmt] =
  ## Guarded transfer for a pure boolean leaf:
  ## - jumps to `trueL`/`falseL`, one explicit `jmp` per taken side
  ## - falls through (empty branch) to the other when that one is the textual follower
  ## - the leaf's own hoisted statements (call bindings) prefix the guard
  let cond = ctx.lowerExpr(n)
  if ctx.failing():
    return @[]
  result = ctx.drainHoisted()
  result.add(ctx.branchOn(cond, trueL, falseL, fallL))

proc genCond(ctx: var FrontendCtx, n: NimNode, trueL, falseL, fallL: string): seq[CirStmt] =
  ## Compiles a condition against two targets over forward-only scoped
  ## `lab`/`jmp`:
  ## - jump to `trueL` when true, to `falseL` when false
  ## - fall through when that target equals `fallL`
  ## - `and` chains on the false target, `or` on the true target, `not` swaps targets
  ## - each join is a fresh shared label, a nested boolean stays flat
  if ctx.failing():
    return @[]
  let node = unwrapConv(n)
  if node.kind == nnkInfix and node[0].strVal in ["and", "or"] and
      node[1].getTypeInst().typeKind == ntyBool:
    # a call-free and/or tree folds into one boolean expression
    # every inner false (true) target of the chain reaches the same branch
    # a label chain cannot express that inside nested ites
    # that drop rule would lose the inner exits
    if not containsCall(node):
      let condE = ctx.genCondBoolExpr(node)
      if ctx.failing():
        return @[]
      return ctx.branchOn(condE, trueL, falseL, fallL)
    let a = unwrapConv(node[1])
    let b = unwrapConv(node[2])
    let z = ctx.freshLabel("cz")
    if node[0].strVal == "and":
      result = ctx.genCond(a, z, falseL, z) & labS(z) & ctx.genCond(b, trueL, falseL, fallL)
    else:
      result = ctx.genCond(a, trueL, z, z) & labS(z) & ctx.genCond(b, trueL, falseL, fallL)
    return result
  if node.kind == nnkPrefix and node[0].strVal == "not":
    return ctx.genCond(node[1], falseL, trueL, fallL)
  return ctx.genLeaf(node, trueL, falseL, fallL)

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc drainHoisted(ctx: var FrontendCtx): seq[CirStmt] =
  ## Returns and clears the statements hoisted during the last lowering.
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
  if value.isNil:
    ctx.fail("branch without a value in an if expression")
    return @[]
  return stmts & ctx.drainHoisted() & @[asgnS(dest, value)]

proc lowerIfForm(ctx: var FrontendCtx, branches: seq[tuple[cond, body: NimNode]], elseBody: NimNode, dest: CirExpr): seq[CirStmt] =
  ## Forms produced, by branch shape:
  ## - one branch, leaf condition, statement form: a single `ite`
  ## - one branch, leaf condition, expression form: `ite` over both bodies
  ## - short-circuit condition: the two-target compiler over `then`/`merge`
  ##   labels
  ## - elif chains: flat, every branch jumps one shared merge label
  if ctx.failing():
    return @[]
  let thenStmts = ctx.buildBranchBody(branches[0].body, dest)
  if ctx.failing():
    return @[]
  if branches.len == 1 and not containsShortCircuit(branches[0].cond):
    let cond = ctx.lowerExpr(unwrapConv(branches[0].cond))
    if ctx.failing():
      return @[]
    if not elseBody.isNil:
      let elseStmts = ctx.buildBranchBody(elseBody, dest)
      if ctx.failing():
        return @[]
      return ctx.drainHoisted() & @[iteS(cond, thenStmts, elseStmts)]
    if dest.isNil:
      return ctx.drainHoisted() & @[iteS(cond, thenStmts, @[])]
  if elseBody.isNil and dest != nil:
    ctx.fail("if expression without an else branch")
    return @[]
  # flat elif chain over one shared merge label
  let mergeL = ctx.freshLabel("merge")
  var targeted = false
  result = @[]
  for i, br in branches:
    let body = ctx.buildBranchBody(br.body, dest)
    if ctx.failing():
      return @[]
    if containsShortCircuit(br.cond):
      let thenL = ctx.freshLabel("then")
      let nextL = ctx.freshLabel("next")
      result.add(ctx.genCond(br.cond, thenL, nextL, thenL))
      result.add(labS(thenL))
      result.add(body)
      result.add(jmpS(mergeL))
      result.add(labS(nextL))
      targeted = true
    else:
      let cond = ctx.lowerExpr(unwrapConv(br.cond))
      if ctx.failing():
        return @[]
      # a branch condition's own hoists run only after the earlier
      # branches' tests failed: they sit between the previous branch's
      # test and this branch's test
      result.add(ctx.drainHoisted() & iteS(cond, body & jmpS(mergeL), @[]))
      targeted = true
  if not elseBody.isNil:
    let elseStmts = ctx.buildBranchBody(elseBody, dest)
    if ctx.failing():
      return @[]
    result.add(elseStmts)
  if targeted:
    result.add(labS(mergeL))

proc lowerIfStmt(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## An `if` statement: branches and optional else.
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
  if branches.len == 0:
    ctx.fail("if statement without a branch")
    return @[]
  return ctx.lowerIfForm(branches, elseBody, nil)

func isUnrollPragma(n: NimNode): bool =
  ## True for the statement pragma `{.unroll.}`.
  if n.kind != nnkPragma:
    return false
  for p in n:
    if p.kind in {nnkIdent, nnkSym} and p.strVal == "unroll":
      return true
  return false

func bodyMarksUnroll(body: NimNode): bool =
  ## True when the loop body states `{.unroll.}` directly: the pragma is
  ## one statement of the body's own statement list.
  ## V1 semantics:
  ## - any position inside the body marks the loop
  ## - a pragma nested inside an inner loop's body marks that inner loop
  if body.kind != nnkStmtList:
    return false
  for stmt in body:
    if isUnrollPragma(stmt):
      return true
  return false

proc lowerWhile(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## A `while` loop lowers to a conditionless `loop` with a leading guard
  ## and the exit label after the loop (final_ir.md rule 2).
  if ctx.failing():
    return @[]
  let exitL = ctx.freshLabel("lx")
  let unroll = bodyMarksUnroll(n[1])
  let body = ctx.lowerStmt(n[1])
  if ctx.failing():
    return @[]
  var guard: seq[CirStmt] = @[]
  if n[0].kind == nnkSym and n[0].strVal == "true":
    discard
  else:
    guard = ctx.genCond(n[0], "", exitL, "")
    if ctx.failing():
      return @[]
  result = @[loopS(guard & body, unroll), labS(exitL)]

proc lowerFor(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## A `for` over a literal range or `countup`:
  ## - a loop variable declared in front
  ## - a conditionless `loop`, its leading guard is the continue-condition
  ## - the increment last, the exit label after the loop
  if ctx.failing():
    return @[]
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
      if startNode.isNil or stopNode.isNil:
        ctx.fail("for loop over a Slice without a and b bounds: " & n.repr)
        return @[]
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
      stepIsConst = false
      stepV = ctx.lowerExpr(unwrapped)
  if ctx.failing():
    return @[]
  let stepT = operandLiteralTypedAs(stepV, varType)
  ctx.declareLocal(vars, sym(varName, varType))
  let exitL = ctx.freshLabel("lx")
  let contL = ctx.freshLabel("lc")
  let varSym = sym(varName, varType)
  let savedContinue = ctx.continueUsed
  ctx.continueUsed = false
  ctx.loopStack.add((exitL: exitL, contL: contL))
  let unroll = bodyMarksUnroll(n[^1])
  # start/stop/step hoists queued so far belong to the preamble, the body's own hoists drain per statement inside it
  let startHoists = ctx.drainHoisted()
  let body = ctx.lowerStmtBody(n[^1])
  let bodyHoists = ctx.drainHoisted()
  let contUsed = ctx.continueUsed
  ctx.loopStack.del(ctx.loopStack.high)
  ctx.continueUsed = savedContinue
  if ctx.failing():
    return @[]
  let cmp = if inclusive:
    leE(varSym, stopT)
  else:
    ltE(varSym, stopT)
  let guard = iteS(cmp, @[], @[jmpS(exitL)])
  var incr: seq[CirStmt] = @[]
  if contUsed:
    incr.add(labS(contL))
  incr.add(asgnS(varSym, addE(varType, varSym, stepT)))
  var preamble = ctx.drainHoisted()
  preamble.add(varD(varName, varType))
  preamble.add(asgnS(varSym, startT))
  if not stepIsConst:
    preamble.add(assumeS(ltE(intLit(0, varType), stepT)))
  result = startHoists & preamble & @[
    loopS(bodyHoists & guard & body & incr, unroll), labS(exitL)]

proc lowerReturn(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## A `return`: the value binds to its destination, then the `ret`.
  ## A call value binds through `asgn result` first.
  if ctx.failing():
    return @[]
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
  if ctx.retType.kind == ctVoid:
    ctx.fail("value returned from a void proc")
    return @[]
  let hoists = ctx.drainHoisted()
  if value.kind == eCall:
    return hoists & @[asgnS(sym("result", ctx.retType), value),
        retS(sym("result", ctx.retType))]
  return hoists & @[retS(value)]

proc lowerVoidCall(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## A void call in statement position:
  ## - `threadgroup_barrier()`
  ## - a kernel call
  ##
  ## The only statement-position call the normal form allows.
  if ctx.failing():
    return @[]
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
  if callee.kind == nnkSym and
      callee.getImpl().kind in {nnkTemplateDef, nnkMacroDef}:
    ctx.fail("unexpanded template/macro call " & callee.repr &
      ": templates must resolve before the kernel lowers")
    return @[]
  let calleeSym = ctx.resolveCallCallee(callee, n)
  if ctx.failing():
    return @[]
  let fnType = ctx.resolveCalleeProcType(calleeSym)
  if ctx.failing():
    return @[]
  let cirName = ctx.computeInstProcName(calleeSym)
  let calleeBuiltin = hasPragmaNim(calleeSym, "builtin")
  let calleeName = if calleeBuiltin: calleeSym.strVal
                   else: cirName
  if isBodilessMagic(callee):
    ctx.fail("magic call " & callee.repr & " in statement position without a destination")
    return @[]
  if fnType.ret.kind != ctVoid:
    # a discardable non-void call in statement position: the value is
    # computed into a fresh temp and dropped
    if not hasPragmaNim(calleeSym, "builtin"):
      ctx.harvestProc(calleeSym, cirName)
      if ctx.failing():
        return @[]
    let args = ctx.lowerCallArgs(calleeSym, n)
    if ctx.failing():
      return @[]
    let tmpName = ctx.uniqueName("tmp")
    ctx.hoisted.add(varD(tmpName, fnType.ret))
    ctx.hoisted.add(asgnS(sym(tmpName, fnType.ret),
      callE(sym(calleeName, fnType, calleeBuiltin), args)))
    return ctx.drainHoisted()
  if not hasPragmaNim(calleeSym, "builtin"):
    ctx.harvestProc(calleeSym, cirName)
    if ctx.failing():
      return @[]
  let args = ctx.lowerCallArgs(calleeSym, n)
  if ctx.failing():
    return @[]
  return ctx.drainHoisted() & @[callS(sym(calleeName, fnType, calleeBuiltin), args)]

proc resolveLocalSpace(ctx: var FrontendCtx, nameNode: NimNode): CirAddrSpace =
  ## Memory class of a local declaration. The pragma list decides:
  ## - `smem` → threadgroup memory
  ## - `const_mem` → constant memory
  ## - `rmem` or no annotation → per-thread registers, the `asRMEM`
  ##   default of the type constructors this replaces
  result = caThread
  if nameNode.kind != nnkPragmaExpr:
    return
  var seen = false
  for p in nameNode[1]:
    case p.strVal.toLowerAscii
    of "smem":
      if seen:
        ctx.fail("multiple address-space pragmas on one local")
        return
      seen = true
      result = caThreadgroup
    of "const_mem":
      if seen:
        ctx.fail("multiple address-space pragmas on one local")
        return
      seen = true
      result = caConstant
    of "rmem":
      if seen:
        ctx.fail("multiple address-space pragmas on one local")
        return
      seen = true
      result = caThread
    of "noinit", "inject", "gensym":
      discard
    else:
      ctx.fail("unexpected pragma " & p.repr & " on a local declaration")

proc lowerLetSection(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## A `let`/`var` section:
  ## - one typed declaration per name
  ## - a call-valued initializer binds through an `asgn` after the declaration
  ## - the memory class comes from the declaration's pragma list: `smem`,
  ##   `const_mem`, `rmem`, unannotated is per-thread registers
  if ctx.failing():
    return @[]
  for defs in n:
    if defs.kind == nnkVarTuple:
      ctx.fail("tuple unpacking declaration")
      return @[]
    if defs.kind notin {nnkIdentDefs, nnkConstDef}:
      ctx.fail("declaration: unexpected child " & $defs.kind)
      return @[]
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
  if ctx.failing():
    return @[]
  for defs in n:
    defs.expectKind(nnkConstDef)
    let nameNode = defs[0]
    var nimName = if nameNode.kind == nnkPragmaExpr: nameNode[0].strVal
                  else: nameNode.strVal
    let typeNode = defs[1]
    let valueNode = defs[2]
    if valueNode.kind in {nnkCall, nnkCommand}:
      ctx.fail("const " & nimName & " initialized by a call: layout arithmetic is folded by Nim before lowering")
      return @[]
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
  ## True when a statement-position compound assign's operands are lattice
  ## scalars and the primitive arithmetic applies: the left operand goes
  ## through its `var` pointer when the target is a var parameter.
  ##
  ## An overloaded operand (a register tile, an element-wise `op=` proc)
  ## lowers as the proc's call instead.
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
  ## Translates one statement node.
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
    if n[1].kind in {nnkCall, nnkCommand}:
      # destination-bound: the call binds directly to the assignment
      let callee = n[1][0]
      if callee.kind notin {nnkSym, nnkOpenSymChoice, nnkClosedSymChoice}:
        ctx.fail("call callee " & callee.repr & " is not a proc symbol")
        return @[]
      # a magic callee lowers to the primitive operation as the right-hand side
      if isBodilessMagic(callee):
        let magic = nimMagicName(callee)
        let arith = ctx.lowerMagicPrimitive(callee, n[1], magic)
        if ctx.failing():
          return @[]
        if arith.isNil:
          ctx.fail("magic call " & callee.repr & " (" & magic & ") is outside the supported set")
          return @[]
        return ctx.drainHoisted() & @[asgnS(lhs, arith)]
      let calleeSym = ctx.resolveCallCallee(callee, n[1])
      if ctx.failing():
        return @[]
      let numeric = ctx.numericBuiltinCall(calleeSym, n[1])
      if ctx.failing():
        return @[]
      if numeric != nil:
        return ctx.drainHoisted() & @[asgnS(lhs, numeric)]
      let fnType = ctx.resolveCalleeProcType(calleeSym)
      if ctx.failing():
        return @[]
      let cirName = ctx.computeInstProcName(calleeSym)
      if not hasPragmaNim(calleeSym, "builtin"):
        ctx.harvestProc(calleeSym, cirName)
        if ctx.failing():
          return @[]
      let args = ctx.lowerCallArgs(calleeSym, n[1])
      if ctx.failing():
        return @[]
      # a builtin callee emits the canonical native name, an instantiation hash would
      # name a proc the backend never defines. Builtin-marking the callee symbol
      # lets the alias canonicalization pass kind-assign from the canonical tables
      let calleeBuiltin = hasPragmaNim(calleeSym, "builtin")
      let calleeName = if calleeBuiltin: calleeSym.strVal
                       else: cirName
      return ctx.drainHoisted() & @[asgnS(lhs,
        callE(sym(calleeName, fnType, calleeBuiltin), args))]
    let rhs = ctx.lowerExpr(n[1])
    if ctx.failing():
      return @[]
    # a literal rhs re-types to the destination, the conversion is exact
    let rhsT = operandLiteralTypedAs(rhs, getExprType(lhs))
    return ctx.drainHoisted() & @[asgnS(lhs, rhsT)]
  of nnkInfix:
    # compound assignment reaches the statement position as an infix:
    # lattice-scalar operands take the primitive arithmetic, any other `op=`
    # (`qk -.= m_ij` over register tiles) resolves to its overloaded proc
    # and lowers as a var-target call in statement position
    let op = n[0].strVal
    if op in ["+=", "-=", "*=", "/="] and ctx.isScalarCompoundAssign(n):
      var lhs = ctx.lowerExpr(n[1])
      if ctx.failing():
        return @[]
      if lhs.kind == eAddr:
        # a var-returning `[]` hands the compound assignment the element's address
        # and the assignment target is the location behind it
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
    if n[0].kind in {nnkSym, nnkIdent}:
      ctx.fail("labeled break")
      return @[]
    if ctx.loopStack.len == 0:
      ctx.fail("break outside a loop")
      return @[]
    return @[jmpS(ctx.loopStack[^1].exitL)]
  of nnkContinueStmt:
    if ctx.loopStack.len == 0:
      ctx.fail("continue outside a loop")
      return @[]
    ctx.continueUsed = true
    return @[jmpS(ctx.loopStack[^1].contL)]
  of nnkCaseStmt:
    ctx.fail("case statement: outside the supported set")
    return @[]
  of nnkWhenStmt:
    ctx.fail("when statement: resolve statically before lowering")
    return @[]
  of nnkTypeSection:
    ctx.resolveTypeSection(n)
    return @[]
  of nnkPragma:
    var emitArgs: seq[NimNode] = @[]
    for p in n:
      if p.kind in {nnkExprColonExpr, nnkCall} and p.len >= 2 and
          p[0].kind in {nnkIdent, nnkSym} and p[0].strVal == "emit":
        emitArgs.add(p[1])
    if emitArgs.len == 0:
      # a non-emit statement pragma is a compile-time annotation, nothing emits
      return @[]
    var emitted: seq[CirStmt] = @[]
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
  else:
    ctx.fail("statement construct not supported: " & $n.kind & " " & n.repr)
    return @[]

proc lowerStmt(ctx: var FrontendCtx, n: NimNode): seq[CirStmt] =
  ## Translates one statement node, its hoists (fresh temps, call bindings) sitting in front of the statement.
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
  ## - the body lowers statement by statement, an implicit `ret` closes
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
      params.add(CirParam(name: pname, typ: typ))
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
  if ret.kind != ctVoid:
    result.add(retS(sym("result", ret)))
  else:
    result.add(retS(nil))

proc applyProcPragmas(fn: NimNode, p: CirProc) =
  ## Reads the public pragma surface off a proc definition: `global` marks the compute entry point,
  ## `forceinline` the inlining hint, `cudaName` the entry-point name override,
  ## `workgroup(x, y, z)` the baked launch shape. Other pragmas are legal and dropped.
  for prag in fn.pragma:
    case prag.kind
    of nnkIdent, nnkSym:
      case prag.strVal
      of "global":
        p.isKernel = true
      of "forceinline", "inline":
        # both marks splice the body into its callers, a device function
        # cannot take the thread builtins as attributed parameters
        p.forceInline = true
      else:
        discard
    of nnkExprColonExpr, nnkCall, nnkCommand:
      if prag.len >= 2:
        case prag[0].strVal
        of "cudaName":
          p.cudaName = prag[1].strVal
        of "workgroup":
          # both call and colon forms: `workgroup(x, y, z)`, `workgroup: (x, y, z)`
          var dims: NimNode =
            if prag.kind == nnkExprColonExpr:
              prag[1]
            else:
              prag
          if dims.kind in {nnkPar, nnkTupleConstr, nnkBracket}:
            for d in 0 ..< 3:
              # a missing trailing dimension defaults to 1: workgroup: (4, 2) is the (4, 2, 1) launch shape, 0 is not a legal GLSL/WGSL dimension
              if d < dims.len:
                p.workgroup[d] = dims[d].intVal.int
              else:
                p.workgroup[d] = 1
          elif dims.kind in nnkIntLit .. nnkUInt64Lit:
            p.workgroup[0] = dims.intVal.int
            p.workgroup[1] = 1
            p.workgroup[2] = 1
        else:
          discard
    else:
      discard

proc lowerProcDef(ctx: var FrontendCtx, fn: NimNode): CirProc =
  ## One proc definition:
  ## - parameters typed off Nim once, the body lowered into normal form
  ## - generic definitions skip here, their instantiations harvest at the calls that use them
  if ctx.failing():
    return nil
  if isGenericDef(fn):
    return nil
  let formals = getFormalParams(fn)
  if formals.isNil:
    ctx.fail("proc " & $fn[0] & " without formal parameters")
    return nil
  let savedName = ctx.fnName
  let savedRet = ctx.retType
  let savedCounts = ctx.nameCounts
  let savedScopes = ctx.scopes
  let savedLoops = ctx.loopStack
  let savedCont = ctx.continueUsed
  ctx.fnName = sanitizeIdent($fn[0])
  ctx.nameCounts = initTable[string, int]()
  ctx.scopes = @[ctx.moduleScope]
  ctx.localSpaces = @[ctx.moduleSpaces]
  ctx.loopStack = @[]
  ctx.continueUsed = false
  ctx.retType = ctx.resolver.resolveCirType(formals[0])
  if ctx.failing():
    return nil
  var params: seq[CirParam] = @[]
  let fnRet = ctx.retType
  let fnName = ctx.fnName
  let body = ctx.lowerProcBody(fn, formals, getProcBody(fn), params)
  ctx.fnName = savedName
  ctx.retType = savedRet
  ctx.nameCounts = savedCounts
  ctx.scopes = savedScopes
  ctx.loopStack = savedLoops
  ctx.continueUsed = savedCont
  if ctx.failing():
    return nil
  result = procD(fnName, params, fnRet, body)
  if ctx.procKeys.hasKey(result.name):
    ctx.fail("duplicate proc name " & result.name)
    return nil
  ctx.procKeys[result.name] = true
  applyProcPragmas(fn, result)
  ctx.procs.add(result)

proc harvestProc(ctx: var FrontendCtx, callee: NimNode, cirName: string) =
  ## Registers the instantiation of a generic (or external) proc behind
  ## a typed call. Inputs: the callee's `getTypeInst` (instantiated signature) and `getImpl` (body), compile-time parameters erased.
  if ctx.procKeys.hasKey(cirName):
    return
  ctx.procKeys[cirName] = true
  let impl = callee.getImpl()
  if impl.kind notin {nnkProcDef, nnkFuncDef}:
    ctx.fail("callee " & callee.repr & " has no proc body to harvest (" &
      $impl.kind & ")")
    return
  let formals = getFormalParams(callee.getTypeInst())
  if formals.isNil or formals.kind != nnkFormalParams:
    ctx.fail("callee " & callee.repr & " without an instantiated signature")
    return
  let savedName = ctx.fnName
  let savedRet = ctx.retType
  let savedCounts = ctx.nameCounts
  let savedScopes = ctx.scopes
  let savedLoops = ctx.loopStack
  let savedCont = ctx.continueUsed
  let savedProcsLen = ctx.procs.len
  let savedHoisted = ctx.hoisted
  # callee lowers with its own hoist queue: the caller's pending hoists
  # belong to the caller's statement, never to the callee's body
  ctx.hoisted = @[]
  ctx.fnName = cirName
  ctx.nameCounts = initTable[string, int]()
  ctx.scopes = @[ctx.moduleScope]
  ctx.localSpaces = @[ctx.moduleSpaces]
  ctx.loopStack = @[]
  ctx.continueUsed = false
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
  ctx.loopStack = savedLoops
  ctx.continueUsed = savedCont
  # callee's own pending hoists close its body, the caller's queue resumes
  let calleeHoists = ctx.drainHoisted()
  ctx.hoisted = savedHoisted
  if ctx.failing():
    ctx.procs.setLen(savedProcsLen)
    return
  let harvested = procD(cirName, params, fnRet, calleeHoists & body)
  applyProcPragmas(impl, harvested)
  ctx.procs.add(harvested)

# ═══════════════════════════════════════════════════════════════════════
# Entry points
# ═══════════════════════════════════════════════════════════════════════

proc lowerTopLevelDecls(ctx: var FrontendCtx, n: NimNode) =
  ## A top-level `var`/`const` section: program-scope storage.
  ##
  ## - the pragma list picks the memory class: `smem` threadgroup, `const_mem` constant, unannotated is program-scope constant
  ## - the declaration registers in the module scope, every proc body reads it
  ## - kernels must appear after the declarations they read, program order
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
    # a bare kernel proc symbol: translate its definition
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
    else:
      ctx.fail("top-level construct not supported: " & $c.kind & " " & c.repr)
      return

proc lowerCirv2Unverified*(body: NimNode): tuple[m: CirModule, err: string] =
  ## Lowers the typed macro body to CIRv2 without verifying the normal form.
  ## Returns the module and "" when clean, else the first refusal message.
  ##
  ## Pipeline order:
  ## - the constant-shell fold and the structuring pass run on this output
  ## - verification runs after those passes
  var ctx = FrontendCtx()
  ctx.pushScope()
  ctx.lowerModule(body)
  let m = moduleD("kernel", ctx.resolver.structs, ctx.procs)
  m.globals = ctx.moduleGlobals
  return (m, ctx.resolver.err)

proc tryLowerCirv2*(body: NimNode): tuple[m: CirModule, err: string] =
  ## Pure lowering entry: returns the verified CIRv2 module and "" when clean,
  ## else the first refusal message (the zero-tolerance channel).
  let (m, err) = lowerCirv2Unverified(body)
  if err.len == 0:
    let vmsg = verifyMessage(m)
    if vmsg.len > 0:
      return (m, vmsg)
  return (m, err)

proc toCirv2Module*(body: NimNode): CirModule =
  ## Lowers the typed macro body to CIRv2 and verifies the normal form.
  ## Returns the module. A construct outside the supported closure rejects
  ## through `error`, naming the construct.
  let (m, err) = tryLowerCirv2(body)
  if err.len > 0:
    error(err)
  return m
