# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, math, sets, strutils, tables]
import ../ir/cirv2_lang
import ../../builtins/builtins_gpu_types
import ../../crucible_codegen
import ../../builtins/builtins_decl_helpers
import ./targets_registry
export targets_registry

# ############################################################
#
#            Shared print walk
#
# ############################################################

template failing*(ctx: auto): bool =
  ctx.err.len > 0

template fail*(ctx: auto, detail: string) =
  ## Records the first refusal of the walk in `ctx.err`:
  ##
  ## - message: `cirv2 printer <printerTarget>: at <fnName>: <detail>`
  ## - `printerTarget`: the target name each importing printer defines
  ## - first refusal wins, later refusals never overwrite it
  ## - the walk keeps going and the printer returns the recorded message
  mixin printerTarget
  if ctx.err.len == 0:
    ctx.err = "cirv2 printer " & printerTarget & ": at " & ctx.fnName & ": " & detail

## Refusal text for a nil literal in a target whose kind section refuses nil.
const nilLiteralRejection* = "nil literal is outside the target subset"

proc getUnhandledExprRejection*(kind: CirExprKind): string =
  ## Refusal text for an expression kind outside the printer's case.
  ## A routing bug: the dispatcher sends only the kinds the case lists.
  "expression kind " & $kind & " has no printer emission"

# ═══════════════════════════════════════════════════════════════════════
#  Print context
# ═══════════════════════════════════════════════════════════════════════

type BuiltinVocab* = object
  ## Builtin call vocabulary of one target keyed by the family enums:
  ## the call texts of the kinds the target provides natively.
  synchroText*: array[CirSynchroBuiltinKind, string]
  reductionText*: array[CirReductionBuiltinKind, string]
  shuffleText*: array[CirShuffleBuiltinKind, string]
  atomicsText*: array[CirAtomicsBuiltinKind, string]
  ## Call text in value position, `$1`..`$9` for the printed arguments.
  ## "" when the kind has no text: the printer passes the canonical name through,
  ## the target compiler rejects what it cannot parse.
  stmtText*: array[CirSynchroBuiltinKind, string]
  ## Statement-position call text of one single-line statement,
  ## `$1`..`$9` for the printed arguments.

func makeTexts*[K: enum](pairs: openArray[tuple[k: K, txt: string]]):
    array[K, string] =
  ## Sparse kind-indexed text table, kinds without an entry stay "".
  for p in pairs:
    result[p.k] = p.txt

type TargetEmit* = object
  ## Static emission knobs of one target: the builtin vocabulary plus the grammar choices of the universal walk.
  vocab*: BuiltinVocab
  functionalCtor*: bool
  ## Constructions print in the functional form `T(...)`, else braced.
  ctorOpen*, ctorClose*: string
  ## Construction tokens around the values, `{`/`}` braced or `(`/`)`
  ## functional (an overlay like HLSL spaces its braces).
  ctorCast*: bool
  ## A braced object construction parenthesizes the type name: `(T){...}`.
  ctorZero*: string
  ## Zero-field construction text, `$1` for the type name, an empty text prints the empty form.
  aconstrCasts*: bool
  ## Array construction elements cast to the element type.
  dotDerefParens*: bool
  ## A component read over a dereference parenthesizes the base `(base).x`.
  vectorDots*: bool
  ## Component reads compose the vector component letter (F0 -> x), an unset flag leaves the raw field name.
  noPointerValues*: tuple[spaces: set[AddressSpace], text: string]
  ## Refusal for a pointer symbol used as a value. The fingerprint text
  ## takes `$1` for the symbol name and fires for a pointer whose address
  ## space is in `spaces`. An empty set means pointers are values.
  noPointerRebind*: string
  ## Refusal fingerprint for a pointer rebind, `$1` for the symbol name, "" when pointers rebind.
  prefixKernel*, prefixInline*, prefixDevice*: string
  ## Linkage prefixes: kernel, inline device function, device function.

type WalkCtx* {.inheritable.} = object
  ## Shared walk state of one print run over one module and the base type
  ## for each printer's own `PrintCtx`.
  ##
  ## - the per-target derivation gives the shared procs one instantiation per target
  ## - the overlay procs resolve per target at that instantiation
  fnName*: string
  backend*: CodegenKind
  ## Name of the proc in flight, named by the refusal messages.
  err*: string
  ## First refusal of the walk, "" while clean.
  indent*: int
  policy*: WalkPolicy
  ## Walk policy of the target as the shared walk reads it.
  vectorRule*: VectorRule
  ## Vector coverage of the target as read by the declaration procs.
  emit*: TargetEmit
  ## Static emission knobs of the target.
  mathNames*: Table[(string, int), string]
  ## Registry math table of the target, read by the shared math dispatch.
  inKernel*: bool
  ## False inside a device function.
  builtins*: OrderedSet[CirCoordBuiltinKind]
  ## Coordinate builtin kinds that the body references, first-use order.
  workgroup*: array[3, int]
  ## Workgroup sizes of the entry in flight.

template initPrintCtx*(walkPolicy: WalkPolicy, rule: VectorRule, targetEmit: TargetEmit, bk: CodegenKind): untyped =
  ## Walk context of one module print: the target's tables, clean state.
  ## Expands against the importing printer's own `PrintCtx` type.
  ## A target with registry math entries assigns `mathNames` after the init.
  ##
  ## Returns: the initialized context, `fnName` the module-level kernel name.
  PrintCtx(fnName: "<module>", policy: walkPolicy, vectorRule: rule,
    emit: targetEmit, backend: bk)

proc roundUpMultiple*(v, multiple: int): int =
  doAssert multiple > 0
  (v + multiple - 1) div multiple * multiple

proc getScalarByteSize*(t: CirType, boolWidth: static int = 1): int =
  ## Byte width of one scalar value.
  ## A bool is `boolWidth` wide: std430 packs a bool into 1 byte and the WGSL launch ABI into 4.
  case t.kind
  of ctBool: boolWidth
  of ctInt, ctUint, ctFloat, ctBf16: t.bits div 8
  else: 0

proc emitVectorComponent*(name: string): string =
  case name
  of "F0": "x"
  of "F1": "y"
  of "F2": "z"
  of "F3": "w"
  else: ""

proc resolveVectorComponent*(ctx: var auto, field: string): string =
  ## Component letter for one vector field name, refusing on a miss.
  result = emitVectorComponent(field)
  if result.len == 0:
    ctx.fail("no vector component " & field)

proc printSubscriptParts*(ctx: var auto, e: CirExpr): tuple[base, idx: string] =
  ## Printed base and index texts of one subscript.
  let idx = if e.kind == eAt: e.atIndex else: e.patIndex
  let base = if e.kind == eAt: e.atBase else: e.patBase
  let idxTxt = ctx.printExpr(idx)
  var baseTxt = ctx.printExpr(base)
  if base.kind in {eCast, eConv}:
    baseTxt = "(" & baseTxt & ")"
  result = (baseTxt, idxTxt)

proc printVectorDot*(ctx: var auto, e: CirExpr): string =
  ## Text of one vector-component dot `base.x`, "" for a non-vector base.
  let bt = getExprType(e.dotBase)
  if bt == nil or bt.kind != ctVector:
    return ""
  let comp = ctx.resolveVectorComponent(e.fieldName)
  if ctx.failing():
    return ""
  return ctx.printExpr(e.dotBase) & "." & comp

# ═══════════════════════════════════════════════════════════════════════
#  Declarator engine
# ═══════════════════════════════════════════════════════════════════════

proc getScalarName*(names: ScalarNames, t: CirType): string =
  ## Scalar type name from the target's scalar-name table,
  ## "" when the table omits the (kind, width) cell.
  case t.kind
  of ctVoid, ctBool, ctString:
    for c in names.cells:
      if c.kind == t.kind:
        return c.name
  of ctInt, ctUint, ctFloat, ctBf16:
    for c in names.cells:
      if c.kind == t.kind and c.width == t.bits:
        return c.name
  else:
    discard
  return ""

proc getVectorName*(rule: VectorRule, names: ScalarNames, t: CirType): string =
  ## Vector type name from the target's vector rule, "" when the rule
  ## does not cover the element or the lane count.
  if t.kind != ctVector:
    return ""
  for e in rule.elems:
    if e.kind == t.vElem.kind and e.width == t.vElem.bits:
      if t.vN notin rule.lanes:
        return ""
      case rule.nameForm
      of vnfPrefixed:
        let prefix = if e.prefix.len > 0: e.prefix else: getScalarName(names, t.vElem)
        return prefix & $t.vN
      of vnfGenericElem:
        return e.prefix & $t.vN & "<" & getScalarName(names, t.vElem) & ">"
  return ""

proc getLeafTypeName*(names: ScalarNames, rule: VectorRule, t: CirType): string =
  case t.kind
  of ctVoid, ctBool, ctInt, ctUint, ctFloat, ctBf16, ctString:
    result = getScalarName(names, t)
  of ctVector:
    result = getVectorName(rule, names, t)
  of ctStruct:
    result = t.name
  of ctPtr, ctAptr, ctArray, ctProc, ctSpan:
    result = ""

proc getFullTypeName*(names: ScalarNames, rule: VectorRule, t: CirType): string =
  if t.kind in {ctPtr, ctAptr}:
    return getFullTypeName(names, rule, t.pElem) & "*"
  return getLeafTypeName(names, rule, t)

proc printCDecl*(names: ScalarNames, rule: VectorRule, target: string, t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## Declaration text of `t` with `ident` appended in the C declarator form:
  ## the array dimensions move behind the identifier and nest into the declarator
  ## (`T ident[N]`, pointer-to-array `T (*ident)[N]`).
  ##
  ## - `allowEmptyIdent` permits no identifier, return types and casts use it
  ## - the printer refuses an array without an identifier with the `<target: array needs an identifier>` text
  ## - a scalar leaf absent from the scalar names yields an empty type name,
  ##   the target compiler rejects the malformed declaration
  if t.kind == ctArray:
    if ident.len == 0 and not allowEmptyIdent:
      return "<" & target & ": array needs an identifier>"
  if t.kind in {ctPtr, ctAptr} and t.pElem.kind == ctArray:
    # array dimensions bind behind the identifier, one pointer star inside the parenthesized declarator
    return printCDecl(names, rule, target, t.pElem, "(" & '*' & ident & ")",
      allowEmptyIdent)
  if t.kind == ctArray:
    if t.aElem.kind == ctArray:
      # a pointer declarator is already parenthesized (`(*ident)`), one more parenthesized layer would print `((*ident))`
      let declarator = if ident.startsWith('('): ident else: "(" & ident & ")"
      return printCDecl(names, rule, target, t.aElem, declarator & "[" & $t.aN & "]",
        allowEmptyIdent)
    return getFullTypeName(names, rule, t.aElem) & ' ' & ident & '[' & $t.aN & ']'
  result = getFullTypeName(names, rule, t)
  if ident.len > 0:
    result.add ' ' & ident

proc getNonFiniteFloatLit*(e: CirExpr, tok: FloatLitTokens): string =
  ## Emission token of a non-finite float literal, "" when the value is finite.
  case e.fValue.classify
  of fcInf:
    case e.fLitType.bits
    of 16: result = tok.inf16
    of 32: result = tok.inf32
    else: result = tok.inf64
  of fcNegInf:
    case e.fLitType.bits
    of 16: result = tok.negInf16
    of 32: result = tok.negInf32
    else: result = tok.negInf64
  of fcNan:
    case e.fLitType.bits
    of 16: result = tok.nan16
    of 32: result = tok.nan32
    else: result = tok.nan64
  else:
    result = ""

template emitLiteral*(e: CirExpr, tokens: LitTokens, names: ScalarNames): string =
  ## Literal token of `e` over the target's literal tokens.
  mixin printerTarget
  block:
    var lit: string
    case e.kind
    of eIntLit:
      if e.litType.bits notin {32, 64} and tokens.int.narrowCast:
        lit = "(" & getScalarName(names, e.litType) & ")" & $e.value
      else:
        case e.litType.kind
        of ctUint:
          case e.litType.bits
          of 64: lit = tokens.int.u64Prefix & $e.value & tokens.int.u64Suffix
          of 32: lit = $e.value & tokens.int.u32Suffix
          else: lit = $e.value
        else:
          case e.litType.bits
          of 64: lit = tokens.int.i64Prefix & $e.value & tokens.int.i64Suffix
          else: lit = $e.value
    of eFloatLit:
      lit = getNonFiniteFloatLit(e, tokens.nonFinite)
      if lit.len == 0:
        case e.fLitType.bits
        of 16: lit = tokens.f16Prefix & $e.fValue & tokens.f16Suffix
        of 32: lit = $e.fValue & tokens.f32Suffix
        else: lit = tokens.f64Prefix & $e.fValue & tokens.f64Suffix
    of eNil:
      lit = tokens.nilToken
      if lit.len == 0:
        lit = "<" & printerTarget & ": not a literal>"
    of eStrLit:
      lit = '"' & e.sValue & '"'
    of eTrue: lit = "true"
    of eFalse: lit = "false"
    else:
      lit = "<" & printerTarget & ": not a literal>"
    lit

proc matchCastRule(rules: openArray[CastRule], src: CirType, dst: CirType): int =
  ## Index of the first cast rule that matches a cast from `src` to `dst`,
  ## -1 when none does. A nil `src` matches only `srcAny` rules.
  for i, r in rules:
    if not r.srcAny:
      if src == nil or src.kind notin r.srcKinds:
        continue
      if r.srcWidths.len > 0 and src.bits notin r.srcWidths:
        continue
    if dst.kind notin r.dstKinds:
      continue
    if r.dstWidths.len > 0 and dst.bits notin r.dstWidths:
      continue
    if r.sameWidth and (src == nil or src.bits != dst.bits):
      continue
    return i
  return -1

template refuseUninitializedConstMem*(ctx: var auto, name: string): untyped =
  ## Refusal of a const_mem local without an initializer.
  ## The target compilers reject constant storage that lacks an initializer.
  mixin fail
  ctx.fail("const_mem local " & name & " without an initializer")

template printCast*(ctx: var auto, e: CirExpr, rules: openArray[CastRule], dstName: untyped): string =
  ## One bit reinterpret `e` through the target's cast rules.
  ## The first matching rule decides, no match keeps `ctx.printConv`.
  mixin printConv
  block:
    let src = getExprType(e.castOperand)
    let i = matchCastRule(rules, src, e.castType)
    if i >= 0:
      if rules[i].conv.len > 0:
        rules[i].conv % [ctx.printExpr(e.castOperand), dstName]
      else:
        ctx.fail("no bit cast to " & dstName)
        ""
    else:
      ctx.printConv(e)

proc printLoopBodyText(ctx: var auto, body: seq[CirStmt], unroll: bool, head: string, style: LoopStyle): string =
  ## One loop: the unroll directive, the head, the indented body.
  mixin printStmts
  ctx.indent.inc
  let bodyTxt = ctx.printStmts(body)
  ctx.indent.dec
  let pad = spaces(ctx.indent * 2)
  let directive = if unroll and style.unrollDirective.len > 0:
    pad & style.unrollDirective & "\n"
  else:
    ""
  result = directive & pad & head
  if not style.braceSameLine:
    result &= pad & "{\n"
  result &= bodyTxt & "\n" & pad & '}'

proc printWhileText(ctx: var auto, s: CirStmt, style: LoopStyle): string =
  ## One `sWhile` statement in the target loop style.
  mixin printExpr
  let condTxt = ctx.printExpr(s.whileCond)
  let head = if style.braceSameLine:
    "while (" & condTxt & ") {\n"
  else:
    "while (" & condTxt & ")\n"
  printLoopBodyText(ctx, s.whileBody, s.whileUnroll, head, style)

proc printIteElse(ctx: var auto, elseBody: seq[CirStmt], pad: string): string =
  ## Braced else tail of an `sIte` statement, "" for an empty else body.
  mixin printStmts
  if elseBody.len == 0:
    return ""
  ctx.indent.inc
  let elseTxt = ctx.printStmts(elseBody)
  ctx.indent.dec
  result = " else {\n" & elseTxt & '\n' & pad & '}'

proc printEmitOperand(ctx: var auto, op: CirExpr): string =
  if ctx.policy.emitOperandRendersExpr:
    ctx.printExpr(op)
  else:
    op.name

proc printCallArgTexts*(ctx: var auto, callee: CirExpr, args: seq[CirExpr]): seq[string] =
  mixin printExpr
  for a in args:
    result.add ctx.printExpr(a)

proc emitBuiltinCall*(ctx: var auto, callee: CirExpr, args: seq[string]): string =
  ## Target text of one builtin call over the family tables.
  ##
  ## Returns:
  ## - the native text when the kind has one, `$1`..`$9` for the arguments
  ## - "" for a canonical-name pass-through: the target compiler rejects
  ##   what it cannot parse
  if callee.kind != eSym or not callee.builtin:
    return ""
  if callee.synchroBuiltin != cbkNone:
    let txt = ctx.emit.vocab.synchroText[callee.synchroBuiltin]
    if txt.len > 0:
      return txt % args
  elif callee.reductionBuiltin != cbkNone:
    let txt = ctx.emit.vocab.reductionText[callee.reductionBuiltin]
    if txt.len > 0:
      return txt % args
  elif callee.shuffleBuiltin != cbkNone:
    let txt = ctx.emit.vocab.shuffleText[callee.shuffleBuiltin]
    if txt.len > 0:
      return txt % args
  elif callee.atomicsBuiltin != cbkNone:
    let txt = ctx.emit.vocab.atomicsText[callee.atomicsBuiltin]
    if txt.len > 0:
      return txt % args
  return ""

proc convBuiltinText*(ctx: var auto, call: CirExpr, args: seq[string]): string =
  ## Builtins-driven conv text: the ConvBuiltinNames mapping keyed
  ## (backend, source cell, target cell) for the call's signature.
  ##
  ## Returns: the mapping's native name formatted with the argument list.
  ## Returns "" when the mapping is absent: the canonical name passes through.
  if call.callee.kind != eSym or not call.callee.builtin:
    return ""
  let pt = call.callee.typ
  if pt == nil or pt.kind != ctProc or pt.params.len != 1 or pt.ret == nil:
    return ""
  let txt = ConvBuiltinNames.getOrDefault((ctx.backend,
      getCellToken(pt.params[0].kind, pt.params[0].bits),
      getCellToken(pt.ret.kind, pt.ret.bits)), "")
  if txt.len == 0:
    return ""
  txt & "(" & args.join(", ") & ")"

proc mathBuiltinCallText*(names: Table[(string, int), string], call: CirExpr, args: seq[string]): string =
  ## Native text of one width-sensitive scalar-math call over
  ## `<target>MathBuiltinNames`, "" when the canonical name passes through.
  if call.callee.kind != eSym or not call.callee.builtin or call.args.len != 1:
    return ""
  let at = getExprType(call.args[0])
  if at == nil or at.kind != ctFloat:
    return ""
  let name = names.getOrDefault((call.callee.name, at.bits), "")
  if name.len == 0:
    return ""
  return name & "(" & args.join(", ") & ")"

proc getAtomicCallText*(ctx: var auto, callee: CirExpr, argExprs: seq[CirExpr], args: seq[string]): string =
  ""

proc getSymbolText*(ctx: var auto, e: CirExpr): string =
  e.name

proc getDotText*(ctx: var auto, e: CirExpr, baseTxt: string): string =
  baseTxt & '.' & e.fieldName

proc getCtorValue*(ctx: var auto, t: CirType, kvv: CirKv): string =
  ctx.printExpr(kvv.value)

proc printOconstr*(ctx: var auto, e: CirExpr): string =
  ## One object construction in the target ctor form.
  mixin emitTypeName, getCtorValue
  let name = ctx.emitTypeName(e.ocType)
  if e.kvFields.len == 0 and ctx.emit.ctorZero.len > 0:
    return ctx.emit.ctorZero % name
  let typed = if ctx.emit.ctorCast: "(" & name & ")" else: name
  result = typed & ctx.emit.ctorOpen
  for i, kvv in e.kvFields:
    if i > 0:
      result.add ", "
    result.add ctx.getCtorValue(e.ocType, kvv)
  result.add ctx.emit.ctorClose

proc printAconstr*(ctx: var auto, e: CirExpr): string =
  ## One array construction in the target ctor form.
  mixin emitTypeName
  let lead = if ctx.emit.functionalCtor: ctx.emitTypeName(e.acType) else: ""
  result = lead & ctx.emit.ctorOpen
  for i, el in e.elems:
    if i > 0:
      result.add ", "
    if ctx.emit.aconstrCasts:
      result.add "(" & ctx.emitTypeName(e.acType.aElem) & ")"
    result.add ctx.printExpr(el)
  result.add ctx.emit.ctorClose

proc printSizeof*(ctx: var auto, e: CirExpr): string =
  mixin emitTypeName
  "sizeof(" & ctx.emitTypeName(e.ofType) & ")"

proc printExtraExpr*(ctx: var auto, e: CirExpr): string =
  ## Refusal for an expression kind outside every case.
  mixin printerTarget
  ctx.fail(getUnhandledExprRejection(e.kind))
  result = ""

proc printTargetExpr*(ctx: var auto, e: CirExpr): string =
  ## Universal expression variants of the text printers: the eight kinds
  ## every target emits in one shape over the target tables.
  ##
  ## - coordinate threading resolves through `coordRef`/`coordDot`
  ## - conversion and declaration forms resolve through `printConv`
  ##   and `emitTypeName`
  ## - a kind a target routes itself reaches `printExtraExpr`
  mixin coordRef, coordDot, printConv, emitTypeName, printSizeof,
        convBuiltinText,
        printExtraExpr, getAtomicCallText, getSymbolText, getDotText, getCtorValue,
        printOconstr, printAconstr, printCallArgTexts
  case e.kind
  of eSym:
    if e.coordBuiltin != cbkNone:
      return ctx.coordRef(e)
    if e.typ != nil and e.typ.kind == ctPtr and
        e.typ.addressSpace in ctx.emit.noPointerValues.spaces:
      ctx.fail(ctx.emit.noPointerValues.text % e.name)
      return ""
    return ctx.getSymbolText(e)
  of eDot:
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      return ctx.coordDot(e)
    if ctx.emit.vectorDots:
      let vecDot = ctx.printVectorDot(e)
      if vecDot.len > 0 or ctx.failing():
        return vecDot
    let base = ctx.printExpr(e.dotBase)
    if e.dotBase.kind == eDeref and ctx.emit.dotDerefParens:
      return "(" & base & ")." & e.fieldName
    return ctx.getDotText(e, base)
  of eCast:
    return ctx.printCast(e, TargetVocabs[ctx.backend].castRules, ctx.emitTypeName(e.castType))
  of eConv:
    return ctx.printConv(e)
  of eSizeof:
    return ctx.printSizeof(e)
  of eOconstr:
    return ctx.printOconstr(e)
  of eAconstr:
    return ctx.printAconstr(e)
  of eCall:
    let args = ctx.printCallArgTexts(e.callee, e.args)
    var builtinTxt = ctx.emitBuiltinCall(e.callee, args)
    if builtinTxt.len == 0:
      builtinTxt = ctx.convBuiltinText(e, args)
      if builtinTxt.len == 0:
        builtinTxt = mathBuiltinCallText(ctx.mathNames, e, args)
    if builtinTxt.len == 0 and e.callee.atomicsBuiltin == cbkAtomicFetchAdd:
      builtinTxt = ctx.getAtomicCallText(e.callee, e.args, args)
    if builtinTxt.len > 0:
      return builtinTxt
    let calleeName =
      if e.callee.kind == eSym: e.callee.name
      else: ctx.printExpr(e.callee)
    return calleeName & "(" & args.join(", ") & ")"
  else:
    return ctx.printExtraExpr(e)

proc printExpr*(ctx: var auto, e: CirExpr): string =
  ## Prints one structured expression.
  ##
  ## - the operator branches and the literal head emit here.
  ##   The call skeleton prints through `printCallArgTexts`.
  ## - the shared defaults cover subscript, deref, addr-of and ternary
  ## - a printer's own kinds route to `printTargetExpr`
  ## - a target that lists `eNil` refuses the nil literal here
  mixin printExpr, printTargetExpr
  if ctx.failing():
    return ""
  case e.kind
  of eNil:
    if e.kind in ctx.policy.targetExprKinds:
      ctx.fail(nilLiteralRejection)
      return ""
    return emitLiteral(e, TargetVocabs[ctx.backend].lit, TargetVocabs[ctx.backend].scalarNames)
  of eTrue, eFalse, eIntLit, eFloatLit, eStrLit:
    if e.kind in ctx.policy.targetExprKinds:
      return ctx.printTargetExpr(e)
    return emitLiteral(e, TargetVocabs[ctx.backend].lit, TargetVocabs[ctx.backend].scalarNames)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    let op = case e.kind
      of eAdd: "+"
      of eSub: "-"
      of eMul: "*"
      of eDiv: "/"
      of eMod: "%"
      of eShl: "<<"
      of eShr: ">>"
      of eBitAnd: "&"
      of eBitOr: "|"
      else: "^"
    return "(" & ctx.printExpr(e.left) & ' ' & op & ' ' & ctx.printExpr(e.right) & ")"
  of eEq, eNeq, eLt, eLe:
    let op = case e.kind
      of eEq: "=="
      of eNeq: "!="
      of eLt: "<"
      else: "<="
    return "(" & ctx.printExpr(e.a) & ' ' & op & ' ' & ctx.printExpr(e.b) & ")"
  of eNot:
    return "(!" & ctx.printExpr(e.notOperand) & ")"
  of eAnd, eOr:
    let op = if e.kind == eAnd: "&&" else: "||"
    return "(" & ctx.printExpr(e.andLeft) & ' ' & op & ' ' & ctx.printExpr(e.andRight) & ")"
  of eTernary, eAt, ePat, eDeref, eAddr:
    if e.kind in ctx.policy.targetExprKinds:
      return ctx.printTargetExpr(e)
    if e.kind in {eAt, ePat}:
      let (baseTxt, idxTxt) = ctx.printSubscriptParts(e)
      return baseTxt & '[' & idxTxt & ']'
    if e.kind == eDeref:
      return "(*" & ctx.printExpr(e.derefOperand) & ")"
    if e.kind == eAddr:
      return "(&" & ctx.printExpr(e.addrOperand) & ")"
    return "(" & ctx.printExpr(e.cond) & " ? " & ctx.printExpr(e.thenE) &
      " : " & ctx.printExpr(e.elseE) & ")"
  of eNeg:
    return "(-" & ctx.printExpr(e.unOperand) & ")"
  of eSym, eDot, eCast, eConv, eSizeof, eOconstr, eAconstr, eCall:
    return ctx.printTargetExpr(e)

proc printStmt(ctx: var auto, s: CirStmt): string =
  ## Prints one structured statement.
  ##
  ## Shared variants emit here: `sIte`, `sRet`, `sDiscard`, `sEmit`.
  ## A loop prints through `printWhileText` off `printStmtList`.
  ## Per-target policy resolves at instantiation through the printer's own procs:
  ##
  ## - `printVar`: storage-class refusals and declaration text
  ## - `printAsgn`: assignment policy, pointer rebinds and buffer writes
  ## - `printCall`: barrier intrinsics and builtin call vocabulary
  ## - `printAsm`: inline assembly form, refusal text or raw splice
  mixin printStmts, printExpr, printVar, printAsgn, printCall, printAsm,
        printIteElse, printEmitOperand
  if ctx.failing():
    return ""
  let pad = spaces(ctx.indent * 2)
  case s.kind
  of sVar: result = ctx.printVar(s)
  of sAsgn: result = ctx.printAsgn(s)
  of sCall: result = ctx.printCall(s)
  of sAsm: result = ctx.printAsm(s)
  of sIte:
    let cond = ctx.printExpr(s.cond)
    ctx.indent.inc
    let thenTxt = ctx.printStmts(s.thenBody)
    ctx.indent.dec
    result = pad & "if (" & cond & ") {\n" & thenTxt & '\n' & pad & '}'
    if s.elseBody.len == 1 and s.elseBody[0].kind == sIte and ctx.policy.foldsIteElse:
      # an else side of one ite is the elif chain, printed as `else if`
      result &= " else " & ctx.printStmt(s.elseBody[0])
    else:
      result &= ctx.printIteElse(s.elseBody, pad)
  of sWhile:
    result = ""
  of sRet:
    if s.retVal != nil:
      result = pad & "return " & ctx.printExpr(s.retVal) & ";"
    else:
      result = pad & "return;"
  of sDiscard:
    result = ""
  of sEmit:
    # `{.emit.}` raw source: verbatim text spliced with the operand text,
    # no terminator appended as the text supplies its own terminators
    var text = ""
    for i, part in s.emitParts:
      text.add(part)
      if i < s.emitOperands.len:
        text.add(ctx.printEmitOperand(s.emitOperands[i]))
    result = pad & text

proc asmSpliceText*(s: CirStmt): string =
  ## Verbatim template of one assembly statement with the operand names spliced.
  var text = ""
  for i, part in s.asmParts:
    text.add(part)
    if i < s.asmOperands.len:
      text.add(s.asmOperands[i].name)
  text

proc printAsm*(ctx: var auto, s: CirStmt): string =
  ## Refused. A target with a splice form defines `printAsm` itself.
  mixin printerTarget
  ctx.fail("inline assembly has no " & printerTarget & " form")
  result = ""

proc printDeclStmt*(pad: string, decl: string, initTxt: string): string =
  result = pad & decl
  if initTxt.len > 0:
    result &= " = " & initTxt
  result &= ";"

proc getInitText*(ctx: var auto, s: CirStmt): string =
  mixin printExpr
  if s.init != nil:
    return ctx.printExpr(s.init)
  ""

proc printVar*(ctx: var auto, s: CirStmt): string =
  ## One local declaration through the target's space policy.
  mixin checkLocalVar, emitLocalDecl, getInitText
  if not ctx.checkLocalVar(s):
    return ""
  let pad = spaces(ctx.indent * 2)
  let decl = ctx.emitLocalDecl(s)
  if decl.len == 0 or ctx.failing():
    return ""
  let init = if s.init != nil: ctx.getInitText(s) else: ""
  printDeclStmt(pad, decl, init)

proc printAsgn*(ctx: var auto, s: CirStmt): string =
  ## One assignment in the direct form, refusing pointer rebinds per the target knobs.
  let pad = spaces(ctx.indent * 2)
  if ctx.emit.noPointerRebind.len > 0 and s.lhs.kind == eSym and
      s.lhs.typ != nil and s.lhs.typ.kind == ctPtr:
    ctx.fail(ctx.emit.noPointerRebind % s.lhs.name)
    return ""
  result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"

proc getStmtBuiltinText*(ctx: var auto, s: CirStmt, pad: string, args: seq[string]): string =
  ## Statement-position text of one builtin call, "" for the direct form.
  let txt = ctx.emit.vocab.stmtText[s.callCallee.synchroBuiltin]
  if txt.len > 0:
    return pad & (txt % args) & ";"
  return ""

proc printCall*(ctx: var auto, s: CirStmt): string =
  ## One call statement in the target statement-position form.
  mixin printExpr, getStmtBuiltinText, printCallArgTexts, getAtomicCallText
  let pad = spaces(ctx.indent * 2)
  let calleeName =
    if s.callCallee.kind == eSym: s.callCallee.name
    else: ctx.printExpr(s.callCallee)
  let args = ctx.printCallArgTexts(s.callCallee, s.callArgs)
  if s.callCallee.kind == eSym and s.callCallee.synchroBuiltin != cbkNone:
    let stmtTxt = ctx.getStmtBuiltinText(s, pad, args)
    if stmtTxt.len > 0:
      return stmtTxt
  # a builtin statement call emits through the value vocabulary too.
  # Structural atomics carry no text form
  if s.callCallee.kind == eSym and s.callCallee.builtin:
    if s.callCallee.atomicsBuiltin == cbkAtomicFetchAdd:
      let atomicTxt = ctx.getAtomicCallText(s.callCallee, s.callArgs, args)
      if atomicTxt.len > 0:
        return pad & atomicTxt & ";"
    let txt = ctx.emitBuiltinCall(s.callCallee, args)
    if txt.len > 0:
      return pad & txt & ";"
  result = pad & calleeName & "(" & args.join(", ") & ");"

proc printStmtList(ctx: var auto, stmts: seq[CirStmt], style: LoopStyle): string =
  var lines: seq[string]
  for s in stmts:
    let txt = case s.kind
      of sWhile: printWhileText(ctx, s, style)
      else: ctx.printStmt(s)
    if txt.len > 0:
      lines.add txt
  return lines.join("\n")

proc printStmts*(ctx: var auto, stmts: seq[CirStmt]): string =
  printStmtList(ctx, stmts, ctx.policy.loop)

proc getModuleKernels*(m: CirModule): seq[CirProc] =
  for fn in m.procs:
    if fn.isKernel:
      result.add fn

proc getEntryKernel*(ctx: var auto, m: CirModule): CirProc =
  ## One module kernel, nil with a refusal on none or several.
  let kernels = getModuleKernels(m)
  if kernels.len != 1:
    ctx.fail("exactly one kernel per module, found " & $kernels.len)
    return
  result = kernels[0]

proc getModuleDeviceFns*(m: CirModule): seq[CirProc] =
  for fn in m.procs:
    if not fn.isKernel:
      result.add fn

# ═══════════════════════════════════════════════════════════════════════
#  Entry-input machinery
# ═══════════════════════════════════════════════════════════════════════

func getWorkgroup*(fn: CirProc, dflt: array[3, int] = [256, 1, 1]): array[3, int] =
  if fn.workgroup == [0, 0, 0]:
    dflt
  else:
    fn.workgroup

type EntryInputs* = array[cbkThreadPositionInGrid..cbkThreadIndexInThreadgroup, string]
  ## A WGSL `@builtin(...)` parameter or an HLSL semantic-carrying input:
  ## one text per coordinate builtin kind.
  ##
  ## An empty text has no entry input. The read happens another way:
  ##
  ## - the body's workgroup literal
  ## - a lowering-pass materialized parameter
  ## - no read at all

template emitEntryInputs*(kinds: untyped, inputs: EntryInputs): string =
  ## Returns the entry-input texts of one shader entry point.
  ## One text per coordinate builtin the entry reads.
  ## The order is the collection's: first use on WGSL, then the canonical HLSL order.
  var texts: seq[string]
  for k in kinds:
    if inputs[k].len > 0:
      texts.add inputs[k]
  texts.join(", ")

# ═══════════════════════════════════════════════════════════════════════
#  Proc assembly
# ═══════════════════════════════════════════════════════════════════════

type ModuleShape* = object
  ## Static assembly plan of one target's module: the grammar knobs the shared
  ## skeleton prints with, plus the optional sections the target supplies.
  ## Targets pass a `const` and the skeleton's `when` guards specialize on it.
  spaces*: set[AddressSpace]
  ## Program-scope address spaces the target declares.
  structTypedef*: bool
  ## Struct declarations in the C `typedef struct { ... } Name` form.
  structPadding*: string
  ## Member text of the zero-field struct declaration:
  ##
  ## - no indentation, no member terminator
  ## - every target compiler requires at least one struct member
  structFieldTerm*: char
  ## Member terminator: a semicolon on the C family, a comma on WGSL.
  structClose*: string
  ## Closing token of a non-typedef struct declaration: brace plus semicolon
  ## on the C family, a semicolon-free brace on WGSL.
  fwdDecls*: bool
  ## Device functions forward-declare before their definitions.
  moduleOrderProcs*: bool
  ## Every proc prints in `m.procs` order, declaration before use:
  ##
  ## - CUDA C++ and MSL resolve calls at their declaration site, a callee
  ##   prints before its caller
  ## - the device-functions-then-kernels split is off
  kernelSep*: string
  ## Text after each kernel body in the fns-then-kernels assembly.
  customProcHead*: bool
  ## Targets with their own head grammar assemble the proc head: shader entry heads and the GLSL `local_size` qualifier.
  printsRefBlocks*: bool
  ## Declares the module's buffer-reference blocks when set.
  printsParamBindings*: bool
  ## Declares the kernel parameter bindings: resource slots, push-constant members.
  printsReduceTemplates*: bool
  ## Emits the warp-reduction templates that the module's feature scan lists.

proc prepProc*(ctx: var auto, fn: CirProc) =
  ctx.builtins.clear()
  ctx.inKernel = fn.isKernel

proc emitComputePrefix*(ctx: var auto, fn: CirProc): string =
  if fn.isKernel:
    return ctx.emit.prefixKernel
  if fn.inline:
    return ctx.emit.prefixInline
  return ctx.emit.prefixDevice

proc printSig*(ctx: var auto, fn: CirProc): string =
  ## C signature line of one proc. The kernel name comes from exportc.
  ## Returns "" once the walk recorded a refusal.
  mixin emitComputeParams, emitComputePrefix, emitTypeName
  ctx.fnName = fn.name
  let name = if fn.isKernel and fn.exportName.len > 0: fn.exportName else: fn.name
  let params = ctx.emitComputeParams(fn)
  if ctx.failing():
    return ""
  result = ctx.emitComputePrefix(fn) & ctx.emitTypeName(fn.ret) & ' ' & name &
    '(' & params.join(", ") & ")"

proc printStructDecl(ctx: var auto, byName: Table[string, CirType],
    emitted: var HashSet[string], name: string, outp: var string,
    shape: static ModuleShape) =
  ## One struct declaration into `outp`, member structs first, deduplicated by name.
  mixin emitStructField
  if name in emitted:
    return
  let t = byName.getOrDefault(name, nil)
  if t == nil:
    return
  emitted.incl name
  for f in t.fields:
    if f.typ.kind == ctStruct:
      ctx.printStructDecl(byName, emitted, f.typ.name, outp, shape)
    elif f.typ.kind == ctArray and f.typ.aElem.kind == ctStruct:
      ctx.printStructDecl(byName, emitted, f.typ.aElem.name, outp, shape)
  if ctx.failing():
    return
  if shape.structTypedef:
    outp.add "typedef struct {\n"
  else:
    outp.add "struct " & name & " {\n"
  var members = 0
  for f in t.fields:
    let fieldTxt = ctx.emitStructField(t, f)
    if fieldTxt.len == 0:
      continue
    outp.add "  " & fieldTxt & shape.structFieldTerm & "\n"
    members.inc
  if members == 0:
    outp.add "  " & shape.structPadding & shape.structFieldTerm & "\n"
  if shape.structTypedef:
    outp.add "} " & name & ";\n\n"
  else:
    outp.add shape.structClose & "\n\n"

proc printStructs*(ctx: var auto, m: CirModule, shape: static ModuleShape): string =
  ## Struct declarations of the module, "" once the walk refused.
  var byName: Table[string, CirType]
  for t in m.types:
    if t.kind == ctStruct:
      byName[t.name] = t
  var emitted: HashSet[string]
  for t in m.types:
    if t.kind != ctStruct:
      continue
    ctx.printStructDecl(byName, emitted, t.name, result, shape)

proc printProc*(ctx: var auto, fn: CirProc, shape: static ModuleShape): string =
  ## One device function or kernel, "" once the walk refused.
  ## - the body prints before the head assembles
  ## - a shader entry head lists the coordinate-builtin inputs
  ##   that the body walk collected
  mixin prepProc, printProcHead, printStmts
  ctx.fnName = fn.name
  ctx.indent = 0
  ctx.prepProc(fn)
  let bodyTxt = ctx.printStmts(fn.body)
  if ctx.failing():
    return ""
  let head = when shape.customProcHead: ctx.printProcHead(fn)
             else: printSig(ctx, fn) & (if fn.isKernel: " {\n" else: "")
  if ctx.failing():
    return ""
  if fn.isKernel:
    return head & bodyTxt & '\n' & '}'
  return head & " {\n" & bodyTxt & '\n' & '}'

proc declareExtensions*(features: ModuleFeatures, lines: openArray[tuple[on: bool, text: string]]): string =
  ## One pragma line per set flag in the target's feature table, then a blank line.
  for entry in lines:
    if entry.on:
      result.add entry.text
  result.add "\n"

proc printProgramScopeDecls*(ctx: var auto, m: CirModule, spaces: set[AddressSpace]): string =
  ## Program-scope declarations of the relocated storage locals in `m.globals`,
  ## one per `sVar` in `spaces`, then a blank line after a non-empty section.
  mixin emitComputeGlobal
  for s in m.globals:
    if s.kind == sVar and s.addressSpace in spaces:
      let decl = ctx.emitComputeGlobal(s)
      if decl.len > 0:
        result &= decl & '\n'
  if result.len > 0:
    result &= '\n'

proc printModule*(ctx: var auto, m: CirModule, kernels: seq[CirProc], shape: static ModuleShape): tuple[src: string, err: string] =
  ## Full module assembly over the shared section skeleton:
  ## - prelude, ref blocks, structs, parameter bindings
  ## - program-scope declarations, reduction templates, the proc set
  ## - a refusal anywhere records once and returns
  mixin printPrelude, printRefBlocks, printParamBindings, printReduceTemplates
  result.src = ctx.printPrelude(m)
  if ctx.failing():
    return ("", ctx.err)
  when shape.printsRefBlocks:
    result.src &= ctx.printRefBlocks(m)
    if ctx.failing():
      return ("", ctx.err)
  result.src &= printStructs(ctx, m, shape)
  if ctx.failing():
    return ("", ctx.err)
  when shape.printsParamBindings:
    result.src &= ctx.printParamBindings(kernels)
    if ctx.failing():
      return ("", ctx.err)
  result.src &= printProgramScopeDecls(ctx, m, shape.spaces)
  if ctx.failing():
    return ("", ctx.err)
  when shape.printsReduceTemplates:
    result.src &= ctx.printReduceTemplates(m)
    if ctx.failing():
      return ("", ctx.err)
  if shape.moduleOrderProcs:
    for fn in m.procs:
      result.src &= ctx.printProc(fn, shape) & "\n\n"
      if ctx.failing():
        return ("", ctx.err)
  else:
    let fns = getModuleDeviceFns(m)
    if shape.fwdDecls:
      for fn in fns:
        when shape.customProcHead:
          result.src &= ctx.printProcHead(fn) & ";\n"
        else:
          result.src &= printSig(ctx, fn) & ";\n"
        if ctx.failing():
          return ("", ctx.err)
      if fns.len > 0:
        result.src &= "\n"
    for fn in fns:
      result.src &= ctx.printProc(fn, shape) & "\n\n"
    if ctx.failing():
      return ("", ctx.err)
    for kernel in kernels:
      result.src &= ctx.printProc(kernel, shape) & shape.kernelSep
      if ctx.failing():
        return ("", ctx.err)

proc codegenModule*[R](printed: (R, string)): R =
  ## Converts a printer result tuple into the `backends_api` entry-point return value.
  ##
  ## Returns the artifact. A non-empty refusal is a compile error naming the construct.
  if printed[1].len > 0:
    error(printed[1])
  result = printed[0]
