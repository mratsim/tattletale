# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared printer tier over CIRv2: the statement/expression walk, the C declarator engine, and the module assembly.
##
## C declarator engine, driven by the registry tables that each printer supplies.
##
## - `getScalarName`/`getLeafName`/`getBareName`: type names from `ScalarNames` and `VectorRule`
## - `cDecl`: C-style declaration text from type structure
## - a leaf absent from the tables returns "", so the caller composes the name or refuses
##
## Every printer under `backends/` imports these, no printer owns a copy.
## A leaf belongs here when two or more printers need the same text.
## Single-target emission stays in the printer.

import std/[macros, math, sets, strutils, tables]
import ../ir/cirv2_lang
import ./targets_registry
export targets_registry

template failing*(ctx: auto): bool =
  ## True when the walk in `ctx` already recorded a refusal.
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

## Refusal text for a nil literal in a target whose kind row refuses nil.
const nilLiteralRejection* = "nil literal is outside the target subset"

proc unhandledExprRejection*(kind: CirExprKind): string =
  ## Refusal text for an expression kind outside the printer's case.
  ## A routing bug: the dispatcher sends only the kinds the case lists.
  "expression kind " & $kind & " has no printer emission"

proc roundUpMultiple*(v, multiple: int): int =
  ## Smallest multiple of `multiple` that is >= `v`.
  doAssert multiple > 0
  (v + multiple - 1) div multiple * multiple

proc scalarByteSize*(t: CirType, boolWidth: static int = 1): int =
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

proc getVectorComponent*(ctx: var auto, field: string): string =
  ## Component letter for one vector field name.
  ##
  ## Returns "" when the name maps to no component letter. The walk records
  ## the refusal and the caller discards its partial text.
  ##
  ##   let comp = ctx.getVectorComponent(e.fieldName)
  result = emitVectorComponent(field)
  if result.len == 0:
    ctx.fail("no vector component " & field)

# ═══════════════════════════════════════════════════════════════════════
# Declarator engine: type names and declaration forms over the registry tables
# ═══════════════════════════════════════════════════════════════════════

proc getScalarName*(names: ScalarNames, t: CirType): string =
  ## Scalar type name of a leaf type from the target's scalar-name table.
  ##
  ## Returns "" when the kind is not a scalar leaf or the table omits the (kind, width) cell.
  ## The caller composes the name or refuses, per its own policy.
  ##
  ##   result = getScalarName(mslScalarNames, t)  # "half" for a 16-bit float
  case t.kind
  of ctVoid, ctBool, ctString:
    for c in names.cells:
      if c.kind == t.kind:
        return c.spelling
  of ctInt, ctUint, ctFloat, ctBf16:
    for c in names.cells:
      if c.kind == t.kind and c.width == t.bits:
        return c.spelling
  else:
    discard
  return ""

proc scalarRefusal*(target: string, names: ScalarNames, t: CirType): string =
  ## Returns the one-line refusal message for a scalar leaf the target's scalar
  ## names do not cover: `<target>: no <cell>` with the declared reason appended.
  ##
  ##   result = scalarRefusal("msl", mslScalarNames, tFloat(64))
  ##   # "<msl: no f64, double has no Apple GPU support>"
  var tok = ""
  for a in ScalarCellAxis:
    if a.kind == t.kind and (t.kind in {ctVoid, ctBool} or a.width == t.bits):
      tok = a.tok
  for r in names.refusals:
    if r.kind == t.kind and (t.kind in {ctVoid, ctBool} or r.width == t.bits):
      return "<" & target & ": no " & tok & ", " & r.reason & ">"
  return "<" & target & ": no " & tok & ">"

proc getVectorName*(rule: VectorRule, names: ScalarNames, t: CirType): string =
  ## Vector type name of `t` from the target's vector rule.
  ##
  ## Returns "" when `t` is not a vector, its element has no cell in the rule,
  ## or its lane count is outside `rule.lanes`. The caller refuses.
  ##
  ##   result = getVectorName(glslVectorRule(), glslScalarNames, t)  # "vec4"
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

proc getLeafName*(names: ScalarNames, rule: VectorRule, t: CirType): string =
  ## Bare type name of a non-array leaf type: the scalar, vector or struct name.
  ##
  ## Returns "" for pointers, arrays, proc and span types. The caller composes
  ## the name or refuses, per its own policy.
  case t.kind
  of ctVoid, ctBool, ctInt, ctUint, ctFloat, ctBf16, ctString:
    result = getScalarName(names, t)
  of ctVector:
    result = getVectorName(rule, names, t)
  of ctStruct:
    result = t.name
  of ctPtr, ctAptr, ctArray, ctProc, ctSpan:
    result = ""

proc getBareName*(names: ScalarNames, rule: VectorRule, t: CirType): string =
  ## Bare type name of `t`: the form casts and compound literals use.
  ## No identifier, pointer stars trailing the pointee name.
  ##
  ## Returns "" for arrays, proc and span types.
  ## The caller composes the name or refuses, per its own policy.
  ##
  ##   result = getBareName(cudaScalarNames, cudaVectorRule, t)  # "float**"
  if t.kind in {ctPtr, ctAptr}:
    return getBareName(names, rule, t.pElem) & "*"
  return getLeafName(names, rule, t)

proc getUnnamedScalarCell*(names: ScalarNames, t: CirType): CirType =
  ## Returns the scalar leaf of `t` when the table has no name for it, after peeling pointer stars and array dimensions.
  ##
  ## Returns nil when every scalar leaf has a type name or when the leaf is not a scalar kind.
  ##
  ##   let missing = getUnnamedScalarCell(mslScalarNames, tFloat(64))
  ##   # tFloat(64) while msl names no f64, nil for tFloat(16) ("half")
  var leaf = t
  while leaf.kind in {ctPtr, ctAptr, ctArray}:
    leaf = if leaf.kind == ctArray: leaf.aElem else: leaf.pElem
  if leaf.kind in {ctVoid, ctBool, ctInt, ctUint, ctFloat, ctBf16, ctString} and
      getScalarName(names, leaf).len == 0:
    return leaf
  return nil

proc cDecl*(names: ScalarNames, rule: VectorRule, target: string, t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## Declaration text of `t` with `ident` appended in the C declarator form:
  ## the array dimensions move behind the identifier and nest into the declarator
  ## (`T ident[N]`, pointer-to-array `T (*ident)[N]`).
  ##
  ## - `allowEmptyIdent` permits no identifier, return types and casts use it
  ## - the printer refuses an array without an identifier with the `<target: array needs an identifier>` text
  ## - the printer refuses a scalar leaf absent from the scalar names with the one-line
  ##   `scalarRefusal` message on the same `<target: ...>` channel
  ## - the leaf check runs before the printer composes the name, so the refusal
  ##   also fires when pointer stars or array dimensions wrap the absent cell
  ##
  ##   result = cDecl(cudaScalarNames, cudaVectorRule, "cuda", t, "buf", false)
  ##   # `ptr array[4, float32]` with the ident "v": "float (*v)[4]",
  ##   # the empty-ident cast form: "float (*)[4]"
  if t.kind == ctArray:
    if ident.len == 0 and not allowEmptyIdent:
      return "<" & target & ": array needs an identifier>"
  let missing = getUnnamedScalarCell(names, t)
  if missing != nil:
    return scalarRefusal(target, names, missing)
  if t.kind in {ctPtr, ctAptr} and t.pElem.kind == ctArray:
    # array dimensions bind behind the identifier, one pointer star inside the parenthesized declarator
    return cDecl(names, rule, target, t.pElem, "(" & '*' & ident & ")",
      allowEmptyIdent)
  if t.kind == ctArray:
    if t.aElem.kind == ctArray:
      # a pointer declarator is already parenthesized (`(*ident)`), one more parenthesized layer would print `((*ident))`
      let declarator = if ident.startsWith('('): ident else: "(" & ident & ")"
      return cDecl(names, rule, target, t.aElem, declarator & "[" & $t.aN & "]",
        allowEmptyIdent)
    return getBareName(names, rule, t.aElem) & ' ' & ident & '[' & $t.aN & ']'
  result = getBareName(names, rule, t)
  if ident.len > 0:
    result.add ' ' & ident

proc nonFiniteFloatLit*(e: CirExpr, tok: FloatLitTokens): string =
  ## Emission token of a non-finite float literal.
  ##
  ## Returns "" when the value is finite.
  ##
  ## Finite values carry the caller's own width suffix:
  ##
  ##   of eFloatLit:
  ##     result = nonFiniteFloatLit(e, targetTokens)
  ##     if result.len == 0:
  ##       result = $e.fValue & "f"
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
  ## Returns the literal token of `e`, chosen from the target's literal tokens
  ## `tokens` according to the literal type of `e`.
  ##
  ## - a narrow (8/16-bit) integer prints as a cast to its scalar name
  ##   when the tokens set `narrowCast`, otherwise bare
  ## - a non-finite float emits the `tokens.nonFinite` bit-pattern token
  ## - a literal kind outside the tables prints `<printerTarget: not a literal>`
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
      lit = nonFiniteFloatLit(e, tokens.nonFinite)
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

template printCast*(ctx: var auto, e: CirExpr, rules: openArray[CastRule], dstName: untyped): string =
  ## Prints one bit reinterpret `e` through the target's cast rules `rules`.
  ##
  ## - the first matching rule decides the emission: a `spelling` rule formats
  ##   `$1` as the printed operand and `$2` as `dstName`, a refusal rule
  ##   records its message
  ## - no matching rule keeps the target's numeric-conversion form, `ctx.printConv`
  ## - `dstName` evaluates only when a rule formats it
  ##
  ##   of eCast:
  ##     return ctx.printCast(e, cudaCastRules, emitCudaType(e.castType, "", true))
  mixin printConv
  block:
    let src = getExprType(e.castOperand)
    let i = matchCastRule(rules, src, e.castType)
    if i >= 0:
      if rules[i].spelling.len > 0:
        rules[i].spelling % [ctx.printExpr(e.castOperand), dstName]
      else:
        ctx.fail(rules[i].refusal % ["", dstName])
        ""
    else:
      ctx.printConv(e)

proc printLoopBodyText(ctx: var auto, body: seq[CirStmt], unroll: bool, head: string, style: LoopStyle): string =
  ## Returns the text of one loop: the unroll directive, then the loop head
  ## given as `head`, with the body indented under it.
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
  ## Returns the text of one `sWhile` statement: the unroll directive, then the `while (cond)` head.
  ## `cond` prints through `ctx.printExpr`.
  mixin printExpr
  let condTxt = ctx.printExpr(s.whileCond)
  let head = if style.braceSameLine:
    "while (" & condTxt & ") {\n"
  else:
    "while (" & condTxt & ")\n"
  printLoopBodyText(ctx, s.whileBody, s.whileUnroll, head, style)

proc printIteElse(ctx: var auto, elseBody: seq[CirStmt], pad: string): string =
  ## Else tail of an `sIte` statement, appended after its then branch.
  ##
  ## Returns "" when the else body is empty, otherwise the braced form:
  ##
  ##   result = pad & "if (" & cond & ") { ... }" & ctx.printIteElse(s.elseBody, pad)
  mixin printStmts
  if elseBody.len == 0:
    return ""
  ctx.indent.inc
  let elseTxt = ctx.printStmts(elseBody)
  ctx.indent.dec
  result = " else {\n" & elseTxt & '\n' & pad & '}'

proc printEmitOperand(ctx: var auto, op: CirExpr): string =
  ## Returns the splice text of one `{.emit.}` operand: the rendered expression
  ## when the target's walk policy sets `emitOperandRendersExpr`.
  ## The symbol name is the default splice.
  if ctx.policy.emitOperandRendersExpr:
    ctx.printExpr(op)
  else:
    op.name

proc printCallArgs*(ctx: var auto, e: CirExpr): seq[string] =
  ## Printed texts of one `eCall`'s arguments in order, in the plain expression form.
  ## A printer with per-argument policy prints its own argument texts.
  mixin printExpr
  for a in e.args:
    result.add ctx.printExpr(a)

proc printCallText*(ctx: var auto, e: CirExpr, args: seq[string], builtinTxt: string): string =
  ## Returns the text of one `eCall`: the builtin vocabulary `builtinTxt`
  ## when non-empty, else the plain call form over the printed arguments.
  mixin printExpr
  if builtinTxt.len > 0:
    return builtinTxt
  let calleeName =
    if e.callee.kind == eSym: e.callee.name
    else: ctx.printExpr(e.callee)
  calleeName & "(" & args.join(", ") & ")"

proc mathBuiltinCallText*(names: Table[(string, int), string], call: CirExpr, args: seq[string]): string =
  ## Native text of a `{.builtin.}` scalar-math call whose target name differs
  ## from the canonical name (`<target>MathBuiltinNames`), "" otherwise.
  ##
  ## - only a single float operand selects the mapped name, an integer operand
  ##   or a width absent from the table keeps the canonical name
  ## - a user proc that shares a builtin's name stays an ordinary call
  if call.callee.kind != eSym or not call.callee.builtin or call.args.len != 1:
    return ""
  let at = getExprType(call.args[0])
  if at == nil or at.kind != ctFloat:
    return ""
  let name = names.getOrDefault((call.callee.name, at.bits), "")
  if name.len == 0:
    return ""
  return name & "(" & args.join(", ") & ")"

proc printExpr*(ctx: var auto, e: CirExpr): string =
  ## Prints one structured expression.
  ##
  ## - shared variants emit here: the operator branches and the literal head
  ##   through `emitLiteral` over the context's lit tokens and scalar names
  ## - the call skeleton prints through `printCallArgs` and `printCallText`
  ## - the kinds a printer emits itself route to its `printTargetExpr`,
  ##   named by its `targetExprKinds` set
  ## - a target that lists `eNil` refuses the nil literal here, one home for the refusing targets
  ## - the shared default forms below cover subscript, dereference, address-of and ternary kinds without a target route
  mixin printExpr, printTargetExpr
  if ctx.failing():
    return ""
  case e.kind
  of eNil:
    if e.kind in ctx.policy.targetExprKinds:
      ctx.fail(nilLiteralRejection)
      return ""
    return emitLiteral(e, ctx.lit, ctx.scalarNames)
  of eTrue, eFalse, eIntLit, eFloatLit, eStrLit:
    if e.kind in ctx.policy.targetExprKinds:
      return ctx.printTargetExpr(e)
    return emitLiteral(e, ctx.lit, ctx.scalarNames)
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
      let base = if e.kind == eAt: e.atBase else: e.patBase
      let idx = if e.kind == eAt: e.atIndex else: e.patIndex
      var baseTxt = ctx.printExpr(base)
      if base.kind in {eCast, eConv}:
        # a cast binds looser than the subscript, so parenthesize the base
        baseTxt = "(" & baseTxt & ")"
      return baseTxt & '[' & ctx.printExpr(idx) & ']'
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
  ## - `printVar`, `printConst`: storage-class refusals and declaration text
  ## - `printAsgn`: assignment policy, pointer rebinds and buffer writes
  ## - `printCall`: barrier intrinsics and builtin call vocabulary
  ## - `printAsm`: inline assembly form, refusal text or raw splice
  mixin printStmts, printExpr, printVar, printConst, printAsgn, printCall, printAsm,
        printIteElse, printEmitOperand
  if ctx.failing():
    return ""
  let pad = spaces(ctx.indent * 2)
  case s.kind
  of sVar: result = ctx.printVar(s)
  of sConst: result = ctx.printConst(s)
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
    # loops print through printStmtList
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

proc printAsgn*(ctx: var auto, s: CirStmt): string =
  ## One assignment in the plain form `lhs = rhs;`.
  ##
  ## - a printer with assignment policy over pointer rebinds, buffer writes
  ##   or constructor assignments defines `printAsgn` itself
  let pad = spaces(ctx.indent * 2)
  result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"

proc printCall*(ctx: var auto, s: CirStmt): string =
  ## One call statement: the target's builtin call vocabulary when the callee
  ## is a builtin, else the plain call form. The builtin vocabulary resolves
  ## through the printer's `emitBuiltinCall(callee, args)` proc.
  mixin printExpr, emitBuiltinCall
  let pad = spaces(ctx.indent * 2)
  let calleeName =
    if s.callCallee.kind == eSym: s.callCallee.name
    else: ctx.printExpr(s.callCallee)
  var args: seq[string]
  for a in s.callArgs:
    args.add ctx.printExpr(a)
  let builtinTxt = emitBuiltinCall(s.callCallee, args)
  if builtinTxt.len > 0:
    result = pad & builtinTxt & ";"
  else:
    result = pad & calleeName & "(" & args.join(", ") & ");"

proc printStmtList(ctx: var auto, stmts: seq[CirStmt], style: LoopStyle): string =
  ## Only `sWhile` consumes `style`, every other statement prints through `ctx.printStmt`.
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

proc moduleKernels*(m: CirModule): seq[CirProc] =
  ## Returns the kernel procs of `m.procs`, in encounter order.
  ##
  ## The compute class emits every kernel of the module.
  ## The shader class compiles exactly one through `getEntryKernel`.
  for fn in m.procs:
    if fn.isKernel:
      result.add fn

proc getEntryKernel*(ctx: var auto, m: CirModule): CirProc =
  ## Returns the module's one kernel: a shader engine compiles one entry point per module invocation.
  ## Returns nil with a refusal recorded when the module has none or several kernels.
  ##
  ##   if ctx.failing():
  ##     return ("", ctx.err)
  let kernels = moduleKernels(m)
  if kernels.len != 1:
    ctx.fail("exactly one kernel per module, found " & $kernels.len)
    return
  result = kernels[0]

proc moduleDeviceFns*(m: CirModule): seq[CirProc] =
  for fn in m.procs:
    if not fn.isKernel:
      result.add fn

# ═══════════════════════════════════════════════════════════════════════
# Entry-input machinery of the shader entry heads
# ═══════════════════════════════════════════════════════════════════════

func getWorkgroup*(fn: CirProc, dflt: array[3, int] = [256, 1, 1]): array[3, int] =
  ## Baked launch shape of a kernel: the `{.workgroup.}` pragma's shape, `dflt` when the pragma is absent (the absent shape reads all zero).
  if fn.workgroup == [0, 0, 0]:
    dflt
  else:
    fn.workgroup

type EntryInputs* = Table[string, string]
  ## a WGSL `@builtin(...)` parameter, an HLSL semantic-carrying input.
  ##
  ## A name absent from the table carries no entry input, the target reads it another way:
  ## the body's workgroup literal, a lowering-pass materialized parameter, no read at all.

template emitEntryInputs*(kinds: untyped, inputs: EntryInputs): string =
  ## Returns the entry-input texts of one shader entry point: one text per
  ## coordinate builtin that the entry reads.
  ##
  ## `kinds` is the collection that the entry's body walk collected (`ctx.builtins` on WGSL, the canonically ordered reads on HLSL).
  ## The entry declares exactly the inputs its body reads, in the collection's own order: first use on WGSL, canonical order on HLSL.
  ##
  ##   "fn k(" & emitEntryInputs(ctx.builtins, wgslEntryInputs) & ") {"
  var texts: seq[string]
  for name in kinds:
    if inputs.hasKey(name):
      texts.add inputs[name]
  texts.join(", ")

# ═══════════════════════════════════════════════════════════════════════
# Proc assembly: the one proc printer over both entry models
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
  ## on the C family, a bare brace on WGSL.
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

proc printSig*(ctx: var auto, fn: CirProc): string =
  ## C signature line of one proc: linkage prefix, return type, name, parameters.
  ##
  ## A kernel's name is its exportc override when one is set, else the proc name:
  ## the host runtime looks a kernel up by its emitted name.
  ## Returns "" once the walk recorded a refusal.
  ##
  ##   printSig(ctx, fn)  # "static inline float f(float x)" on OpenCL C
  mixin emitComputeParams, emitComputePrefix, emitTypeBare
  ctx.fnName = fn.name
  let name = if fn.isKernel and fn.exportName.len > 0: fn.exportName else: fn.name
  let params = ctx.emitComputeParams(fn)
  if ctx.failing():
    return ""
  result = ctx.emitComputePrefix(fn) & emitTypeBare(fn.ret) & ' ' & name &
    '(' & params.join(", ") & ")"

proc printStructDecl(ctx: var auto, byName: Table[string, CirType],
    emitted: var HashSet[string], name: string, outp: var string,
    shape: static ModuleShape) =
  ## Emits one struct declaration into `outp`, member structs first,
  ## deduplicated by name. A name absent from `byName` emits nothing.
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
  if t.fields.len == 0:
    outp.add "  " & shape.structPadding & shape.structFieldTerm & "\n"
  for f in t.fields:
    outp.add "  " & ctx.emitStructField(t, f) & shape.structFieldTerm & "\n"
  if shape.structTypedef:
    outp.add "} " & name & ";\n\n"
  else:
    outp.add shape.structClose & "\n\n"

proc printStructs*(ctx: var auto, m: CirModule, shape: static ModuleShape): string =
  ## Returns the module's struct declaration text, "" once the walk recorded a refusal.
  ##
  ## - one declaration per struct of `CirModule.types`, member structs first,
  ##   deduplicated by name
  ## - declarations separated by a blank line, a non-struct type emits nothing
  ## - the member text is the target's `emitStructField` grammar and the shell is the target's `ModuleShape` grammar
  ##
  ##   printStructs(ctx, m, mslModuleShape)  # "struct Pixel { ... }"
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
  ## Returns the text of one device function or kernel, "" once the walk
  ## recorded a refusal.
  ##
  ## Emission sequence: the target's per-proc state (`prepProc`), then the body, then the head.
  ##
  ## Contract:
  ##
  ## - the body prints before the head assembles: a shader entry head lists
  ##   the coordinate-builtin inputs the body walk collected
  ## - a kernel head carries its own open brace in the target's placement
  ## - a device function head is the bare signature, this proc opens the body
  ##   on that line
  ##
  ##   ctx.printProc(fn, cudaModuleShape)  # 'extern "C" __global__ void k(float* o) { ... }'
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
  ## Extension pragma block of one module over the target's feature table.
  ##
  ## Expected input:
  ## - `features`: the stored module feature scan (`CirModule.features`)
  ## - `lines`: the target's table, one (feature flag, pragma line) pair per
  ##   line the block can emit, in emission order
  ##
  ## Output: one pragma line per set flag, table order, then a blank line.
  ##
  ##   declareExtensions(features, openclFeatureTable)  # pragma lines or ""
  ##
  for entry in lines:
    if entry.on:
      result.add entry.text
  result.add "\n"

proc printProgramScopeDecls*(ctx: var auto, m: CirModule, spaces: set[AddressSpace]): string =
  ## Program-scope declarations of the relocated storage locals in `CirModule.globals`.
  ##
  ## Expected input:
  ##
  ## - `m.globals`: the program-scope declarations that the target's lowering
  ##   pass relocated (const_mem, and smem on GLSL)
  ## - `spaces`: the address spaces this target declares at program scope
  ## - `emitComputeGlobal(s)` (mixin): one declaration's text, "" skips the entry
  ##
  ## Output: one declaration per `sVar` in `spaces`, encounter order, then a blank line after a non-empty section.
  ##
  ##   src &= printProgramScopeDecls(ctx, m, {caConstant})
  mixin emitComputeGlobal
  for s in m.globals:
    if s.kind == sVar and s.space in spaces:
      let decl = ctx.emitComputeGlobal(s)
      if decl.len > 0:
        result &= decl & '\n'
  if result.len > 0:
    result &= '\n'

proc printModule*(ctx: var auto, m: CirModule, kernels: seq[CirProc], shape: static ModuleShape): tuple[src: string, err: string] =
  ## Returns the full module assembly over the shared section skeleton, returning "" as the error when clean, else the first refusal.
  ##
  ## Sections in emission order, each `when`-guarded section specialized by the target's static `shape`:
  ##
  ## | section                  | content                                                                                                             |
  ## | ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
  ## | `printPrelude`           | the target's leading lines                                                                                          |
  ## | `printRefBlocks`         | the buffer-reference block declarations                                                                             |
  ## | `printStructs`           | the module's struct declarations                                                                                    |
  ## | `printParamBindings`     | the kernel parameters' binding section                                                                              |
  ## | `printProgramScopeDecls` | the hoisted program-scope declarations                                                                              |
  ## | `printReduceTemplates`   | the warp-reduction templates in use                                                                                 |
  ## | the proc set             | every proc in `m.procs` order under `moduleOrderProcs`, else the device functions under `fwdDecls` then the kernels |
  ##
  ## Every section checks the walk's first refusal, so a refusal anywhere
  ## records once and returns.
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
    let fns = moduleDeviceFns(m)
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
