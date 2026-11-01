# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared printer tier over CIRv2: the statement/expression walk and the C declarator engine.
##
## | leaf                            | contract                                                                                             |
## | ------------------------------- | ---------------------------------------------------------------------------------------------------- |
## | failing / fail                  | first-refusal-wins walk state over each printer's `PrintCtx` (`err`, `fnName`)                       |
## | codegenModule                   | `(artifact, err)` to entry-point-result conversion                                                   |
## | roundUpMultiple, scalarByteSize | shared layout arithmetic over CIRv2 types                                                            |
## | emitVectorComponent             | vector component letters (`F0`..`F3` to `x`..`w`)                                                    |
## | emitLiteral                     | literal token forms of one target, over its lit and scalar rows                                      |
## | printCast                       | one bit reinterpret through the target's cast rules, unmatched casts keep the conversion form        |
## | LoopStyle, printLoopText        | loop text of one target: unroll directive, head, brace style                                         |
## | printStmtList                   | statement-list walk: sLoop through `printLoopText`, others through `printStmt`                       |
## | printStmts                      | statement-list walk with the target's kind-row loop style                                            |
## | printStmt                       | dispatch: shared variants inline, target policy via printVar/printConst/printAsgn/printCall/printAsm |
## | printExpr                       | expression dispatch: shared branches inline, target kinds via printTargetExpr, default routes inline |
## | printTargetExpr                 | the per-printer expression walk, owning the kinds in the printer's `targetExprKinds` set             |
## | printCallArgs, printCallText    | call argument texts, then one call's text: builtin vocabulary when present, else the plain call form |
## | printIteElse, printEmitOperand  | sIte else tail and `{.emit.}` operand text, shared forms a printer overrides                         |
## | nonFiniteFloatLit               | non-finite float literal token, classified against the printer's token table                         |
##
## C declarator engine, driven by the registry rows that each printer supplies.
##
## - `scalarName`/`leafName`/`bareName`: type names from `ScalarRow` and `VectorRule`
## - `cDecl`: C-style declaration text from type structure
## - a leaf that the rows refuse returns "", so the caller composes the name or refuses
##
## Every printer under `backends/` imports these, no printer owns a copy.
## A leaf belongs here when two or more printers need the same text.
## Single-target emission stays in the printer.

import std/[macros, math, strutils]
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

proc roundUpMultiple*(v, multiple: int): int =
  ## Smallest multiple of `multiple` that is >= `v`.
  doAssert multiple > 0
  (v + multiple - 1) div multiple * multiple

proc scalarByteSize*(t: CirType, boolWidth: static int = 1): int =
  ## Byte width of one scalar value.
  ##
  ## - ctBool: `boolWidth`, std430 packs a bool into 1 byte and the WGSL
  ##   launch ABI packs one into 4
  ## - ctInt, ctUint, ctFloat: `bits div 8`
  ## - widthless kinds: 0
  case t.kind
  of ctBool: boolWidth
  of ctInt, ctUint, ctFloat: t.bits div 8
  else: 0

proc emitVectorComponent*(name: string): string =
  ## Component letter of a CIRv2 vector component name (`F0`..`F3`).
  ##
  ## Returns "x", "y", "z" or "w", "" when the name is not a component.
  case name
  of "F0": "x"
  of "F1": "y"
  of "F2": "z"
  of "F3": "w"
  else: ""

# ═══════════════════════════════════════════════════════════════════════
# Declarator engine: type names and declaration forms over the registry rows
# ═══════════════════════════════════════════════════════════════════════

proc scalarName*(row: ScalarRow, t: CirType): string =
  ## Scalar type name of a leaf type from the target's scalar row.
  ##
  ## Returns "" when the kind is not a scalar leaf or the (kind, width) cell is refused.
  ## The caller composes the name or refuses, per its own policy.
  ##
  ##   result = scalarName(mslScalarRow(), t)  # "half" for a 16-bit float
  case t.kind
  of ctVoid, ctBool:
    for c in row.cells:
      if c.kind == t.kind and c.spelling.len > 0:
        return c.spelling
  of ctInt, ctUint, ctFloat:
    for c in row.cells:
      if c.kind == t.kind and c.width == t.bits and c.spelling.len > 0:
        return c.spelling
  else:
    discard
  return ""

proc vectorName*(rule: VectorRule, row: ScalarRow, t: CirType): string =
  ## Vector type name of `t` from the target's vector rule.
  ##
  ## Returns "" when `t` is not a vector, its element has no cell in the rule,
  ## or its lane count is outside `rule.lanes`. The caller refuses.
  ##
  ##   result = vectorName(glslVectorRule(), glslScalarRow(), t)  # "vec4"
  if t.kind != ctVector:
    return ""
  for e in rule.elems:
    if e.kind == t.vElem.kind and e.width == t.vElem.bits:
      if t.vN notin rule.lanes:
        return ""
      case rule.nameForm
      of vnfPrefixed:
        let prefix = if e.prefix.len > 0: e.prefix else: scalarName(row, t.vElem)
        return prefix & $t.vN
      of vnfGenericElem:
        return e.prefix & $t.vN & "<" & scalarName(row, t.vElem) & ">"
  return ""

proc leafName*(row: ScalarRow, rule: VectorRule, t: CirType): string =
  ## Bare type name of a non-array leaf type: the scalar, vector or struct name.
  ##
  ## Returns "" for pointers, arrays, proc and span types. The caller composes
  ## the name or refuses, per its own policy.
  case t.kind
  of ctVoid, ctBool, ctInt, ctUint, ctFloat:
    result = scalarName(row, t)
  of ctVector:
    result = vectorName(rule, row, t)
  of ctStruct:
    result = t.name
  of ctPtr, ctAptr, ctArray, ctProc, ctSpan:
    result = ""

proc bareName*(row: ScalarRow, rule: VectorRule, t: CirType): string =
  ## Bare type name of `t`: the form casts and compound literals use.
  ## No identifier, pointer stars trailing the pointee name.
  ##
  ## Returns "" for arrays, proc and span types.
  ## The caller composes the name or refuses, per its own policy.
  ##
  ##   result = bareName(cudaScalarRow(), cudaVectorRule(), t)  # "float**"
  if t.kind in {ctPtr, ctAptr}:
    return bareName(row, rule, t.pElem) & "*"
  return leafName(row, rule, t)

proc cDecl*(row: ScalarRow, rule: VectorRule, target: string, t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## Declaration text of `t` with `ident` appended in the C declarator form:
  ## the array dimensions move behind the identifier and nest into the declarator
  ## (`T ident[N]`, pointer-to-array `T (*ident)[N]`).
  ##
  ## - `allowEmptyIdent` permits no identifier, return types and casts use it
  ## - an array without an identifier otherwise refuses: the returned text is
  ##   `<target: array needs an identifier>` (the caller's refusal channel)
  ##
  ##   result = cDecl(cudaScalarRow(), cudaVectorRule(), "cuda", t, "buf", false)
  if t.kind == ctArray:
    if ident.len == 0 and not allowEmptyIdent:
      return "<" & target & ": array needs an identifier>"
    if t.aElem.kind == ctArray:
      return cDecl(row, rule, target, t.aElem, "(" & ident & ")" & "[" & $t.aN & "]",
        allowEmptyIdent)
    return bareName(row, rule, t.aElem) & ' ' & ident & '[' & $t.aN & ']'
  result = bareName(row, rule, t)
  if ident.len > 0:
    result.add ' ' & ident

proc nonFiniteFloatLit*(e: CirExpr, tok: FloatLitTokens): string =
  ## Emission token of a non-finite float literal.
  ##
  ## Returns "" when the value is finite.
  ##
  ## Finite values keep the caller's own width suffix:
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

template emitLiteral*(e: CirExpr, litRow: LitRow, row: ScalarRow): string =
  ## Returns the literal token of `e`, chosen from the target's lit row
  ## `litRow` according to the literal type of `e`.
  ##
  ## - a narrow (8/16-bit) integer prints as a cast to its scalar row name
  ##   when the lit row sets `narrowCast`, otherwise bare
  ## - a non-finite float takes its bit-pattern token from `litRow.nonFinite`
  ## - a literal kind outside the rows prints `<printerTarget: not a literal>`
  mixin printerTarget
  block:
    var lit: string
    case e.kind
    of eIntLit:
      if e.litType.bits notin {32, 64} and litRow.int.narrowCast:
        lit = "(" & scalarName(row, e.litType) & ")" & $e.value
      else:
        case e.litType.kind
        of ctUint:
          case e.litType.bits
          of 64: lit = litRow.int.u64Prefix & $e.value & litRow.int.u64Suffix
          of 32: lit = $e.value & litRow.int.u32Suffix
          else: lit = $e.value
        else:
          case e.litType.bits
          of 64: lit = litRow.int.i64Prefix & $e.value & litRow.int.i64Suffix
          else: lit = $e.value
    of eFloatLit:
      lit = nonFiniteFloatLit(e, litRow.nonFinite)
      if lit.len == 0:
        case e.fLitType.bits
        of 16: lit = litRow.f16Prefix & $e.fValue & litRow.f16Suffix
        of 32: lit = $e.fValue & litRow.f32Suffix
        else: lit = litRow.f64Prefix & $e.fValue & litRow.f64Suffix
    of eNil:
      lit = litRow.nilToken
      if lit.len == 0:
        lit = "<" & printerTarget & ": not a literal>"
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

proc printLoopText*(ctx: var auto, s: CirStmt, style: LoopStyle): string =
  ## Returns the text of one `sLoop` statement: the unroll directive,
  ## then the loop head with its indented body.
  ##
  ## The body recurses through `ctx.printStmts`, the head condition
  ## prints through `ctx.printExpr`.
  mixin printStmts, printExpr
  ctx.indent.inc
  let bodyTxt = ctx.printStmts(s.loopBody)
  ctx.indent.dec
  let pad = spaces(ctx.indent * 2)
  let directive = if s.unroll and style.unrollDirective.len > 0:
    pad & style.unrollDirective & "\n"
  else:
    ""
  let cond = ctx.printExpr(s.loopCond)
  let head = if style.braceSameLine:
    "while (" & cond & ") {\n"
  else:
    "while (" & cond & ")\n"
  result = directive & pad & head
  if not style.braceSameLine:
    result &= pad & "{\n"
  result &= bodyTxt & "\n" & pad & '}'

proc printIteElse*(ctx: var auto, elseBody: seq[CirStmt], pad: string): string =
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

proc printEmitOperand*(ctx: var auto, op: CirExpr): string =
  ## Returns the splice text of one `{.emit.}` operand: the rendered expression
  ## when the target's kind row states `emitOperandRendersExpr`.
  ## The symbol name is the default splice.
  if ctx.policy.emitOperandRendersExpr:
    ctx.printExpr(op)
  else:
    op.name

proc printCallArgs*(ctx: var auto, e: CirExpr): seq[string] =
  ## Printed texts of one `eCall`'s arguments in order, the plain expression form.
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

proc printExpr*(ctx: var auto, e: CirExpr): string =
  ## Prints one structured expression.
  ##
  ## - shared variants emit here: the operator branches, the literal head
  ##   through `emitLiteral` over the context's lit and scalar rows
  ## - the call skeleton prints through `printCallArgs` and `printCallText`
  ## - the kinds a printer emits itself route to its `printTargetExpr`,
  ##   named by its `targetExprKinds` set
  ## - subscript, dereference, address-of and ternary kinds without a target route take the shared default forms below
  mixin printExpr, printTargetExpr
  if ctx.failing():
    return ""
  case e.kind
  of eNil, eTrue, eFalse, eIntLit, eFloatLit:
    if e.kind in ctx.policy.targetExprKinds:
      return ctx.printTargetExpr(e)
    return emitLiteral(e, ctx.lit, ctx.row)
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
    # C-family operators short-circuit, operands may carry calls
    let op = if e.kind == eAnd: "&&" else: "||"
    return "(" & ctx.printExpr(e.andLeft) & ' ' & op & ' ' & ctx.printExpr(e.andRight) & ")"
  of eTernary, eAt, ePat, eDeref, eAddr:
    if e.kind in ctx.policy.targetExprKinds:
      return ctx.printTargetExpr(e)
    # shared default forms for the kinds outside the printer's `targetExprKinds` set
    if e.kind in {eAt, ePat}:
      let base = if e.kind == eAt: e.atBase else: e.patBase
      let idx = if e.kind == eAt: e.atIndex else: e.patIndex
      var baseTxt = ctx.printExpr(base)
      if base.kind in {eCast, eConv}:
        # a cast binds looser than the subscript: parenthesize the base
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

proc printStmt*(ctx: var auto, s: CirStmt): string =
  ## Prints one structured statement.
  ##
  ## Shared variants emit here: `sIte`, `sRet`, `sDiscard`, `sEmit`.
  ## A loop prints through
  ## `printLoopText` off `printStmtList`. Per-target policy resolves
  ## at instantiation through the printer's own procs:
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
  of sLoop:
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
    # no terminator appended, the text supplies its own terminators
    var text = ""
    for i, part in s.emitParts:
      text.add(part)
      if i < s.emitOperands.len:
        text.add(ctx.printEmitOperand(s.emitOperands[i]))
    result = pad & text

proc printAsgn*(ctx: var auto, s: CirStmt): string =
  ## One assignment, the plain form `lhs = rhs;`.
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

proc printStmtList*(ctx: var auto, stmts: seq[CirStmt], style: LoopStyle): string =
  ## Prints one statement list, one line per printed statement:
  ##
  ## - `sLoop`: `printLoopText` with the target's `style`
  ## - every other statement: `ctx.printStmt`, an empty text drops out
  ## - printed lines join with a newline separator
  var lines: seq[string]
  for s in stmts:
    let txt = if s.kind == sLoop: printLoopText(ctx, s, style)
              else: ctx.printStmt(s)
    if txt.len > 0:
      lines.add txt
  return lines.join("\n")

proc printStmts*(ctx: var auto, stmts: seq[CirStmt]): string =
  ## Prints one statement list with the target's loop style from its kind row.
  printStmtList(ctx, stmts, ctx.policy.loop)

proc codegenModule*[R](printed: (R, string)): R =
  ## Converts a printer result tuple into the `backends_api` entry-point return value.
  ##
  ## Returns the artifact. A non-empty refusal is a compile error naming the construct.
  if printed[1].len > 0:
    error(printed[1])
  result = printed[0]
