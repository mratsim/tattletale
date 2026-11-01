# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared printer leaves over CIRv2.
##
## | leaf                            | contract                                                                                             |
## | ------------------------------- | ---------------------------------------------------------------------------------------------------- |
## | failing / fail                  | first-refusal-wins walk state over each printer's `PrintCtx` (`err`, `fnName`)                       |
## | codegenModule                   | `(artifact, err)` to entry-point-result conversion                                                   |
## | roundUpMultiple, scalarByteSize | shared layout arithmetic over CIRv2 types                                                            |
## | emitVectorComponent             | vector component letters (`F0`..`F3` to `x`..`w`)                                                    |
## | ScalarTokens, emitScalarToken   | scalar type names per target table, one walker over the leaf kinds                                   |
## | LitTokens, emitLiteral          | integer/float/nil literal token forms per target table                                               |
## | LoopStyle, printLoopText        | loop text of one target: unroll directive, head, brace style                                         |
## | printStmtList                   | statement-list walk: sLoop through `printLoopText`, others through `printStmt`                       |
## | printStmt                       | dispatch: shared variants inline, target policy via printVar/printConst/printAsgn/printCall/printAsm |
## | printExpr                       | expression dispatch: shared branches inline, target kinds via printTargetExpr, default routes inline |
## | printTargetExpr                 | the per-printer expression walk, owning the kinds in the printer's `targetExprKinds` set             |
## | printCallArgs, printCallText    | call argument texts, then one call's text: builtin vocabulary when present, else the plain call form |
## | printIteElse, printEmitOperand  | sIte else tail and `{.emit.}` operand text, shared forms a printer overrides                         |
## | nonFiniteFloatLit               | non-finite float literal token, classified against the printer's token table                         |
##
## Every printer under `backends/` imports these, no printer owns a copy.

import std/[macros, math, strutils]
import ../ir/cirv2_lang
import ../ir/ast_datatypes

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

type
  ScalarTokens* = object
    ## Scalar type names of one printer target, by type kind and width.
    ##
    ## Widths name different C types on different targets: `long` is
    ## 64-bit on LP64 targets and 32-bit on LLP64 hosts.
    ## Plain `char` takes its signedness from the host ABI.
    ##
    ## Each target defines its own table. A "" field means the type
    ## is refused by the caller.
    void*, bool*: string
    i8*, i16*, i32*, i64*: string
    u8*, u16*, u32*, u64*: string
    f16*, f32*, f64*: string

proc emitScalarToken*(tok: ScalarTokens, t: CirType): string =
  ## Scalar type name of a leaf type from the target table `tok`.
  ##
  ## Returns "" when the kind is not a leaf or the width has no name:
  ## the caller composes the name or refuses, per its own policy.
  ##
  ##   of ctVoid, ctBool, ctInt, ctUint, ctFloat:
  ##     result = emitScalarToken(targetScalarTokens, t)
  case t.kind
  of ctVoid: tok.void
  of ctBool: tok.bool
  of ctInt:
    case t.bits
    of 8: tok.i8
    of 16: tok.i16
    of 32: tok.i32
    of 64: tok.i64
    else: ""
  of ctUint:
    case t.bits
    of 8: tok.u8
    of 16: tok.u16
    of 32: tok.u32
    of 64: tok.u64
    else: ""
  of ctFloat:
    case t.bits
    of 16: tok.f16
    of 32: tok.f32
    of 64: tok.f64
    else: ""
  else: ""

type
  FloatLitTokens* = object
    ## Non-finite float literal tokens for one printer target:
    ##
    ## - one field per literal width
    ## - Nim's `$` renders the special values `inf`, `-inf` and `nan`, and no
    ##   shader language accepts those as a literal token

    inf16*, negInf16*, nan16*: string
    inf32*, negInf32*, nan32*: string
    inf64*, negInf64*, nan64*: string

  IntLitTokens* = object
    ## Integer literal token chosen by signedness and width:
    ## `prefix & $value & suffix`.
    ##
    ## - 64-bit widths take `u64Prefix`/`u64Suffix` or `i64Prefix`/`i64Suffix`
    ## - 32-bit unsigned takes `u32Suffix`, 32-bit signed prints bare
    ## - narrow (8/16-bit) widths print bare or, per `narrowCast`, as a cast
    ##   to the scalar type name from `LitTokens.scalar`
    u64Prefix*, u64Suffix*, u32Suffix*: string
    i64Prefix*, i64Suffix*: string
    narrowCast*: bool

  LitTokens* = object
    ## Literal token table of one printer target:
    ##
    ## - `int`: integer token fields by signedness and width
    ## - `f16Prefix`/`f16Suffix`, `f32Suffix`, `f64Prefix`/`f64Suffix`:
    ##   finite float forms around the numeric text
    ## - `nilToken`: null pointer token, "" when the target refuses nil
    ##   before the literal printer sees it
    ## - `nonFinite` and `scalar`: non-finite float tokens and the scalar
    ##   type names feeding the narrow integer cast, see `nonFiniteFloatLit`
    int*: IntLitTokens
    f16Prefix*, f16Suffix*: string
    f32Suffix*: string
    f64Prefix*, f64Suffix*: string
    nilToken*: string
    nonFinite*: FloatLitTokens
    scalar*: ScalarTokens

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

template emitLiteral*(e: CirExpr, tok: LitTokens): string =
  ## Returns the literal token of `e`, chosen from the target table `tok`
  ## according to the literal type of `e`.
  ##
  ## - a width the table does not name prints the plain numeric text
  ## - a non-finite float takes its bit-pattern token from `tok.nonFinite`
  ## - a literal kind outside the table prints `<printerTarget: not a literal>`
  mixin printerTarget
  block:
    var lit: string
    case e.kind
    of eIntLit:
      if e.litType.bits notin {32, 64} and tok.int.narrowCast:
        lit = "(" & emitScalarToken(tok.scalar, e.litType) & ")" & $e.value
      else:
        case e.litType.kind
        of ctUint:
          case e.litType.bits
          of 64: lit = tok.int.u64Prefix & $e.value & tok.int.u64Suffix
          of 32: lit = $e.value & tok.int.u32Suffix
          else: lit = $e.value
        else:
          case e.litType.bits
          of 64: lit = tok.int.i64Prefix & $e.value & tok.int.i64Suffix
          else: lit = $e.value
    of eFloatLit:
      lit = nonFiniteFloatLit(e, tok.nonFinite)
      if lit.len == 0:
        case e.fLitType.bits
        of 16: lit = tok.f16Prefix & $e.fValue & tok.f16Suffix
        of 32: lit = $e.fValue & tok.f32Suffix
        else: lit = tok.f64Prefix & $e.fValue & tok.f64Suffix
    of eNil:
      lit = tok.nilToken
      if lit.len == 0:
        lit = "<" & printerTarget & ": not a literal>"
    of eTrue: lit = "true"
    of eFalse: lit = "false"
    else:
      lit = "<" & printerTarget & ": not a literal>"
    lit

type
  LoopStyle* = object
    ## Loop text of one printer target:
    ##
    ## - `unrollDirective`: `{.unroll.}` directive text, printed before
    ##   the loop at the loop indentation, "" prints no directive
    ## - `infiniteHead`: head of a condition-less loop, with its open brace
    ##   (`for (;;) {\n`, `loop {\n`)
    ## - `braceSameLine`: the open brace sits on the head line, otherwise it
    ##   prints on its own line under the head (HLSL)
    unrollDirective*, infiniteHead*: string
    braceSameLine*: bool

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
  let head = if s.loopCond != nil:
    if style.braceSameLine:
      "while (" & ctx.printExpr(s.loopCond) & ") {\n"
    else:
      "while (" & ctx.printExpr(s.loopCond) & ")\n"
  else:
    style.infiniteHead
  result = directive & pad & head
  if not style.braceSameLine:
    result &= pad & "{\n"
  result &= bodyTxt & "\n" & pad & '}'

proc foldsIteElse*(ctx: var auto): bool =
  ## Returns true when a single-`sIte` else body folds into
  ## an `else if` chain, false for the default braced else.
  false

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
  ## Returns the splice text of one `{.emit.}` operand.
  ##
  ## The shared form splices the operand's symbol name, a printer whose
  ## operands render as expressions overrides it.
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
  ##   through `emitLiteral` over the printer's `targetLitTokens`
  ## - the call skeleton prints through `printCallArgs` and `printCallText`
  ## - the kinds a printer emits itself route to its `printTargetExpr`,
  ##   named by its `targetExprKinds` set
  ## - subscript, dereference, address-of and ternary kinds without a target route take the shared default forms below
  mixin printExpr, printTargetExpr, targetExprKinds, targetLitTokens
  if ctx.failing():
    return ""
  case e.kind
  of eNil, eTrue, eFalse, eIntLit, eFloatLit:
    if e.kind in targetExprKinds:
      return ctx.printTargetExpr(e)
    return emitLiteral(e, targetLitTokens)
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
    if e.kind in targetExprKinds:
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
  of eBitNot:
    return "(~" & ctx.printExpr(e.unOperand) & ")"
  of eNeg:
    return "(-" & ctx.printExpr(e.unOperand) & ")"
  of eSym, eDot, eCast, eConv, eSizeof, eOconstr, eAconstr, eCall:
    return ctx.printTargetExpr(e)

proc printStmt*(ctx: var auto, s: CirStmt): string =
  ## Prints one structured statement.
  ##
  ## Shared variants emit here: `sIte`, `sLoop`, `sScope`, `sAssume`,
  ## `sRet`, `sDiscard`, `sCase`, `sEmit`. Per-target policy resolves
  ## at instantiation through the printer's own procs:
  ##
  ## - `printVar`, `printConst`: storage-class refusals and declaration text
  ## - `printAsgn`: assignment policy, pointer rebinds and buffer writes
  ## - `printCall`: barrier intrinsics and builtin call vocabulary
  ## - `printAsm`: inline assembly form, refusal text or raw splice
  mixin printStmts, printExpr, printVar, printConst, printAsgn, printCall, printAsm,
        foldsIteElse, printIteElse, printEmitOperand
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
    if s.elseBody.len == 1 and s.elseBody[0].kind == sIte and ctx.foldsIteElse():
      # an else side of one ite is the elif chain, printed as `else if`
      result &= " else " & ctx.printStmt(s.elseBody[0])
    else:
      result &= ctx.printIteElse(s.elseBody, pad)
  of sLoop:
    # loops print through printStmtList
    result = ""
  of sScope:
    result = pad & "{\n"
    ctx.indent.inc
    result &= ctx.printStmts(s.scopeBody)
    ctx.indent.dec
    result &= '\n' & pad & '}'
  of sAssume:
    # a loop-bound hint, no target prints one
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
  of sCase:
    ctx.fail("case statements are outside the CIRv2 closure, the frontend refuses them")
    result = ""

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

proc codegenModule*[R](printed: (R, string)): R =
  ## Converts a printer result tuple into the `backends_api` entry-point return value.
  ##
  ## Returns the artifact. A non-empty refusal is a compile error naming the construct.
  if printed[1].len > 0:
    error(printed[1])
  result = printed[0]
