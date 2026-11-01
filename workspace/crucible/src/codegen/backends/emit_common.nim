# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared printer leaves over CIRv2.
##
## | leaf                            | contract                                                                       |
## | ------------------------------- | ------------------------------------------------------------------------------ |
## | failing / fail                  | first-refusal-wins walk state over each printer's `PrintCtx` (`err`, `fnName`) |
## | codegenModule                   | `(artifact, err)` to entry-point-result conversion                             |
## | roundUpMultiple, scalarByteSize | shared layout arithmetic over CIRv2 types                                      |
## | emitVectorComponent             | vector component letters (`F0`..`F3` to `x`..`w`)                              |
## | ScalarTokens, emitScalarToken   | scalar type names per target table, one walker over the leaf kinds             |
## | LitTokens, emitLiteral          | integer/float/nil literal token spellings per target table                     |
## | LoopStyle, printLoopText        | loop text of one target: unroll directive, head, brace style                   |
## | printStmtList                   | statement-list walk: sLoop through `printLoopText`, others through `printStmt` |
## | nonFiniteFloatLit               | non-finite float literal token, classified against the printer's token table   |
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

proc printStmtList*(ctx: var auto, stmts: seq[CirStmt], style: LoopStyle): string =
  ## Prints one statement list, one line per printed statement:
  ##
  ## - `sLoop`: `printLoopText` with the target's `style`
  ## - every other statement: `ctx.printStmt`, an empty text drops out
  ## - printed lines join with a newline separator
  mixin printStmt
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
