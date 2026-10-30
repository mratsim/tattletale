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
## | nonFiniteFloatLit               | non-finite float literal token, classified against the printer's token table   |
##
## Every printer under `backends/` imports these, no printer owns a copy.

import std/[macros, math]
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
  FloatLitTokens* = object
    ## Non-finite float literal tokens for one printer target:
    ##
    ## - one field per literal width
    ## - Nim's `$` renders the special values `inf`, `-inf` and `nan`, and no
    ##   shader language accepts those as a literal token

    inf16*, negInf16*, nan16*: string
    inf32*, negInf32*, nan32*: string
    inf64*, negInf64*, nan64*: string

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

proc codegenModule*[R](printed: (R, string)): R =
  ## Converts a printer result tuple into the `backends_api` entry-point return value.
  ##
  ## Returns the artifact. A non-empty refusal is a compile error naming the construct.
  if printed[1].len > 0:
    error(printed[1])
  result = printed[0]
