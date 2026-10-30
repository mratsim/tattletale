# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared printer leaves over CIRv2.
##
## - failing / fail: the first-refusal-wins walk state, a template over each
##   printer's own `PrintCtx` record (the `err` and `fnName` fields)
## - codegenModule: the `(artifact, err)` to entry-point-result conversion
## - roundUpMultiple, scalarByteSize: shared layout arithmetic over CIRv2 types
## - emitVectorComponent: vector component letters (`F0`..`F3` to `x`..`w`)
##
## Every printer under `backends/` imports these, no printer owns a copy.

import std/macros
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

proc codegenModule*[R](printed: (R, string)): R =
  ## Converts a printer result tuple into the `backends_api` entry-point return value.
  ##
  ## Returns the artifact. A non-empty refusal is a compile error naming the construct.
  if printed[1].len > 0:
    error(printed[1])
  result = printed[0]
