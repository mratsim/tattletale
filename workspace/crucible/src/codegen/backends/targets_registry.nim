# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Schema of the printers' static target data.
##
## A row is a const object of plain data: strings, bools, ints, enums, sets and tables of those.
## No field stores a proc or a placeholder whose value depends on a body walk.
##
## | type       | contents                                      |
## | ---------- | --------------------------------------------- |
## | ScalarRow  | type names by (kind, width) cell              |
## | LitRow     | literal token forms                           |
## | KindRow    | walk policy: loops, else folding, routing set |
## | VectorRule | a target's real vector coverage               |
## | CastRule   | one bit-reinterpret coverage rule             |
##
## Targets state their rows in the `genTargets:` DSL of `targets_rows.nim`.
## The macro in `targets_macros.nim` validates each row against these types
## during compilation and generates one lookup per row family.
## A row missing the schema fails the compile at the row definition.

import ../ir/cirv2_lang

type
  FloatLitTokens* = object
    ## Non-finite float literal tokens for one printer target:
    ##
    ## - one field per literal width
    ## - an empty token is a refusal: the target emits no form for that width and the printer refuses the literal at the emission site
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
    ##   to the scalar type name from the target's scalar row
    u64Prefix*, u64Suffix*, u32Suffix*: string
    i64Prefix*, i64Suffix*: string
    narrowCast*: bool

  LoopStyle* = object
    ## Loop text of one printer target:
    ##
    ## - `unrollDirective`: `{.unroll.}` directive text, printed before
    ##   the loop at the loop indentation, "" prints no directive
    ## - `braceSameLine`: the open brace sits on the head line, otherwise it
    ##   prints on its own line under the head (HLSL)
    unrollDirective*: string
    braceSameLine*: bool

type
  ScalarCell* = object
    ## One (kind, width) cell of a scalar row.
    kind*: CirTypeKind
    width*: int
    ## Bit width, 0 for the widthless kinds (void, bool).
    spelling*: string
    ## Type name the target prints, "" for a refused cell.

  ScalarRow* = object
    ## Scalar type names of one target by (kind, width) cell.
    ## Every cell of `ScalarCellAxis` appears exactly once per target.
    cells*: seq[ScalarCell]

  LitRow* = object
    ## Literal token forms of one target.
    ## No exponent or hex-float fields: no printer handles either form.
    int*: IntLitTokens
    f16Prefix*, f16Suffix*: string
    f32Suffix*: string
    f64Prefix*, f64Suffix*: string
    nilToken*: string
    nonFinite*: FloatLitTokens

  KindRow* = object
    ## Per-target policy where the shared expression and statement walks branch on target.
    ## No for-range, switch or sizeof field: no printer emits the first two, sizeof routes per target through `printTargetExpr`.
    loop*: LoopStyle
    foldsIteElse*: bool
    ## A single-statement else body folds into an `else if` chain.
    emitOperandRendersExpr*: bool
    ## A `{.emit.}` operand renders as an expression when true, splices its symbol name when false.
    targetExprKinds*: set[CirExprKind]
    ## Optional expression kinds the printer routes through its own `printTargetExpr` proc.
    ## Eight common kinds route unconditionally and are not members of this set.

  VectorNameForm* = enum
    ## Composition of a vector type name from the element's prefix and the lane count.
    vnfPrefixed
    ## `prefix & $laneCount` forms the C-family names (`vec3`, `float4`). An empty prefix composes the element's scalar name from the scalar row.
    vnfGenericElem
    ## `prefix & $laneCount & '<' & elementName & '>'` forms the generic-element names (`vec3<f32>`).

  VectorRule* = object
    ## A target's real vector coverage, as its printer emits it today.
    elems*: seq[tuple[kind: CirTypeKind, width: int, prefix: string]]
    ## Element types the target names, with the vector name's prefix.
    ## The name composes from the prefix and the lane count per `nameForm`.
    ## Empty when the target names no vectors.
    lanes*: seq[int]
    ## Lane counts the target emits.
    nameForm*: VectorNameForm = vnfPrefixed
    ## Composition of the vector type name from the element's prefix and the lane count.

  CastWidth* = range[0..255]
  ## Bit width a cast rule matches.

  CastRule* = object
    ## One bit-cast coverage rule of a target: the first matching rule emits.
    ##
    ## - a rule with `srcAny` matches any source type, otherwise the source
    ##   type must exist and be a scalar leaf of `srcKinds` and, when
    ##   `srcWidths` is non-empty, of `srcWidths`
    ## - the destination must be of `dstKinds` and, when `dstWidths` is
    ##   non-empty, of `dstWidths`, with `sameWidth` also requiring equal
    ##   source and destination widths
    ## - a `spelling` rule emits its text, `$1` stands for the printed
    ##   operand and `$2` for the destination type name
    ## - an empty `spelling` makes the rule a refusal carrying `refusal`,
    ##   `$2` stands for the destination type name
    srcAny*: bool
    srcKinds*, dstKinds*: set[CirTypeKind]
    srcWidths*, dstWidths*: set[CastWidth]
    sameWidth*: bool
    spelling*, refusal*: string

const ScalarCellAxis*: array[13, tuple[tok: string, kind: CirTypeKind, width: int]] = [
  ("void", ctVoid, 0), ("bool", ctBool, 0),
  ("i8", ctInt, 8), ("i16", ctInt, 16), ("i32", ctInt, 32), ("i64", ctInt, 64),
  ("u8", ctUint, 8), ("u16", ctUint, 16), ("u32", ctUint, 32), ("u64", ctUint, 64),
  ("f16", ctFloat, 16), ("f32", ctFloat, 32), ("f64", ctFloat, 64)]
  ## (kind, width) cells with the DSL token naming each cell.
  ## Every scalar row and vector element row keys its entries to this axis.
  ##
  ## - widths name different C types on different hosts, 64-bit `long`
  ##   on LP64 hosts against 32-bit on LLP64 hosts
  ## - each target states its own names, one cell never inherits another
  ##   target's name
