# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Declarative registry of the text printers' target data.
##
## A row is a const object of plain data: strings, bools, ints, enums and tables of those.
## No field stores a proc or a placeholder whose value depends on a body walk.
##
## | family | row        | contents                                      |
## | ------ | ---------- | --------------------------------------------- |
## | scalar | ScalarRow  | type names and support cells by (kind, width) |
## | lit    | LitRow     | literal token forms                           |
## | kind   | KindRow    | walk policy: loops, else folding, routing set |
## | conv   | ConvRow    | conversion name tables                        |
## | vector | VectorRule | a target's real vector coverage               |
##
## Token types that the rows embed (`ScalarTokens`, `LitTokens`, `LoopStyle`) are defined here,
## so the registry stays the one home of the printers' static data schema.
##
## `genTargets` (targets_macros) parses row blocks into these types and generates one lookup proc per family and target.
## Checks run at registration as one compile-time block: full (kind, width) coverage per key,
## refused cells without a name, and cross-target name collisions. A missing or drifted row is a compile error.

import std/tables
import ../ir/cirv2_lang

type
  ScalarTokens* = object
    ## Scalar type names of one printer target, by type kind and width.
    ##
    ## Plain `char` takes its signedness from the host ABI.
    ##
    ## Each target defines its own table. A "" field means the type
    ## is refused by the caller.
    void*, bool*: string
    i8*, i16*, i32*, i64*: string
    u8*, u16*, u32*, u64*: string
    f16*, f32*, f64*: string

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

type
  ScalarCell* = object
    ## One (kind, width) cell of a scalar row.
    kind*: CirTypeKind
    width*: int
    ## Bit width, 0 for the widthless kinds (void, bool).
    spelling*: string
    ## Type name the target prints, "" for a refused cell.
    supported*: bool

  ScalarRow* = object
    ## Scalar type names of one target by (kind, width) cell.
    ## Every cell of `ScalarCellAxis` appears exactly once per target.
    ## Widths name different C types on different hosts: 64-bit `long` on LP64 hosts, 32-bit on LLP64 hosts.
    ## Each target states its own names, no cell inherits a name from another target.
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

  ConvRow* = object
    ## Name tables of one target: conversion data that is a pure name-to-text mapping.
    ## Conversions needing operand type analysis or multi-statement emission stay procedural in the printers.
    builtinNames*: Table[string, string]
    ## `{.builtin.}` numeric conversion name to call text, `$1` stands for the call's argument list.
    fpRefusals*: seq[string]
    ## `{.builtin.}` fp-conversion names the target refuses.

  VectorNameForm* = enum
    ## Composition of a vector type name from the element's prefix and the lane count.
    vnfPrefixed
    ## `prefix & $laneCount` forms the C-family names (`vec3`, `float4`). An empty prefix composes the element's scalar name from the scalar row.
    vnfGenericElem
    ## `prefix & $laneCount & '<' & elementName & '>'` forms the generic-element names (`vec3<f32>`).

  VectorContext* = enum
    ## Where a target's printer emits vector types.
    vcNone
    vcCoordinate
    ## Only in coordinate-builtin context.
    vcEverywhere

  VectorRule* = object
    ## A target's real vector coverage, as its printer emits it today.
    context*: VectorContext
    elems*: seq[tuple[kind: CirTypeKind, width: int, prefix: string]]
    ## Element types the target names, with the vector name's prefix.
    ## The name composes from the prefix and the lane count per `nameForm`.
    ## Empty for `vcNone`.
    lanes*: seq[int]
    ## Lane counts the target emits.
    nameForm*: VectorNameForm = vnfPrefixed
    ## Composition of the vector type name from the element's prefix and the lane count.

const ScalarCellAxis*: array[13, tuple[kind: CirTypeKind, width: int]] = [
  (ctVoid, 0), (ctBool, 0),
  (ctInt, 8), (ctInt, 16), (ctInt, 32), (ctInt, 64),
  (ctUint, 8), (ctUint, 16), (ctUint, 32), (ctUint, 64),
  (ctFloat, 16), (ctFloat, 32), (ctFloat, 64)]

proc checkScalarRow*(row: ScalarRow, target: string) {.compileTime.} =
  ## Registration checks over one target's scalar row:
  ##
  ## - every cell of `ScalarCellAxis` appears exactly once, a second row for an existing cell fails the compile
  ## - a supported cell has a type name, a refused cell has none
  ## - a type name denotes at most one cell, a bool cell's name may repeat another
  ##   cell's name (OpenCL prints bool locals as `int` for the 4-byte launch ABI)
  ##
  ## Raises `AssertionDefect` at compile time on the first violated check.
  for axisCell in ScalarCellAxis:
    var count = 0
    for c in row.cells:
      if c.kind == axisCell.kind and c.width == axisCell.width:
        inc count
    doAssert count == 1, "target " & target & ": scalar cell (" &
      $axisCell.kind & ", " & $axisCell.width & ") appears " & $count &
      " times, exactly once required"
  for c in row.cells:
    if c.supported:
      doAssert c.spelling.len > 0,
        "target " & target & ": supported cell (" & $c.kind & ", " & $c.width &
        ") needs a name"
    else:
      doAssert c.spelling.len == 0,
        "target " & target & ": refused cell (" & $c.kind & ", " & $c.width &
        ") needs no name"
  for i, a in row.cells:
    if a.spelling.len == 0 or a.kind == ctBool:
      continue
    for j in i + 1 ..< row.cells.len:
      if row.cells[j].kind == ctBool:
        continue
      doAssert row.cells[j].spelling != a.spelling,
        "target " & target & ": name '" & a.spelling & "' denotes two cells, (" &
        $a.kind & ", " & $a.width & ") and (" & $row.cells[j].kind & ", " &
        $row.cells[j].width & ")"

proc checkScalarRowsTogether*(
    rows: openArray[tuple[name: string, row: ScalarRow]]) {.compileTime.} =
  ## Cross-target scalar check:
  ##
  ## - a name two targets share must denote the same (kind, width) cell
  ## - divergent names for one cell are explicit per-target data, msl `char`/`long`
  ##   against cuda `signed char`/`long long`
  ## - a bool cell's name may repeat a non-bool cell's name, see `checkScalarRow`
  ##
  ## Returns after catching a name that drifted onto a cell of another width or kind.
  ## Raises `AssertionDefect` naming both targets at compile time.
  for i in 0 ..< rows.len:
    for j in i + 1 ..< rows.len:
      for a in rows[i].row.cells:
        if a.spelling.len == 0:
          continue
        for b in rows[j].row.cells:
          if b.spelling == a.spelling:
            if a.kind == ctBool or b.kind == ctBool:
              continue
            doAssert a.kind == b.kind and a.width == b.width,
              "name '" & a.spelling & "' denotes (" & $a.kind & ", " &
              $a.width & ") on " & rows[i].name & " but (" & $b.kind & ", " &
              $b.width & ") on " & rows[j].name
