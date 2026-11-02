# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## `genTargets`: the declarative target-table macro and the schema it fills.
##
## One `genTargets:` invocation declares every target's sections
## and generates one lookup const per (target, family).
## Every authoring mistake fails the compile at its own line:
##
## - an unknown family, field, cell or value shape
## - a duplicate target or family section
## - a target that misses a required family
## - one type name that denotes two (kind, width) cells

import std/[macros, sequtils, strutils, tables]
import ../codegen/ir/ast_datatypes
import ../codegen/ir/cirv2_lang

# ═══════════════════════════════════════════════════════════════════════
#  Schema
# ═══════════════════════════════════════════════════════════════════════

type
  LoopStyle* = object
    ## Loop text of one printer target:
    ##
    ## - `unrollDirective`: the `{.unroll.}` directive text, the printer
    ##   places it before the loop at the loop indentation, "" places none
    ## - `braceSameLine`: the printer places the open brace on the head line,
    ##   otherwise on its own line under the head (HLSL)
    unrollDirective*: string
    braceSameLine*: bool

  WalkPolicy* = object
    ## Per-target policy where the shared expression and statement walks
    ## branch on target, sizeof routes per target through `printTargetExpr`,
    ## no printer emits for-range or switch.
    loop*: LoopStyle
    foldsIteElse*: bool
    ## When true the printer folds a single-statement else body into an `else if` chain.
    emitOperandRendersExpr*: bool
    ## When true the printer renders a `{.emit.}` operand as an expression,
    ## when false it splices the operand's symbol name.
    targetExprKinds*: set[CirExprKind]
    ## Expression kinds routed through the printer's `printTargetExpr` proc,
    ## eight common kinds route unconditionally outside this set.

  VectorNameForm* = enum
    ## How the printer composes a vector type name from the element's
    ## prefix and the lane count.
    vnfPrefixed
    ## C-family names (`vec3`, `float4`) composed as `prefix & $laneCount`,
    ## the element's scalar name substitutes an empty prefix.
    vnfGenericElem
    ## Generic-element names (`vec3<f32>`) composed as
    ## `prefix & $laneCount & '<' & elementName & '>'`.

  VectorRule* = object
    ## A target's real vector coverage, as its printer emits it today.
    elems*: seq[tuple[kind: CirTypeKind, width: int, prefix: string]]
    ## Element types with the vector name's prefix, empty when the target
    ## has no vectors. The printer composes the name from the prefix
    ## and the lane count per `nameForm`.
    lanes*: seq[int]
    ## Lane counts the target emits.
    nameForm*: VectorNameForm = vnfPrefixed
    ## How the printer composes the vector type name.

# ═══════════════════════════════════════════════════════════════════════
#  The conversion-builtin mapping macro
# ═══════════════════════════════════════════════════════════════════════

const ConvCellTokens* = ["f16", "bf16", "f32", "f64", "i16", "i32", "i64", "u16", "u32", "u64"]

macro convBuiltin*(body: untyped) =
  ## Declarative conversion mapping: one `bkX:` section per backend,
  ## one arrow row per mapping, `f32->f16: "half"`.
  ##
  ## - the sections key on the BackendKind members
  ## - the rows key on the (source, target) cell tokens shared with
  ##   the codegen ScalarCellAxis vocabulary
  ## - a row's value names the native call, the emission appends
  ##   the argument list in parentheses
  ## - an absent backend section or cell pair lets the canonical name
  ##   pass through
  var pairs: seq[NimNode]
  for section in body:
    if section.kind != nnkCall or section[0].kind != nnkIdent or
        section[1].kind != nnkStmtList:
      error("convBuiltin: expected a 'bkX:' section", section)
    let backend = $section[0]
    var known = false
    for member in BackendKind:
      if $member == backend:
        known = true
        break
    if not known:
      error("convBuiltin: unknown backend '" & backend & "'", section)
    for row in section[1]:
      if row.kind != nnkInfix or $row[0] != "->" or
          row[1].kind != nnkIdent or row[2].kind != nnkIdent or
          row[3].kind != nnkStmtList or row[3].len != 1 or row[3][0].kind != nnkStrLit:
        error("convBuiltin: expected a 'fromCell->toCell: \"native\"' row in the " & backend & " section", row)
      for cellTok in [row[1], row[2]]:
        if $cellTok notin ConvCellTokens:
          error("convBuiltin: unknown cell '" & $cellTok & "' in the " & backend & " section", row)
      let key = nnkPar.newTree(ident backend, newLit $row[1], newLit $row[2])
      pairs.add nnkExprColonExpr.newTree(key, row[3][0])
  result = nnkConstSection.newTree(nnkConstDef.newTree(
    nnkPostfix.newTree(ident"*", ident"ConvBuiltinNames"),
    newEmptyNode(),
    nnkCall.newTree(nnkDotExpr.newTree(
      nnkTableConstr.newTree(pairs), ident"toTable"))))

# ═══════════════════════════════════════════════════════════════════════
#  The target vocabulary macro, cell axis and section parsers
# ═══════════════════════════════════════════════════════════════════════

## Cell axis: the shared (kind, width) vocabulary for the scalar names,
## the literal rows and the vector elements.
##
## - the same bit width has different C type names on different hosts:
##   64-bit `long` on LP64 hosts, 32-bit on LLP64 hosts
## - each backend declares its own names
const ScalarCellAxis*: array[15, tuple[tok: string, kind: CirTypeKind, width: int]] = [
  ("void", ctVoid, 0), ("bool", ctBool, 0), ("string", ctString, 0),
  ("i8", ctInt, 8), ("i16", ctInt, 16), ("i32", ctInt, 32), ("i64", ctInt, 64),
  ("u8", ctUint, 8), ("u16", ctUint, 16), ("u32", ctUint, 32), ("u64", ctUint, 64),
  ("f16", ctFloat, 16), ("bf16", ctBf16, 16), ("f32", ctFloat, 32), ("f64", ctFloat, 64)]
  ## (kind, width) cells with the DSL token naming each cell.
  ## Every scalar-name table and vector element list keys its entries to this axis.
  ##
  ## - widths name different C types on different hosts, 64-bit `long`
  ##   on LP64 hosts against 32-bit on LLP64 hosts
  ## - each target declares its own names, one cell never inherits another
  ##   target's name

proc getCellToken*(kind: CirTypeKind, width: int): string =
  ## Returns the `ScalarCellAxis` token of a (kind, width) cell, "" off-axis.
  for axis in ScalarCellAxis:
    if axis.kind == kind and axis.width == width:
      return axis.tok
  return ""

proc findCellIndex(tok: string): int {.compileTime.} =
  ## Index of a cell token in `ScalarCellAxis`, -1 when unknown.
  for i, axis in ScalarCellAxis:
    if axis.tok == tok:
      return i
  return -1

type
  ParsedCell = tuple[idx: int, name: string, node: NimNode]

proc parseCellPair*(arg: NimNode, display: string): ParsedCell {.compileTime.} =
  ## Returns the `ScalarCellAxis` index and type name of one `cell "name"` entry,
  ## an unprefixed cell carries the empty name there.
  if arg.kind == nnkIdent:
    result = (findCellIndex($arg), "", arg)
  elif arg.kind == nnkCommand and arg.len == 2 and arg[1].kind == nnkStrLit:
    result = (findCellIndex($arg[0]), arg[1].strVal, arg[1])
  else:
    error("targetVocab: expected a 'cell \"name\"' entry in " & display, arg)
  if result.idx < 0:
    error("targetVocab: unknown cell '" & (if arg.kind == nnkIdent: $arg else: $arg[0]) & "' in " & display, arg)

proc cellKey(idx: int): string {.compileTime.} =
  "(" & $ScalarCellAxis[idx].kind & ", " & $ScalarCellAxis[idx].width & ")"

proc checkCellNames(all: seq[tuple[display: string, cells: seq[ParsedCell]]]) {.compileTime.} =
  ## Fails the compile when one type name denotes two (kind, width) cells.
  ##
  ## - a name repeated inside one backend fails at the second cell
  ## - a name two backends share must denote the same cell on both, msl
  ##   declares `char`/`long` where cuda declares `signed char`/`long long`
  ## - bool cells are exempt, OpenCL declares bool locals as `int` to match
  ##   the 4-byte launch ABI
  ## - a 16-bit float or bf16 cell may share its name with a 16-bit integer
  ##   cell or with the other 16-bit float cell, OpenCL stores fp16 and bf16
  ##   values as 16-bit bit patterns stored in one `ushort`
  for i in 0 ..< all.len:
    for a in 0 ..< all[i].cells.len:
      let ca = all[i].cells[a]
      if ScalarCellAxis[ca.idx].kind == ctBool:
        continue
      for j in i ..< all.len:
        for b in 0 ..< all[j].cells.len:
          if (i == j and b <= a) or all[j].cells[b].name != ca.name or
              ScalarCellAxis[all[j].cells[b].idx].kind == ctBool:
            continue
          let cb = all[j].cells[b]
          if ScalarCellAxis[ca.idx].width == 16 and ScalarCellAxis[cb.idx].width == 16 and
              ({ScalarCellAxis[ca.idx].kind, ScalarCellAxis[cb.idx].kind} -
               {ctFloat, ctBf16, ctInt, ctUint} == {}):
            continue
          if i == j:
            error("targetVocab: name '" & ca.name & "' denotes two cells of " & all[i].display & ", " & cellKey(ca.idx) & " and " & cellKey(cb.idx), cb.node)
          elif cb.idx != ca.idx:
            error("targetVocab: name '" & ca.name & "' denotes " & cellKey(ca.idx) & " on " & all[i].display & " but " & cellKey(cb.idx) & " on " & all[j].display, cb.node)

proc cellCtor(idx: int, name: string): NimNode {.compileTime.} =
  let axis = ScalarCellAxis[idx]
  nnkObjConstr.newTree(ident"ScalarCell", nnkExprColonExpr.newTree(ident"kind", ident $axis.kind),
    nnkExprColonExpr.newTree(ident"width", newLit axis.width),
    nnkExprColonExpr.newTree(ident"name", newLit name))

# ═══════════════════════════════════════════════════════════════════════
#  The lit section parser
# ═══════════════════════════════════════════════════════════════════════

const genLitBoolFields = ["narrowCast"]

const genLitUnknownField = "\x00"
  ## Marks a lit field outside the schema.

## (field, constructor section, section type) rows for `litCtor`.
## The "" section collects the table's own fields.
const genLitFields = [
  ("u64Prefix", "int", "IntLitTokens"), ("u64Suffix", "int", "IntLitTokens"),
  ("u32Suffix", "int", "IntLitTokens"), ("i64Prefix", "int", "IntLitTokens"),
  ("i64Suffix", "int", "IntLitTokens"), ("narrowCast", "int", "IntLitTokens"),
  ("f16Prefix", "", ""), ("f16Suffix", "", ""), ("f32Suffix", "", ""),
  ("f64Prefix", "", ""), ("f64Suffix", "", ""), ("nilToken", "", ""),
  ("inf16", "nonFinite", "FloatLitTokens"), ("negInf16", "nonFinite", "FloatLitTokens"),
  ("nan16", "nonFinite", "FloatLitTokens"), ("inf32", "nonFinite", "FloatLitTokens"),
  ("negInf32", "nonFinite", "FloatLitTokens"), ("nan32", "nonFinite", "FloatLitTokens"),
  ("inf64", "nonFinite", "FloatLitTokens"), ("negInf64", "nonFinite", "FloatLitTokens"),
  ("nan64", "nonFinite", "FloatLitTokens")]
  ## Lit fields with the constructor section their value splices into.
  ## Each entry names the table field and the type of the constructor nested there.
  ## The "" section is the table's own fields, an unknown field is a compile error.

proc litCtor(entries: seq[NimNode], display: string): NimNode {.compileTime.} =
  var args = @[ident"LitTokens"]
  var top: seq[NimNode]
  var subs: seq[tuple[section, sectType: string, fields: seq[NimNode]]]
  for line in entries:
    if line.kind != nnkCommand or line.len != 2:
      error("targetVocab: expected a 'field value' entry in the lit section of " & display, line)
    let field = $line[0]
    var section = genLitUnknownField
    var sectType = ""
    for f in genLitFields:
      if f[0] == field:
        section = f[1]
        sectType = f[2]
    if section == genLitUnknownField:
      error("targetVocab: unknown field '" & field & "' in the lit section of " & display, line)
    let val = line[1]
    if field in genLitBoolFields:
      if val.kind != nnkIdent or ($val != "true" and $val != "false"):
        error("targetVocab: field '" & field & "' takes true or false", line)
    elif val.kind != nnkStrLit:
      error("targetVocab: field '" & field & "' takes a string literal, got " & $val.kind, line)
    let pair = nnkExprColonExpr.newTree(ident field, val)
    if section == "":
      top.add pair
    else:
      var slot = subs.len
      for i, sub in subs:
        if sub.section == section:
          slot = i
      if slot == subs.len:
        subs.add (section, sectType, @[pair])
      else:
        subs[slot].fields.add pair
  for sub in subs:
    args.add nnkExprColonExpr.newTree(ident sub.section, nnkObjConstr.newTree(@[ident sub.sectType] & sub.fields))
  nnkObjConstr.newTree(args & top)

# ═══════════════════════════════════════════════════════════════════════
#  The bitcast section parser
# ═══════════════════════════════════════════════════════════════════════

proc castKeyCtor(key: NimNode, isSrc: bool, display: string): seq[NimNode] {.compileTime.} =
  if key.kind != nnkIdent:
    error("targetVocab: unknown bitcast rule key in " & display, key)
  let tok = $key
  if tok == "all":
    if isSrc:
      return @[nnkExprColonExpr.newTree(ident"srcAny", ident"true")]
    return @[nnkExprColonExpr.newTree(ident"dstKinds", nnkCurly.newTree(
      nnkInfix.newTree(ident"..",
        nnkCall.newTree(ident"low", ident"CirTypeKind"),
        nnkCall.newTree(ident"high", ident"CirTypeKind"))))]
  if tok == "any":
    let kinds = nnkCurly.newTree(ident $ctInt, ident $ctUint, ident $ctFloat, ident $ctBf16)
    let field = if isSrc: "srcKinds" else: "dstKinds"
    return @[nnkExprColonExpr.newTree(ident field, kinds)]
  let idx = findCellIndex(tok)
  if idx < 0:
    error("targetVocab: unknown cell '" & tok & "' in the bitcast section of " & display, key)
  let axis = ScalarCellAxis[idx]
  let kindField = if isSrc: "srcKinds" else: "dstKinds"
  let widthField = if isSrc: "srcWidths" else: "dstWidths"
  return @[nnkExprColonExpr.newTree(ident kindField, nnkCurly.newTree(ident $axis.kind)),
    nnkExprColonExpr.newTree(ident widthField, nnkCurly.newTree(newLit axis.width))]

proc castRulesCtor(entries: seq[NimNode], display: string): NimNode {.compileTime.} =
  var rules: seq[NimNode]
  for entry in entries:
    if entry.kind != nnkInfix or $entry[0] != "->" or entry.len != 4 or
        entry[3].kind != nnkStmtList:
      error("targetVocab: expected a 'src->dst:' rule in the bitcast section of " & display, entry)
    var fields = castKeyCtor(entry[1], true, display) & castKeyCtor(entry[2], false, display)
    var terminal = 0
    for row in entry[3]:
      if row.kind == nnkIdent and $row == "sameWidth":
        fields.add nnkExprColonExpr.newTree(ident"sameWidth", ident"true")
      elif row.kind == nnkStrLit:
        fields.add nnkExprColonExpr.newTree(ident"conv", row)
        terminal.inc
      elif row.kind == nnkNilLit:
        # no conv field: the object constructor leaves it nil
        terminal.inc
      else:
        error("targetVocab: unknown bitcast rule row in " & display, row)
    if terminal != 1:
      error("targetVocab: bitcast rule takes one conv text or nil in " & display, entry)
    rules.add nnkObjConstr.newTree(@[ident"CastRule"] & fields)
  nnkBracket.newTree(rules)

# ═══════════════════════════════════════════════════════════════════════
#  The macro
# ═══════════════════════════════════════════════════════════════════════

macro targetVocab*(body: untyped) =
  ## Declares every backend's vocabulary in one block and generates
  ## the `TargetVocabs` array keyed by `BackendKind`:
  ##
  ##   targetVocab:
  ##     bkMetal:
  ##       scalar void "void", bool "bool", i8 "char", ...
  ##       lit u64Suffix "ULL", inf32 "INFINITY", ...
  ##       bitcast:
  ##         any->any:
  ##           conv: "as_type<$2>($1)"
  ##           sameWidth
  ##
  ## Section grammars:
  ##
  ## - the scalar rows list `cell "type name"` entries
  ## - the lit rows list `field value` entries
  ## - the bitcast section lists one `src->dst:` rule per row
  ##
  ## Bitcast rules:
  ##
  ## - a key token is a `ScalarCellAxis` cell, `any` covers
  ##   the width-carrying scalar cells, `all` covers any type
  ## - a rule carries one emit text or nil for a refusal, `sameWidth`
  ##   requires equal widths
  ##
  ## Failure contract:
  ##
  ## - a section key must match a BackendKind member, an unknown key
  ##   fails the compile
  ## - each backend declares the scalar and lit sections, a missing
  ##   section fails the compile
  ## - an unknown field, cell or rule shape fails the compile at its line
  ## - one type name denotes one (kind, width) cell across all backends,
  ##   a name that denotes two cells fails the compile naming both backends
  body.expectKind nnkStmtList
  var
    backends: seq[string]
    vocabCells: seq[tuple[display: string, cells: seq[ParsedCell]]]
  var slots: seq[NimNode]
  for section in body:
    if section.kind != nnkCall or section[0].kind != nnkIdent or
        section[1].kind != nnkStmtList:
      error("targetVocab: expected a 'bkX:' section", section)
    let backend = $section[0]
    var known = false
    for member in BackendKind:
      if $member == backend:
        known = true
        break
    if not known:
      error("targetVocab: unknown backend '" & backend & "'", section)
    if backend in backends:
      error("targetVocab: duplicate backend section '" & backend & "'", section)
    backends.add backend
    let display = backend
    var seen: seq[string]
    var fields: seq[NimNode]
    for fam in section[1]:
      if fam[0].kind != nnkIdent or (fam.kind != nnkCommand and
          (fam.kind != nnkCall or fam[1].kind != nnkStmtList)):
        error("targetVocab: expected a '<family> ...' section in " & display, fam)
      let family = $fam[0]
      if family notin ["scalar", "lit", "bitcast", "storageBool"]:
        error("targetVocab: unknown family '" & family & "' in " & display, fam)
      if family in seen:
        error("targetVocab: duplicate family '" & family & "' in " & display, fam)
      seen.add family
      var entries: seq[NimNode]
      if fam.kind == nnkCall:
        for line in fam[1]:
          entries.add line
      else:
        for arg in fam[1 ..^ 1]:
          entries.add arg
      case family
      of "scalar":
        var cells: seq[ParsedCell]
        var seenCells: seq[int]
        for entry in entries:
          let c = parseCellPair(entry, display)
          if c.name.len == 0:
            error("targetVocab: scalar cell '" & ScalarCellAxis[c.idx].tok & "' of " & display & " needs a type name", entry)
          if c.idx in seenCells:
            error("targetVocab: scalar cell '" & ScalarCellAxis[c.idx].tok & "' appears twice in " & display, entry)
          seenCells.add c.idx
          cells.add c
        vocabCells.add (display, cells)
        fields.add nnkExprColonExpr.newTree(ident"scalarNames", nnkObjConstr.newTree(@[ident"ScalarNames",
          nnkExprColonExpr.newTree(ident"cells", nnkCall.newTree(ident"@",
            nnkBracket.newTree(cells.mapIt(cellCtor(it.idx, it.name)))))]))
      of "lit":
        fields.add nnkExprColonExpr.newTree(ident"lit", litCtor(entries, display))
      of "bitcast":
        fields.add nnkExprColonExpr.newTree(ident"castRules", nnkCall.newTree(ident"@",
          castRulesCtor(entries, display)))
      of "storageBool":
        if entries.len != 1 or entries[0].kind != nnkStrLit:
          error("targetVocab: storageBool takes one string value in " & display, section)
        fields.add nnkExprColonExpr.newTree(ident"storageBool", entries[0])
    for required in ["scalar", "lit"]:
      if required notin seen:
        error("targetVocab: backend " & display & " is missing the '" & required & "' section", section)
    slots.add nnkObjConstr.newTree(@[ident"TargetVocab"] & fields)
  checkCellNames(vocabCells)
  # Array slots fill in declaration order, one per BackendKind member
  var elems: seq[NimNode]
  for member in BackendKind:
    var slot = nnkObjConstr.newTree(ident"TargetVocab")
    for i, b in backends:
      if $member == b:
        slot = slots[i]
        break
    elems.add slot
  result = nnkStmtList.newTree(nnkConstSection.newTree(nnkConstDef.newTree(
    nnkPostfix.newTree(ident"*", ident"TargetVocabs"),
    nnkBracketExpr.newTree(ident"array", ident"BackendKind", ident"TargetVocab"),
    nnkBracket.newTree(elems))))

# ═══════════════════════════════════════════════════════════════════════
#  Schema
# ═══════════════════════════════════════════════════════════════════════

type
  FloatLitTokens* = object
    ## Non-finite float literal tokens for one printer target, one field per literal width.
    ## The printer emits a token only for a declared width, other widths fall through the scalar table.

    inf16*, negInf16*, nan16*: string
    inf32*, negInf32*, nan32*: string
    inf64*, negInf64*, nan64*: string

  IntLitTokens* = object
    ## Integer literal token parts, the printer composes `prefix & $value & suffix`.
    ##
    ## - 64-bit widths take `u64Prefix`/`u64Suffix` or `i64Prefix`/`i64Suffix`
    ## - 32-bit unsigned takes `u32Suffix`, 32-bit signed takes none
    ## - the printer emits narrow (8/16-bit) widths unadorned or, per
    ##   `narrowCast`, as a cast to the scalar type name
    u64Prefix*, u64Suffix*, u32Suffix*: string
    i64Prefix*, i64Suffix*: string
    narrowCast*: bool

  ScalarCell* = object
    ## One (kind, width) cell of a target's scalar names.
    kind*: CirTypeKind
    width*: int
    ## Bit width, 0 for the widthless kinds (void, bool).
    name*: string
    ## Type name the target prints.

  ScalarNames* = object
    ## Scalar type names of one target by (kind, width) cell.
    ## The table carries one entry per named cell, the lookup fails on a missing cell and the target compiler rejects the emitted text.
    cells*: seq[ScalarCell]

  LitTokens* = object
    ## Literal token forms of one target.
    ## No exponent or hex-float fields: no printer handles either form.
    int*: IntLitTokens
    f16Prefix*, f16Suffix*: string
    f32Suffix*: string
    f64Prefix*, f64Suffix*: string
    nilToken*: string
    nonFinite*: FloatLitTokens

  CastWidth* = range[0..255]
  ## Bit width a cast rule matches.

  CastRule* = object
    ## One bit-reinterpret rule, the printer matches the rules first to last.
    ##
    ## - `srcAny` matches any source type, otherwise the source type must
    ##   carry a `srcKinds` kind and one of the `srcWidths` widths when set
    ## - the destination type must carry a `dstKinds` kind and, when set,
    ##   a `dstWidths` width, `sameWidth` also requires equal
    ##   source and destination widths
    ## - in `conv` the printer substitutes `$1` with the printed operand
    ##   and `$2` with the destination type name
    ## - an unset `conv` (nil) marks a refusal, the printer then fails
    ##   with the shared no-bit-cast message naming the destination type
    srcAny*: bool
    srcKinds*, dstKinds*: set[CirTypeKind]
    srcWidths*, dstWidths*: set[CastWidth]
    sameWidth*: bool
    conv*: string

  TargetVocab* = object
    ## One backend's static vocabulary as the shared emission reads it.
    scalarNames*: ScalarNames
    lit*: LitTokens
    castRules*: seq[CastRule]
    storageBool*: string
    ## Storage text of a bool scalar, "" when the target has none.


# ═══════════════════════════════════════════════════════════════════════
#  The target vocabulary: types, literal forms, cast rules per backend
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  The cell axis
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  Family table and field schemas
# ═══════════════════════════════════════════════════════════════════════

const genFamilies = [
  ("kind", "WalkPolicy", true), ("math", "MathBuiltinNames", false),
  ("vector", "VectorRule", true)]
  ## (family, lookup name suffix, required per target) rows.

const genBoolFields = ["braceSameLine", "foldsIteElse", "emitOperandRendersExpr"]

const unknownField = "\x00"
  ## Marks a kind field outside the schema.

## (family, field, constructor section, section type) rows for `litKindCtor`,
## the "" section collects the table's own fields.
const genFieldSections = [
  ("kind", "unrollDirective", "loop", "LoopStyle"), ("kind", "braceSameLine", "loop", "LoopStyle"),
  ("kind", "foldsIteElse", "", ""), ("kind", "emitOperandRendersExpr", "", ""),
  ("kind", "targetExprKinds", "", "")]

proc getFamilySpec(family: string): tuple[suffix: string, required: bool] {.compileTime.} =
  for f in genFamilies:
    if f[0] == family:
      return (f[1], f[2])

# ═══════════════════════════════════════════════════════════════════════
#  Section parsers
# ═══════════════════════════════════════════════════════════════════════

proc parseSectionKey(n: NimNode): tuple[target, flavor: string] {.compileTime.} =
  if n.kind == nnkIdent:
    result = ($n, "")
  elif n.kind == nnkCall and n.len == 2 and n[0].kind == nnkIdent and n[1].kind == nnkIdent:
    result = ($n[0], $n[1])
  else:
    error("genTargets: expected a 'target <name>:' or 'target <name>(<flavor>):' section header", n)

proc getKeyNames(key: tuple[target, flavor: string]): tuple[display, suffix: string] {.compileTime.} =
  if key.flavor == "":
    result = (key.target, key.target)
  else:
    result = (key.target & "(" & key.flavor & ")",
      key.target & key.flavor[0].toUpperAscii & key.flavor[1 ..^ 1])

proc colon(name: string, val: NimNode): NimNode {.compileTime.} =
  nnkExprColonExpr.newTree(ident name, val)

proc litKindCtor(entries: seq[NimNode], family, suffix, display: string): NimNode {.compileTime.} =
  var args = @[ident suffix]
  var top: seq[NimNode]
  var subs: seq[tuple[section, sectType: string, fields: seq[NimNode]]]
  for line in entries:
    if line.kind != nnkCommand or line.len != 2:
      error("genTargets: expected a 'field value' entry in the " & family & " section of " & display, line)
    let field = $line[0]
    var section = unknownField
    var sectType = ""
    for f in genFieldSections:
      if f[0] == family and f[1] == field:
        section = f[2]
        sectType = f[3]
    if section == unknownField:
      error("genTargets: unknown field '" & field & "' in the " & family & " section of " & display, line)
    let val = line[1]
    var want: string
    var ok = false
    if field in genBoolFields:
      want = "true or false"
      ok = val.kind == nnkIdent and ($val == "true" or $val == "false")
    elif field == "targetExprKinds":
      want = "a set of expression kinds"
      ok = val.kind == nnkCurly
      if ok:
        for e in val:
          ok = ok and e.kind == nnkIdent
    else:
      want = "a string literal"
      ok = val.kind == nnkStrLit
    if not ok:
      error("genTargets: field '" & field & "' takes " & want, line)
    let pair = colon(field, val)
    if section == "":
      top.add pair
    else:
      var slot = subs.len
      for i, sub in subs:
        if sub.section == section:
          slot = i
      if slot == subs.len:
        subs.add (section, sectType, @[pair])
      else:
        subs[slot].fields.add pair
  for sub in subs:
    args.add colon(sub.section, nnkObjConstr.newTree(@[ident sub.sectType] & sub.fields))
  nnkObjConstr.newTree(args & top)

proc builtinTableCtor(entries: seq[NimNode], family, display: string): NimNode {.compileTime.} =
  var names: seq[string]
  var pairs: seq[NimNode]
  for line in entries:
    # a math entry nests its `width "text"` pair under the name
    let ok = line.kind == nnkCommand and line.len == 2 and line[0].kind == nnkIdent and
      line[1].kind == nnkCommand and line[1].len == 2 and
      line[1][0].kind == nnkIntLit and line[1][1].kind == nnkStrLit
    if not ok:
      error("genTargets: expected a 'builtin width \"text\"' entry in the " & family & " section of " & display, line)
    let name = $line[0]
    let key = name & ":" & $line[1][0].intVal
    if key in names:
      error("genTargets: builtin '" & key & "' appears twice in the " & family & " section of " & display, line)
    names.add key
    pairs.add nnkExprColonExpr.newTree(nnkPar.newTree(newLit name, newLit line[1][0].intVal.int), line[1][1])
  # generated `{key: value, ...}.toTable()` expression
  nnkCall.newTree(nnkDotExpr.newTree(nnkTableConstr.newTree(pairs), ident"toTable"))

proc vectorCtor(entries: seq[NimNode], display: string): NimNode {.compileTime.} =
  var elems: seq[NimNode]
  var lanes: seq[NimNode]
  var nameForm = ""
  for entry in entries:
    if entry.kind == nnkTupleConstr and entry.len == 0:
      continue
    if entry.kind != nnkCommand or entry.len < 2:
      error("genTargets: expected an 'elems' / 'lanes' / 'nameForm' entry in the vector section of " & display, entry)
    case $entry[0]
    of "elems":
      for arg in entry[1 ..^ 1]:
        let c = parseCellPair(arg, display)
        let axis = ScalarCellAxis[c.idx]
        elems.add nnkPar.newTree(ident $axis.kind, newLit axis.width, newLit c.name)
    of "lanes":
      for arg in entry[1 ..^ 1]:
        if arg.kind != nnkIntLit:
          error("genTargets: lane counts are integers in the vector section of " & display, arg)
        lanes.add newLit arg.intVal.int
    of "nameForm":
      if entry.len != 2 or entry[1].kind != nnkIdent or
          ($entry[1] != "vnfPrefixed" and $entry[1] != "vnfGenericElem"):
        error("genTargets: nameForm takes vnfPrefixed or vnfGenericElem", entry)
      nameForm = $entry[1]
    else:
      error("genTargets: unknown field '" & $entry[0] & "' in the vector section of " & display, entry)
  var args = @[ident"VectorRule"]
  if elems.len > 0:
    args.add colon("elems", nnkCall.newTree(ident"@", nnkBracket.newTree(elems)))
  if lanes.len > 0:
    args.add colon("lanes", nnkCall.newTree(ident"@", nnkBracket.newTree(lanes)))
  if nameForm != "":
    args.add colon("nameForm", ident nameForm)
  nnkObjConstr.newTree(args)

# ═══════════════════════════════════════════════════════════════════════
#  The macro
# ═══════════════════════════════════════════════════════════════════════

macro genTargets*(body: untyped) =
  ## Registers every target's sections of one module and generates one
  ## lookup const per (target, family):
  ##
  ##   genTargets:
  ##     target msl:
  ##       kind unrollDirective "#pragma unroll", targetExprKinds {eNil}
  ##       math sin 32 "sin($1)"
  ##       vector elems f32 "float", lanes 2, 3, 4, nameForm vnfPrefixed
  ##
  ## - the kind rows list `field value` entries, string fields take
  ##   a string literal, bool fields take true or false,
  ##   `targetExprKinds` takes a set of expression kinds
  ## - the math rows list `builtin width "text"` entries keyed by name
  ##   and bit width, `$1` stands for the argument list
  ## - the vector rows list `elems`, `lanes` and an optional `nameForm`,
  ##   the `elems` entries take `cell "prefix"`, a vectorless target
  ##   lists `vector ()`
  ##
  ## Failure contract:
  ##
  ## - an unknown family, field, cell or value shape fails the compile
  ##   on its own line
  ## - a duplicate target or family section fails the compile
  ## - a target that misses a required family fails the compile
  ## - a type name that denotes two (kind, width) cells fails the compile
  ##   naming both targets
  body.expectKind nnkStmtList
  var
    keys: seq[tuple[target, flavor: string]]
  result = nnkStmtList.newTree()
  for section in body:
    if section.kind != nnkCommand or section.len != 3 or $section[0] != "target":
      error("genTargets: expected a 'target <name>:' section", section)
    let key = parseSectionKey(section[1])
    if key in keys:
      error("genTargets: duplicate target section '" & getKeyNames(key).display & "'", section)
    keys.add key
    let display = getKeyNames(key).display
    var seen: seq[string]
    for fam in section[2]:
      if fam[0].kind != nnkIdent or (fam.kind != nnkCommand and
          (fam.kind != nnkCall or fam[1].kind != nnkStmtList)):
        error("genTargets: expected a '<family> ...' section in target " & display, fam)
      let family = $fam[0]
      let suffix = getFamilySpec(family).suffix
      if suffix == "":
        error("genTargets: unknown family '" & family & "' in target " & display, fam)
      if family in seen:
        error("genTargets: duplicate family '" & family & "' in target " & display, fam)
      seen.add family
      var entries: seq[NimNode]
      if fam.kind == nnkCall:
        for line in fam[1]:
          entries.add line
      else:
        for arg in fam[1 ..^ 1]:
          entries.add arg
      var ctor: NimNode
      case family
      of "kind":
        ctor = litKindCtor(entries, family, suffix, display)
      of "math":
        ctor = builtinTableCtor(entries, family, display)
      of "vector":
        ctor = vectorCtor(entries, display)
      else:
        discard
      result.add nnkConstSection.newTree(nnkConstDef.newTree(
        nnkPostfix.newTree(ident"*", ident(getKeyNames(key).suffix & suffix)),
        newEmptyNode(), ctor))
    for f in genFamilies:
      if f[2] and f[0] notin seen:
        error("genTargets: target " & display & " is missing family '" & f[0] & "'", section)
