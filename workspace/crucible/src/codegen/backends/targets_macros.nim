# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## `genTargets`: the declarative target-table macro.
##
## One invocation carries every target's sections in the block DSL and generates
## one lookup const per (target, flavor) key and family.
## Every failure is a compile error at the section definition:
##
## - a target key that misses a required family, an unknown field, cell
##   or value shape: each fails the compile at its line, naming the family
##   at the target's section
## - a type name that denotes two cells fails the compile naming both targets
## - a scalar section lists any subset of the `ScalarCellAxis` cells, an omitted
##   message declared in the section
## - one invocation registers every target of its module, generated names
##   would collide across invocations
##
## Each family's entry grammar is the doc of its parse proc below.
## `targets_registry.nim` declares every family of all six targets.

import std/macros
import std/sequtils
import std/strutils
import ../ir/cirv2_lang
import ./targets_registry

const genFamilies* = [
  ("scalar", "ScalarNames", true), ("lit", "LitTokens", true), ("kind", "WalkPolicy", true),
  ("math", "MathBuiltinNames", false),
  ("vector", "VectorRule", true),
  ("bitcast", "CastRules", true)]
  ## Eight families. Each entry names the DSL family, gives the generated
  ## lookup's name suffix and marks whether every target key must carry it.
  ## Optional families (`conv`, `math`) are declared only by targets
  ## whose printers read them. One lookup const per (target, flavor) key.

const genBoolFields = ["narrowCast", "braceSameLine", "foldsIteElse", "emitOperandRendersExpr"]
  ## Lit and kind fields whose value is `true` or `false`.

const unknownField = "\x00"
  ## `litKindCtor`'s section for a field the family's schema does not name.

const genFieldSections = [
  ("lit", "u64Prefix", "int", "IntLitTokens"), ("lit", "u64Suffix", "int", "IntLitTokens"),
  ("lit", "u32Suffix", "int", "IntLitTokens"), ("lit", "i64Prefix", "int", "IntLitTokens"),
  ("lit", "i64Suffix", "int", "IntLitTokens"), ("lit", "narrowCast", "int", "IntLitTokens"),
  ("lit", "f16Prefix", "", ""), ("lit", "f16Suffix", "", ""), ("lit", "f32Suffix", "", ""),
  ("lit", "f64Prefix", "", ""), ("lit", "f64Suffix", "", ""), ("lit", "nilToken", "", ""),
  ("lit", "inf16", "nonFinite", "FloatLitTokens"), ("lit", "negInf16", "nonFinite", "FloatLitTokens"),
  ("lit", "nan16", "nonFinite", "FloatLitTokens"), ("lit", "inf32", "nonFinite", "FloatLitTokens"),
  ("lit", "negInf32", "nonFinite", "FloatLitTokens"), ("lit", "nan32", "nonFinite", "FloatLitTokens"),
  ("lit", "inf64", "nonFinite", "FloatLitTokens"), ("lit", "negInf64", "nonFinite", "FloatLitTokens"),
  ("lit", "nan64", "nonFinite", "FloatLitTokens"),
  ("kind", "unrollDirective", "loop", "LoopStyle"), ("kind", "braceSameLine", "loop", "LoopStyle"),
  ("kind", "foldsIteElse", "", ""), ("kind", "emitOperandRendersExpr", "", ""),
  ("kind", "targetExprKinds", "", "")]
  ## Lit and kind fields with the constructor section their value splices into.
  ## Each entry names the section field in the table object and the type of the constructor nested there.
  ## The "" section is the table's own fields, an unknown field is a compile error.

proc getFamilySpec(family: string): tuple[suffix: string, required: bool] {.compileTime.} =
  ## Table entry of one DSL family, ("", false) for an unknown family.
  for f in genFamilies:
    if f[0] == family:
      return (f[1], f[2])

proc findCellIndex(tok: string): int {.compileTime.} =
  ## Index of a cell token in `ScalarCellAxis`, -1 when unknown.
  for i, axis in ScalarCellAxis:
    if axis.tok == tok:
      return i
  return -1

proc parseSectionKey(n: NimNode): tuple[target, flavor: string] {.compileTime.} =
  ## Target key of one section header, `target <name>:` or `target <name>(<flavor>):`.
  if n.kind == nnkIdent:
    result = ($n, "")
  elif n.kind == nnkCall and n.len == 2 and n[0].kind == nnkIdent and n[1].kind == nnkIdent:
    result = ($n[0], $n[1])
  else:
    error("genTargets: expected a 'target <name>:' or 'target <name>(<flavor>):' section header", n)

proc getKeyNames(key: tuple[target, flavor: string]): tuple[display, suffix: string] {.compileTime.} =
  ## Two renderings of one target key. `display` names the key in check messages, `suffix` prefixes the generated lookup names.
  if key.flavor == "":
    result = (key.target, key.target)
  else:
    result = (key.target & "(" & key.flavor & ")",
      key.target & key.flavor[0].toUpperAscii & key.flavor[1 ..^ 1])

proc colon(name: string, val: NimNode): NimNode {.compileTime.} =
  ## One `name: val` field of an object constructor.
  nnkExprColonExpr.newTree(ident name, val)

type
  ParsedCell = tuple[idx: int, spelling: string, node: NimNode]
    ## One parsed cell: its `ScalarCellAxis` index, type name, and the node
    ## the name checks anchor.

proc parseCellPair(arg: NimNode, display: string): ParsedCell {.compileTime.} =
  ## One `cell "name"` entry, also the vector-element form:
  ## a cell without a prefix states the empty prefix there.
  if arg.kind == nnkIdent:
    result = (findCellIndex($arg), "", arg)
  elif arg.kind == nnkCommand and arg.len == 2 and arg[1].kind == nnkStrLit:
    result = (findCellIndex($arg[0]), arg[1].strVal, arg[1])
  else:
    error("genTargets: expected a 'cell \"name\"' entry in " & display, arg)
  if result.idx < 0:
    error("genTargets: unknown cell '" & (if arg.kind == nnkIdent: $arg else: $arg[0]) & "' in " & display, arg)

proc cellKey(idx: int): string {.compileTime.} =
  ## One cell's "(kind, width)" form in check messages.
  "(" & $ScalarCellAxis[idx].kind & ", " & $ScalarCellAxis[idx].width & ")"

proc checkCellNames(all: seq[tuple[display: string, cells: seq[ParsedCell]]]) {.compileTime.} =
  ## A type name denotes one (kind, width) cell.
  ##
  ## - within a target a repeated name fails at the second cell
  ## - a name two targets share must denote the same cell on both: msl
  ##   declares `char`/`long` where cuda declares `signed char`/`long long`,
  ##   the data is explicit per target
  ## - bool cells are exempt: OpenCL prints bool locals as `int` to match
  ##   the 4-byte launch ABI
  ## - a 16-bit float or bf16 cell may share its name with a 16-bit integer
  ##   cell or with the other 16-bit float cell
  ##   OpenCL stores fp16/bf16 values as 16-bit bit patterns in a `ushort`,
  ##   the shape v1 emitted
  for i in 0 ..< all.len:
    for a in 0 ..< all[i].cells.len:
      let ca = all[i].cells[a]
      if ScalarCellAxis[ca.idx].kind == ctBool:
        continue
      for j in i ..< all.len:
        for b in 0 ..< all[j].cells.len:
          if (i == j and b <= a) or all[j].cells[b].spelling != ca.spelling or
              ScalarCellAxis[all[j].cells[b].idx].kind == ctBool:
            continue
          let cb = all[j].cells[b]
          if ScalarCellAxis[ca.idx].width == 16 and ScalarCellAxis[cb.idx].width == 16 and
              ({ScalarCellAxis[ca.idx].kind, ScalarCellAxis[cb.idx].kind} -
               {ctFloat, ctBf16, ctInt, ctUint} == {}):
            continue
          if i == j:
            error("genTargets: name '" & ca.spelling & "' denotes two cells of " & all[i].display & ", " & cellKey(ca.idx) & " and " & cellKey(cb.idx), cb.node)
          elif cb.idx != ca.idx:
            error("genTargets: name '" & ca.spelling & "' denotes " & cellKey(ca.idx) & " on " & all[i].display & " but " & cellKey(cb.idx) & " on " & all[j].display, cb.node)

proc cellCtor(idx: int, spelling: string): NimNode {.compileTime.} =
  ## One ScalarCell constructor from a `ScalarCellAxis` entry.
  let axis = ScalarCellAxis[idx]
  nnkObjConstr.newTree(ident"ScalarCell", colon("kind", ident $axis.kind),
    colon("width", newLit axis.width), colon("spelling", newLit spelling))

proc litKindCtor(entries: seq[NimNode], family, suffix, display: string): NimNode {.compileTime.} =
  ## Constructor of the lit or kind table over its validated fields.
  ##
  ## - string fields take a string literal, bool fields `true` or `false`,
  ##   `targetExprKinds` a set of expression kinds
  ## - a nested-section field (the table above) splices into a constructor
  ##   of that section's type, a table field takes the generated
  ##   constructor's top level
  ## - a bad member name fails the compile at the section: value nodes splice
  ##   into the generated constructor at their original position
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
    if field in genBoolFields:
      if val.kind != nnkIdent or ($val != "true" and $val != "false"):
        error("genTargets: field '" & field & "' takes true or false", line)
    elif field == "targetExprKinds":
      if val.kind != nnkCurly:
        error("genTargets: field '" & field & "' takes a set of expression kinds, got " & $val.kind, line)
      for e in val:
        if e.kind != nnkIdent:
          error("genTargets: field '" & field & "' takes expression kinds, got " & $e.kind, e)
    elif val.kind != nnkStrLit:
      error("genTargets: field '" & field & "' takes a string literal, got " & $val.kind, line)
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
  ## Builtin name table of a math section (`builtin width "text"`).
  ##
  ## - a math key is the (name, bit width) pair, `$1` the argument list
  var names: seq[string]
  var pairs: seq[NimNode]
  for line in entries:
    # a math entry nests its `width "spelling"` pair under the name
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
  ## Vector coverage rule of a vector section:
  ##
  ## - `elems` entries of `cell "prefix"` elements, a cell with no prefix an empty prefix
  ## - one `lanes n, ...` entry, an optional `nameForm` entry
  ## - a target with no vectors lists `vector ()`
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
        elems.add nnkPar.newTree(ident $axis.kind, newLit axis.width, newLit c.spelling)
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

proc castRulesCtor(entries: seq[NimNode], display: string): NimNode {.compileTime.} =
  ## Bit-reinterpret coverage rules of a bitcast section, one `rule` entry per
  ## rule:
  ##
  ## - the rule's fields in schema names: `srcAny`, `srcKinds`, `srcWidths`,
  ##   `dstKinds`, `dstWidths`, `sameWidth`, kind sets over the `ctVoid`/
  ##   `ctBool`/`ctInt`/`ctUint`/`ctFloat`/`ctBf16` names, with the every-kind
  ##   destination set as `dstKinds {low(CirTypeKind)..high(CirTypeKind)}`
  ## - then one format text or `refusal "text"`
  ## - set and text nodes splice into the generated constructor unchanged,
  ##   so a bad member name fails the compile at the section
  var rules: seq[NimNode]
  for entry in entries:
    if entry.kind != nnkCommand or entry.len < 2 or $entry[0] != "rule":
      error("genTargets: expected a 'rule ...' entry in the bitcast section of " & display, entry)
    var fields: seq[NimNode]
    var dst = false
    var terminal = 0
    for arg in entry[1 ..^ 1]:
      if arg.kind == nnkStrLit:
        fields.add colon("spelling", arg)
        terminal.inc
        continue
      let head = if arg.kind == nnkCommand: $arg[0] else: $arg
      case head
      of "srcAny", "sameWidth":
        if arg.kind != nnkIdent:
          error("genTargets: '" & head & "' takes no value in a bitcast rule of " & display, arg)
        fields.add colon(head, ident"true")
      of "srcKinds", "dstKinds", "srcWidths", "dstWidths":
        if arg.kind != nnkCommand or arg.len != 2 or arg[1].kind != nnkCurly or arg[1].len == 0:
          error("genTargets: '" & head & "' takes a non-empty set in a bitcast rule of " & display, arg)
        if head == "dstKinds":
          dst = true
        fields.add colon(head, arg[1])
      of "refusal":
        if arg.kind != nnkCommand or arg.len != 2 or arg[1].kind != nnkStrLit:
          error("genTargets: 'refusal' takes a refusal text in a bitcast rule of " & display, arg)
        fields.add colon("refusal", arg[1])
        terminal.inc
      else:
        error("genTargets: unknown rule element '" & head & "' in a bitcast rule of " & display, arg)
    if not dst:
      error("genTargets: a bitcast rule needs 'dstKinds {kinds}' in " & display, entry)
    if terminal != 1:
      error("genTargets: a bitcast rule takes one spelling or one refusal text in " & display, entry)
    rules.add nnkObjConstr.newTree(@[ident"CastRule"] & fields)
  nnkBracket.newTree(rules)

macro genTargets*(body: untyped) =
  ## Registers every target's sections of one module in the block DSL.
  ## See the module header for the failure contract.
  body.expectKind nnkStmtList
  var
    keys: seq[tuple[target, flavor: string]]
    scalarCells: seq[tuple[display: string, cells: seq[ParsedCell]]]
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
      of "scalar":
        var cells: seq[ParsedCell]
        var seenCells: seq[int]
        for entry in entries:
          let c = parseCellPair(entry, display)
          if c.spelling.len == 0:
            error("genTargets: scalar cell '" & ScalarCellAxis[c.idx].tok & "' of " & display & " needs a type name", entry)
          if c.idx in seenCells:
            error("genTargets: scalar cell '" & ScalarCellAxis[c.idx].tok & "' appears twice in " & display, entry)
          seenCells.add c.idx
          cells.add c
        scalarCells.add (display, cells)
        var ctorArgs = @[ident"ScalarNames", colon("cells", nnkCall.newTree(ident"@",
          nnkBracket.newTree(cells.mapIt(cellCtor(it.idx, it.spelling)))))]
        ctor = nnkObjConstr.newTree(ctorArgs)
      of "lit", "kind":
        ctor = litKindCtor(entries, family, suffix, display)
      of "conv", "math":
        ctor = builtinTableCtor(entries, family, display)
      of "vector":
        ctor = vectorCtor(entries, display)
      of "bitcast":
        ctor = castRulesCtor(entries, display)
      else:
        discard
      result.add nnkConstSection.newTree(nnkConstDef.newTree(
        nnkPostfix.newTree(ident"*", ident(getKeyNames(key).suffix & suffix)),
        newEmptyNode(), ctor))
    for f in genFamilies:
      if f[2] and f[0] notin seen:
        error("genTargets: target " & display & " is missing family '" & f[0] & "'", section)
  checkCellNames(scalarCells)
