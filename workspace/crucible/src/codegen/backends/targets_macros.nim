# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## `genTargets`: the declarative target-row macro.
##
## One invocation carries every target's rows in the block DSL and generates one
## lookup const per (target, flavor) key and row family. Validation is compile
## time and every failure is a compile error at the row definition:
##
## - a target key that misses a required family fails the compile, naming
##   the family at the target's section
## - an unknown field, cell or value shape fails the compile at its line
## - a type name that denotes two cells fails the compile naming both targets
## - a scalar row lists any subset of the `ScalarCellAxis` cells, an omitted
##   cell refuses at the lookup with its one-line `refused <cell> "reason"`
##   message stated in the row
## - one invocation registers every target of its module, generated names
##   would collide across invocations
##
## Each family's entry grammar is the doc of its parse proc below.
## `targets_registry.nim` declares every family of all six targets.

import std/macros
import std/strutils
import ../ir/cirv2_lang
import ./targets_registry

const genFamilies* = [
  ("scalar", "ScalarRow", true), ("lit", "LitRow", true), ("kind", "KindRow", true),
  ("conv", "ConvBuiltinNames", false), ("math", "MathBuiltinNames", false),
  ("refusals", "ConvFpRefusals", false), ("vector", "VectorRule", true),
  ("bitcast", "CastRules", true)]
  ## Eight row families. Each entry names the DSL family, gives the generated
  ## lookup's name suffix and states whether every target key must carry it.
  ## Optional families (`conv`, `math`, `refusals`) are stated only by targets
  ## whose printers read them. One lookup const per (target, flavor) key.

const genBoolFields = ["narrowCast", "braceSameLine", "foldsIteElse", "emitOperandRendersExpr"]
  ## Lit and kind fields whose value is `true` or `false`.

const unknownField = "\x00"
  ## `fieldSection` result for a field the family's schema does not name.

proc familyRow(family: string): tuple[suffix: string, required: bool] {.compileTime.} =
  ## Table row of one DSL family, ("", false) for an unknown family.
  for f in genFamilies:
    if f[0] == family:
      return (f[1], f[2])

proc cellIndex(tok: string): int {.compileTime.} =
  ## Index of a cell token in `ScalarCellAxis`, -1 when unknown.
  for i, axis in ScalarCellAxis:
    if axis.tok == tok:
      return i
  return -1

proc sectionKey(n: NimNode): tuple[target, flavor: string] {.compileTime.} =
  ## Target key of one section header, `target <name>:` or `target <name>(<flavor>):`.
  if n.kind == nnkIdent:
    result = ($n, "")
  elif n.kind == nnkCall and n.len == 2 and n[0].kind == nnkIdent and n[1].kind == nnkIdent:
    result = ($n[0], $n[1])
  else:
    error("genTargets: expected a 'target <name>:' or 'target <name>(<flavor>):' section header", n)

proc keyDisplay(key: tuple[target, flavor: string]): string {.compileTime.} =
  ## Name of one target key in check messages, `target(flavor)` for a flavor key.
  if key.flavor == "":
    key.target
  else:
    key.target & "(" & key.flavor & ")"

proc keySuffix(key: tuple[target, flavor: string]): string {.compileTime.} =
  ## Prefix a target key contributes to generated lookup names.
  if key.flavor == "":
    key.target
  else:
    key.target & key.flavor[0].toUpperAscii & key.flavor[1 ..^ 1]

const genFieldSections = [
  ("lit", "u64Prefix", "int"), ("lit", "u64Suffix", "int"), ("lit", "u32Suffix", "int"),
  ("lit", "i64Prefix", "int"), ("lit", "i64Suffix", "int"), ("lit", "narrowCast", "int"),
  ("lit", "f16Prefix", ""), ("lit", "f16Suffix", ""), ("lit", "f32Suffix", ""),
  ("lit", "f64Prefix", ""), ("lit", "f64Suffix", ""), ("lit", "nilToken", ""),
  ("lit", "inf16", "nonFinite"), ("lit", "negInf16", "nonFinite"), ("lit", "nan16", "nonFinite"),
  ("lit", "inf32", "nonFinite"), ("lit", "negInf32", "nonFinite"), ("lit", "nan32", "nonFinite"),
  ("lit", "inf64", "nonFinite"), ("lit", "negInf64", "nonFinite"), ("lit", "nan64", "nonFinite"),
  ("kind", "unrollDirective", "loop"), ("kind", "braceSameLine", "loop"),
  ("kind", "foldsIteElse", ""), ("kind", "emitOperandRendersExpr", ""),
  ("kind", "targetExprKinds", "")]
  ## Lit and kind fields with their constructor section. "" is the row's
  ## own fields. A field outside the table is a compile error.

const genSectionTypes = [("int", "IntLitTokens"), ("nonFinite", "FloatLitTokens"),
  ("loop", "LoopStyle")]
  ## Schema type of each nested constructor section.

proc fieldSection(family, field: string): string {.compileTime.} =
  ## Constructor section of one lit/kind field, `unknownField` for a field
  ## the family's schema does not name.
  for f in genFieldSections:
    if f[0] == family and f[1] == field:
      return f[2]
  return "\x00"

proc sectionType(section: string): string {.compileTime.} =
  ## Schema type of one nested constructor section.
  for t in genSectionTypes:
    if t[0] == section:
      return t[1]
  return ""

proc colon(name: string, val: NimNode): NimNode {.compileTime.} =
  ## One `name: val` field of an object constructor.
  nnkExprColonExpr.newTree(ident name, val)

proc objCtor(typeName: string, pairs: seq[tuple[section: string, pair: NimNode]]): NimNode {.compileTime.} =
  ## Object constructor over validated field pairs. Top-level fields come
  ## from the "" section, nested sections nest a constructor of their type.
  var args = @[ident typeName]
  var top: seq[NimNode]
  var sections: seq[string]
  for (section, pair) in pairs:
    if section == "":
      top.add pair
    elif section notin sections:
      sections.add section
  for section in sections:
    var sub: seq[NimNode]
    for (s, pair) in pairs:
      if s == section:
        sub.add pair
    args.add colon(section, nnkObjConstr.newTree(@[ident sectionType(section)] & sub))
  nnkObjConstr.newTree(args & top)

proc famEntries(fam: NimNode): seq[NimNode] {.compileTime.} =
  ## Entries of one family row: the arguments of a `family ...` line
  ## (comma-separated), or the block lines of the multi-line families.
  ## a `cell` / `"name"` pair parses as a nested command.
  if fam.kind == nnkCall:
    for line in fam[1]:
      result.add line
  else:
    for arg in fam[1 ..^ 1]:
      result.add arg

type
  ParsedCell = tuple[idx: int, spelling: string, node: NimNode]
    ## One parsed cell: its `ScalarCellAxis` index, type name, and the node
    ## the name checks anchor.

proc cellPair(arg: NimNode, display: string): ParsedCell {.compileTime.} =
  ## One `cell "name"` entry, also the vector-element form: a bare cell
  ## states an empty prefix there.
  if arg.kind == nnkIdent:
    result = (cellIndex($arg), "", arg)
  elif arg.kind == nnkCommand and arg.len == 2 and arg[1].kind == nnkStrLit:
    result = (cellIndex($arg[0]), arg[1].strVal, arg[1])
  else:
    error("genTargets: expected a 'cell \"name\"' entry in " & display, arg)
  if result.idx < 0:
    error("genTargets: unknown cell '" &
      (if arg.kind == nnkIdent: $arg else: $arg[0]) & "' in " & display, arg)

proc cellKey(idx: int): string {.compileTime.} =
  ## One cell's "(kind, width)" form in check messages.
  "(" & $ScalarCellAxis[idx].kind & ", " & $ScalarCellAxis[idx].width & ")"

proc checkCellNames(all: seq[tuple[display: string, cells: seq[ParsedCell]]]) {.compileTime.} =
  ## A type name denotes one (kind, width) cell.
  ##
  ## - within a target a repeated name fails at the second cell
  ## - a name two targets share must denote the same cell on both: msl
  ##   the row keeps `char`/`long` against cuda `signed char`/`long long` explicit
  ##   per-target data
  ## - bool cells are exempt: OpenCL prints bool locals as `int` to match
  ##   the 4-byte launch ABI
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
          if i == j:
            error("genTargets: name '" & ca.spelling & "' denotes two cells of " &
              all[i].display & ", " & cellKey(ca.idx) & " and " & cellKey(cb.idx), cb.node)
          elif cb.idx != ca.idx:
            error("genTargets: name '" & ca.spelling & "' denotes " & cellKey(ca.idx) &
              " on " & all[i].display & " but " & cellKey(cb.idx) &
              " on " & all[j].display, cb.node)

proc cellCtor(idx: int, spelling: string): NimNode {.compileTime.} =
  ## One ScalarCell constructor from a `ScalarCellAxis` entry.
  let axis = ScalarCellAxis[idx]
  nnkObjConstr.newTree(ident"ScalarCell", colon("kind", ident $axis.kind),
    colon("width", newLit axis.width), colon("spelling", newLit spelling))

proc parseFieldLines(entries: seq[NimNode], family, display: string): seq[tuple[section: string, pair: NimNode]] {.compileTime.} =
  ## Validated (section, constructor field) entries of a lit or kind row.
  ##
  ## - string fields take a string literal, bool fields take `true` or `false`
  ## - `targetExprKinds` takes a set of expression kinds
  ## - a bad member name fails the compile at the row: value nodes splice
  ##   into the generated constructor at their original position
  for line in entries:
    if line.kind != nnkCommand or line.len != 2:
      error("genTargets: expected a 'field value' entry in the " & family & " row of " & display, line)
    let field = $line[0]
    let section = fieldSection(family, field)
    if section == unknownField:
      error("genTargets: unknown field '" & field & "' in the " & family & " row of " & display, line)
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
    result.add (section, colon(field, val))

proc builtinTableCtor(entries: seq[NimNode], display: string, widthKeyed: bool): NimNode {.compileTime.} =
  ## Builtin name table of a conv row (`builtin "text"`) or a math row
  ## (`builtin width "text"`). `$1` is the call's argument list, a math key
  ## is the (name, bit width) pair.
  var names: seq[string]
  var pairs: seq[NimNode]
  for line in entries:
    # a math entry nests its `width "spelling"` pair, a conv entry's pair is flat
    let ok = line.kind == nnkCommand and line.len == 2 and line[0].kind == nnkIdent and
      (if widthKeyed: line[1].kind == nnkCommand and line[1].len == 2 and
        line[1][0].kind == nnkIntLit and line[1][1].kind == nnkStrLit
       else: line[1].kind == nnkStrLit)
    if not ok:
      error("genTargets: expected a 'builtin " &
        (if widthKeyed: "width " else: "") & "\"text\"' entry in the " &
        (if widthKeyed: "math" else: "conv") & " row of " & display, line)
    let name = $line[0]
    let key = if widthKeyed: name & ":" & $line[1][0].intVal else: name
    if key in names:
      error("genTargets: builtin '" & name & "' appears twice in the " &
        (if widthKeyed: "math" else: "conv") & " row of " & display, line)
    names.add key
    let k = if widthKeyed: nnkPar.newTree(newLit name, newLit line[1][0].intVal.int)
            else: newLit name
    pairs.add nnkExprColonExpr.newTree(k,
      (if widthKeyed: line[1][1] else: line[1]))
  # generated `{key: value, ...}.toTable()` expression
  nnkCall.newTree(nnkDotExpr.newTree(nnkTableConstr.newTree(pairs), ident"toTable"))

proc refusalsCtor(entries: seq[NimNode], display: string): NimNode {.compileTime.} =
  ## Refused builtin names of a refusals row, one bare name per entry.
  ## The generated value is the string array the printer matches against.
  var names: seq[string]
  var items: seq[NimNode]
  for line in entries:
    var name: string
    if line.kind == nnkIdent:
      name = $line
    elif line.kind == nnkStrLit:
      name = line.strVal
    else:
      error("genTargets: expected a builtin name in the refusals row of " & display, line)
    if name in names:
      error("genTargets: builtin '" & name & "' appears twice in the refusals row of " & display, line)
    names.add name
    items.add newLit name
  nnkBracket.newTree(items)

proc vectorCtor(entries: seq[NimNode], display: string): NimNode {.compileTime.} =
  ## Vector coverage rule of a vector row:
  ##
  ## - `elems` entries of `cell "prefix"` elements, a bare cell an empty prefix
  ## - one `lanes n, ...` entry, an optional `nameForm` entry
  ## - a target with no vectors lists `vector ()`
  var elems: seq[NimNode]
  var lanes: seq[NimNode]
  var nameForm = ""
  for entry in entries:
    if entry.kind == nnkTupleConstr and entry.len == 0:
      continue
    if entry.kind != nnkCommand or entry.len < 2:
      error("genTargets: expected an 'elems', 'lanes' or 'nameForm' entry in the vector row of " & display, entry)
    case $entry[0]
    of "elems":
      for arg in entry[1 ..^ 1]:
        let c = cellPair(arg, display)
        let axis = ScalarCellAxis[c.idx]
        elems.add nnkPar.newTree(ident $axis.kind, newLit axis.width, newLit c.spelling)
    of "lanes":
      for arg in entry[1 ..^ 1]:
        if arg.kind != nnkIntLit:
          error("genTargets: lane counts are integers in the vector row of " & display, arg)
        lanes.add newLit arg.intVal.int
    of "nameForm":
      if entry.len != 2 or entry[1].kind != nnkIdent or
          ($entry[1] != "vnfPrefixed" and $entry[1] != "vnfGenericElem"):
        error("genTargets: nameForm takes vnfPrefixed or vnfGenericElem", entry)
      nameForm = $entry[1]
    else:
      error("genTargets: unknown field '" & $entry[0] & "' in the vector row of " & display, entry)
  var args = @[ident"VectorRule"]
  if elems.len > 0:
    args.add colon("elems", nnkCall.newTree(ident"@", nnkBracket.newTree(elems)))
  if lanes.len > 0:
    args.add colon("lanes", nnkCall.newTree(ident"@", nnkBracket.newTree(lanes)))
  if nameForm != "":
    args.add colon("nameForm", ident nameForm)
  nnkObjConstr.newTree(args)

proc castRulesCtor(entries: seq[NimNode], display: string): NimNode {.compileTime.} =
  ## Bit-reinterpret coverage rules of a bitcast row, one `rule` entry per
  ## rule:
  ##
  ## - the rule's fields in schema names: `srcAny`, `srcKinds`, `srcWidths`,
  ##   `dstKinds`, `dstWidths`, `sameWidth`, kind sets over the `ctVoid`/
  ##   `ctBool`/`ctInt`/`ctUint`/`ctFloat` names, with the every-kind
  ##   destination set as `dstKinds {low(CirTypeKind)..high(CirTypeKind)}`
  ## - then one bare format text or `refusal "text"`
  ## - set and text nodes splice into the generated constructor unchanged,
  ##   so a bad member name fails the compile at the row
  var rules: seq[NimNode]
  for entry in entries:
    if entry.kind != nnkCommand or entry.len < 2 or $entry[0] != "rule":
      error("genTargets: expected a 'rule ...' entry in the bitcast row of " & display, entry)
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
  ## Registers every target's rows of one module in the block DSL.
  ## See the module header for the failure contract.
  body.expectKind nnkStmtList
  var
    keys: seq[tuple[target, flavor: string]]
    sections: seq[NimNode]
    ctors: seq[seq[tuple[family: string, ctor: NimNode]]]
    scalarCells: seq[tuple[display: string, cells: seq[ParsedCell]]]
  for section in body:
    if section.kind != nnkCommand or section.len != 3 or $section[0] != "target":
      error("genTargets: expected a 'target <name>:' section", section)
    let key = sectionKey(section[1])
    if key in keys:
      error("genTargets: duplicate target section '" & keyDisplay(key) & "'", section)
    keys.add key
    sections.add section
    let display = keyDisplay(key)
    var fams: seq[tuple[family: string, ctor: NimNode]]
    var seen: seq[string]
    for fam in section[2]:
      if fam[0].kind != nnkIdent or (fam.kind != nnkCommand and
          (fam.kind != nnkCall or fam[1].kind != nnkStmtList)):
        error("genTargets: expected a '<family> ...' row in target " & display, fam)
      let family = $fam[0]
      let (suffix, required) = familyRow(family)
      if suffix == "":
        error("genTargets: unknown row family '" & family & "' in target " & display, fam)
      if family in seen:
        error("genTargets: duplicate row family '" & family & "' in target " & display, fam)
      seen.add family
      let entries = famEntries(fam)
      var ctor: NimNode = nil
      case family
      of "scalar":
        var cells: seq[ParsedCell]
        var seen: seq[int]
        var refused: seq[tuple[idx: int, reason: NimNode]]
        for entry in entries:
          if entry.kind == nnkCommand and entry.len == 2 and $entry[0] == "refused":
            if entry[1].kind != nnkCommand or entry[1].len != 2 or
                entry[1][0].kind != nnkIdent or entry[1][1].kind != nnkStrLit:
              error("genTargets: expected a 'refused <cell> \"reason\"' entry in " &
                display, entry)
            let idx = cellIndex($entry[1][0])
            if idx < 0:
              error("genTargets: unknown cell '" & $entry[1][0] & "' in " & display, entry[1][0])
            if '\n' in entry[1][1].strVal:
              error("genTargets: the reason of refused cell '" & $entry[1][0] &
                "' is one line, in " & display, entry)
            for r in refused:
              if r.idx == idx:
                error("genTargets: refused cell '" & $entry[1][0] &
                  "' appears twice in " & display, entry)
            refused.add (idx, entry[1][1])
            continue
          let c = cellPair(entry, display)
          if c.spelling.len == 0:
            error("genTargets: scalar cell '" & ScalarCellAxis[c.idx].tok & "' of " &
              display & " takes a type name, a refused cell is omitted", entry)
          if c.idx in seen:
            error("genTargets: scalar cell '" & ScalarCellAxis[c.idx].tok &
              "' appears twice in " & display, entry)
          seen.add c.idx
          cells.add c
        for r in refused:
          if r.idx in seen:
            error("genTargets: refused cell '" & ScalarCellAxis[r.idx].tok &
              "' is also stated in " & display, r.reason)
        scalarCells.add (display, cells)
        var ctorArgs = @[ident"ScalarRow",
          colon("cells", nnkCall.newTree(ident"@", nnkBracket.newTree(
            block:
              var cellNodes: seq[NimNode]
              for c in cells:
                cellNodes.add cellCtor(c.idx, c.spelling)
              cellNodes)))]
        if refused.len > 0:
          var refusalNodes: seq[NimNode]
          for r in refused:
            refusalNodes.add nnkPar.newTree(ident $ScalarCellAxis[r.idx].kind,
              newLit ScalarCellAxis[r.idx].width, r.reason)
          ctorArgs.add colon("refusals",
            nnkCall.newTree(ident"@", nnkBracket.newTree(refusalNodes)))
        ctor = nnkObjConstr.newTree(ctorArgs)
      of "lit", "kind":
        ctor = objCtor(suffix, parseFieldLines(entries, family, display))
      of "conv":
        ctor = builtinTableCtor(entries, display, false)
      of "math":
        ctor = builtinTableCtor(entries, display, true)
      of "refusals":
        ctor = refusalsCtor(entries, display)
      of "vector":
        ctor = vectorCtor(entries, display)
      of "bitcast":
        ctor = castRulesCtor(entries, display)
      else:
        discard
      fams.add (family, ctor)
    ctors.add fams
  for i, key in keys:
    var fams: seq[string]
    for r in ctors[i]:
      fams.add r.family
    for f in genFamilies:
      if f[2] and f[0] notin fams:
        error("genTargets: target " & keyDisplay(key) & " is missing row family '" &
          f[0] & "'", sections[i])
  checkCellNames(scalarCells)
  result = nnkStmtList.newTree()
  for i, key in keys:
    for r in ctors[i]:
      result.add nnkConstSection.newTree(nnkConstDef.newTree(
        nnkPostfix.newTree(ident"*", ident(keySuffix(key) & familyRow(r.family).suffix)),
        newEmptyNode(), r.ctor))
