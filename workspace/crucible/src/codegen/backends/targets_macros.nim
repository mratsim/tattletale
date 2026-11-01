# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## `genTargets`: the declarative target-row macro.
##
## One invocation parses the rows of its module into the registry's schema objects (targets_registry).
## It generates one lookup proc per family and target and emits the registry's checks as one compile-time block.
## A missing or drifted row is a compile error.
##
## - block syntax: one `target <name>:` section per target key, one row expression per family.
##   A target with flavors keys its flavor-dependent families in a `target <name>(<flavor>):` section
## - every one of the eight families resolves per target key, out of the key's own section
##   plus its target's flavorless base section. A family declared in both the base
##   and a flavor section of one target fails the compile
## - one invocation registers every target of its module, generated lookup proc names would collide across invocations
##
## Row expressions pass through unchanged and target-unique emission code stays in the printers as ordinary procs.

import std/macros
import std/strutils
import ./targets_registry

const genFamilies*: array[8, string] = [
  "scalar", "lit", "kind", "space", "conv", "coord", "sig", "vector"]

proc rowTypeName(family: string): string {.compileTime.} =
  ## Schema type of one row family, "" for an unknown family.
  case family
  of "scalar": "ScalarRow"
  of "lit": "LitRow"
  of "kind": "KindRow"
  of "space": "SpaceRow"
  of "conv": "ConvRow"
  of "coord": "CoordRow"
  of "sig": "SigRow"
  of "vector": "VectorRule"
  else: ""

proc sectionKey(n: NimNode): tuple[target, flavor: string] {.compileTime.} =
  ## Target key of one section header, `target <name>:` or `target <name>(<flavor>):`.
  if n.kind == nnkIdent:
    result = ($n, "")
  elif n.kind == nnkCall and n.len == 2 and n[0].kind == nnkIdent and n[1].kind == nnkIdent:
    result = ($n[0], $n[1])
  else:
    error("genTargets: expected a 'target <name>:' or 'target <name>(<flavor>):' section header", n)

proc keyName(key: tuple[target, flavor: string]): string {.compileTime.} =
  ## Name of one target key in check messages. A flavored key prints as `target(flavor)`.
  if key.flavor == "":
    key.target
  else:
    key.target & "(" & key.flavor & ")"

proc lookupSuffix(key: tuple[target, flavor: string]): string {.compileTime.} =
  ## Suffix a target key contributes to generated const and lookup proc names.
  if key.flavor == "":
    key.target
  else:
    key.target & key.flavor[0].toUpperAscii & key.flavor[1 ..^ 1]

macro genTargets*(body: untyped) =
  ## Registers the target rows of one module, see the module header for the block syntax.
  body.expectKind(nnkStmtList)
  var keys: seq[tuple[target, flavor: string]] = @[]
  var fams: seq[seq[string]] = @[]
  var sections: seq[NimNode] = @[]
  for section in body:
    if section.kind != nnkCommand or $section[0] != "target":
      error("genTargets: expected a 'target <name>:' section", section)
    let key = sectionKey(section[1])
    if key in keys:
      error("genTargets: duplicate target section '" & keyName(key) & "'", section)
    keys.add key
    sections.add section
    var seen: seq[string] = @[]
    for fam in section[2]:
      if fam.kind != nnkCall:
        error("genTargets: expected a '<family>: <row>' line in target " & keyName(key), fam)
      let family = $fam[0]
      if rowTypeName(family).len == 0:
        error("genTargets: unknown row family '" & family & "' in target " & keyName(key), fam)
      if family in seen:
        error("genTargets: duplicate row family '" & family & "' in target " & keyName(key), fam)
      seen.add family
      if fam[1].kind != nnkStmtList or fam[1].len != 1:
        error("genTargets: family '" & family & "' takes exactly one row expression", fam)
    fams.add seen
  # Every consumable key resolves all eight row families. A plain target's
  # section states all of them. A flavored target's flavorless section states
  # its target-wide families, and each flavored section adds the flavor-dependent
  # ones beside them
  for i, key in keys:
    var targetHasFlavors = false
    for other in keys:
      if other.target == key.target and other.flavor != "":
        targetHasFlavors = true
    if key.flavor == "" and targetHasFlavors:
      continue
    var available = fams[i]
    for j, base in keys:
      if base.target == key.target and base.flavor == "" and j != i:
        available = available & fams[j]
    for family in genFamilies:
      if family notin available:
        error("genTargets: target " & keyName(key) & " is missing row family '" &
          family & "'", sections[i])
  # a family of a flavored key resolves once: not also in the base section
  for i, key in keys:
    if key.flavor == "":
      continue
    for j, base in keys:
      if base.target == key.target and base.flavor == "":
        for family in fams[i]:
          if family in fams[j]:
            error("genTargets: family '" & family & "' of target " & keyName(key) &
              " is already declared in the flavorless section of " & key.target,
              sections[i])
  result = nnkStmtList.newTree()
  for i, section in sections:
    let key = keys[i]
    for fam in section[2]:
      let family = $fam[0]
      let rowType = rowTypeName(family)
      let constName = ident(lookupSuffix(key) & rowType & "Data")
      let lookupName = ident(lookupSuffix(key) & rowType)
      let typeIdent = ident(rowType)
      let value = fam[1][0]
      result.add quote do:
        const `constName` = `value`
        proc `lookupName`*(): `typeIdent` {.inline.} = `constName`
  var checks = nnkStmtList.newTree()
  var scalarEntries: seq[NimNode] = @[]
  for i, key in keys:
    for family in fams[i]:
      let lookup = ident(lookupSuffix(key) & rowTypeName(family))
      let display = newLit(keyName(key))
      if family == "scalar":
        checks.add quote do:
          checkScalarRow(`lookup`(), `display`)
        scalarEntries.add nnkPar.newTree(display, nnkCall.newTree(lookup))
      elif family == "coord":
        checks.add quote do:
          checkCoordRow(`lookup`(), `display`)
  let togetherArgs = nnkBracket.newTree()
  for entry in scalarEntries:
    togetherArgs.add entry
  checks.add nnkCall.newTree(ident"checkScalarRowsTogether", togetherArgs)
  result.add nnkStaticStmt.newTree(checks)
