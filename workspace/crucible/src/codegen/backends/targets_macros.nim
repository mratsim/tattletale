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
## - block syntax: one `target <name>:` section per target key, one row expression per family
## - every one of the eight families appears per target, a target section appears once
## - one invocation registers every target of its module, generated lookup proc names would collide across invocations
##
## Row expressions pass through unchanged and target-unique emission code stays in the printers as ordinary procs.

import std/macros
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

macro genTargets*(body: untyped) =
  ## Registers the target rows of one module, see the module header for the block syntax.
  body.expectKind(nnkStmtList)
  var targets: seq[string] = @[]
  for section in body:
    if section.kind != nnkCommand or $section[0] != "target":
      error("genTargets: expected a 'target <name>:' section", section)
    let target = $section[1]
    if target in targets:
      error("genTargets: duplicate target section '" & target & "'", section)
    targets.add target
    var seen: seq[string] = @[]
    for fam in section[2]:
      if fam.kind != nnkCall:
        error("genTargets: expected a '<family>: <row>' line in target " & target, fam)
      let family = $fam[0]
      if rowTypeName(family).len == 0:
        error("genTargets: unknown row family '" & family & "' in target " & target, fam)
      if family in seen:
        error("genTargets: duplicate row family '" & family & "' in target " & target, fam)
      seen.add family
      if fam[1].kind != nnkStmtList or fam[1].len != 1:
        error("genTargets: family '" & family & "' takes exactly one row expression", fam)
    for family in genFamilies:
      if family notin seen:
        error("genTargets: target " & target & " is missing row family '" &
          family & "'", section)
  result = nnkStmtList.newTree()
  for section in body:
    let target = $section[1]
    for fam in section[2]:
      let family = $fam[0]
      let rowType = rowTypeName(family)
      let constName = ident(target & rowType & "Data")
      let lookupName = ident(target & rowType)
      let typeIdent = ident(rowType)
      let value = fam[1][0]
      result.add quote do:
        const `constName` = `value`
        proc `lookupName`*(): `typeIdent` {.inline.} = `constName`
  var checks = nnkStmtList.newTree()
  for target in targets:
    let scalarLookup = ident(target & "ScalarRow")
    let coordLookup = ident(target & "CoordRow")
    let targetLit = newLit(target)
    checks.add quote do:
      checkScalarRow(`scalarLookup`(), `targetLit`)
      checkCoordRow(`coordLookup`(), `targetLit`)
  let togetherArgs = nnkBracket.newTree()
  for target in targets:
    let scalarLookup = ident(target & "ScalarRow")
    let targetLit = newLit(target)
    togetherArgs.add nnkPar.newTree(targetLit, nnkCall.newTree(scalarLookup))
  checks.add nnkCall.newTree(ident"checkScalarRowsTogether", togetherArgs)
  result.add nnkStaticStmt.newTree(checks)
