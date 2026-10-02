# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Compile-time AST substitution for macros

import std/macros

proc replaceNodes*(ast: NimNode, target: varargs[tuple[what: string, by: NimNode]]): NimNode =
  var targets: seq[tuple[what: string, by: NimNode]]
  targets = @target
  proc inspect(node: NimNode): NimNode =
    case node.kind
    of {nnkIdent, nnkSym}:
      for (what, by) in targets:
        if node.eqIdent(what):
          return by
      return node
    of nnkEmpty, nnkLiterals:
      return node
    else:
      result = node.copyNimTree()
      for j in 0 ..< node.len:
        result[j] = inspect(node[j])
  result = inspect(ast)

proc replaceNodesAt*(ast: NimNode, target: varargs[tuple[what: string, by: NimNode]], idx: int): NimNode =
  ## Replace every ident/sym named `what` with `by[idx]`.
  var targets = newSeq[tuple[what: string, by: NimNode]](target.len)
  for j in 0 ..< target.len:
    let byElem = nnkBracketExpr.newTree(target[j].by, newLit(idx))
    targets[j] = (what: target[j].what, by: byElem)
  result = replaceNodes(ast, targets)
