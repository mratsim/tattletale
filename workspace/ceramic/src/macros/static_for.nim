# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros

proc replaceNodes(ast: NimNode, what: NimNode, by: NimNode): NimNode =
  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of {nnkIdent, nnkSym}:
      if node.eqIdent(what): return by
      return node
    of nnkEmpty, nnkLiterals:
      return node
    else:
      var rTree = node.kind.newTree()
      for child in node:
        rTree.add inspect(child)
      return rTree
  result = inspect(ast)

macro staticFor*(idx: untyped{nkIdent}, start, stopEx: static int, body: untyped): untyped =
  ## Unroll a loop at compile time, binding `idx` to each integer in [start, stopEx).
  ##
  ## Each iteration is wrapped in a unique block statement to prevent symbol
  ## redefinition errors.
  ##
  ## Examples:
  ##   var r: typeof(shape)
  ##   staticFor i, 0, rank(shape):
  ##     r[i] = 1
  result = newStmtList()
  for i in start ..< stopEx:
    result.add nnkBlockStmt.newTree(
      ident("unrolledIter_" & $idx & $i),
      body.replaceNodes(idx, newLit i))

macro staticForCountdown*(idx: untyped{nkIdent}, start, stopIncl: static int, body: untyped): untyped =
  ## Unroll a loop in REVERSE at compile time, binding `idx` to each
  ## integer from `start` down to `stopIncl` (inclusive).
  ##
  ## Reference: Constantine (static_for.nim) — same API.
  ##
  ## Examples:
  ##   var r: typeof(shape)
  ##   staticForCountdown i, rank(shape)-1, 0:
  ##     r[i] = 1
  result = newStmtList()
  for i in countdown(start, stopIncl):
    result.add nnkBlockStmt.newTree(
      ident("cdIter_" & $idx & $i),
      body.replaceNodes(idx, newLit i))
