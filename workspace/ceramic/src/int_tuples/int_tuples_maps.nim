# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ./int_tuples_datatypes

# ═══════════════════════════════════════════════════════════════════════
#  mapLeavesWith — recursive leaf‑wise tuple map
# ═══════════════════════════════════════════════════════════════════════

macro mapLeavesWith*(t: IntOrIntTuple, body: untyped): untyped =
  ## Recursively walk `t` (int | Int[N] | tuple) and apply `body` to
  ## every leaf.  Returns a value of the same shape with leaves transformed.

  proc replaceNodes(ast, what, by: NimNode): NimNode =
    proc inspect(node: NimNode): NimNode =
      case node.kind
      of {nnkIdent, nnkSym}:
        if node.eqIdent(what): return by
        return node
      of nnkEmpty, nnkLiterals:
        return node
      else:
        result = node.kind.newTree()
        for child in node:
          result.add inspect(child)
    result = inspect(ast)

  let tType = t.getTypeInst()

  if tType.kind in {nnkTupleTy, nnkTupleConstr}:
    var elems: seq[NimNode]
    for i in 0 ..< tType.len:
      let fieldAccess = nnkBracketExpr.newTree(t, newLit i)
      let recurse = newCall(ident"mapLeavesWith", fieldAccess, body)
      elems.add recurse
    result = nnkTupleConstr.newTree(elems)
    return

  result = body.replaceNodes(ident"it", t)

# ═══════════════════════════════════════════════════════════════════════
#  mapModesWith — Top-level only tuple map
# ═══════════════════════════════════════════════════════════════════════

macro mapModesWith*[T: IntOrIntTuple](t: T; body: untyped): untyped =
  ## Apply `body` to each top-level element of tuple `t` (does NOT recurse into nested tuples).
  ## `it` binds to the current element.
  ##
  ## Example:
  ##   map((2, 4, 6)): it * 2  →  (4, 8, 12)
  let tt = getTypeInst(t)
  let n = if tt.kind == nnkTupleConstr: tt.len else: 1

  proc subst(x: NimNode; i: int; ttup: NimNode): NimNode =
    if x.kind in {nnkIdent, nnkSym} and x.eqIdent("it"):
      result = nnkBracketExpr.newTree(ttup, newLit(i))
    else:
      result = x.copyNimTree()
      for j in 0 ..< x.len:
        result[j] = subst(x[j], i, ttup)

  var items: seq[NimNode]
  for i in 0 ..< n:
    items.add subst(body, i, t)
  # nnkPar: single-item result stays scalar (avoids explicit `if result.len == 1`).
  # Multi-item: construct a tuple like nnkTupleConstr.
  result = nnkPar.newTree(items)