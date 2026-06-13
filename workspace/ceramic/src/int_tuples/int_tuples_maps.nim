# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ./int_tuples_datatypes

macro scaleBy*(t: typed; multiplier: typed): untyped =
  ## Multiply each leaf of a (possibly nested) tuple by scalar m.
  ## Preserves nesting structure.
  # `typed` required: type-class constraints (tuple, int or Int)
  # fail in generic functions where types aren't concrete yet.
  proc scaleImpl(nestedExpr, node, mul: NimNode): NimNode =
    if node.kind == nnkTupleConstr:
      result = newNimNode(nnkTupleConstr)
      for i in 0 ..< node.len:
        result.add scaleImpl(
          nnkBracketExpr.newTree(nestedExpr, newLit(i)), node[i], mul)
    else:
      result = newCall(bindSym"*", nestedExpr, mul)
  result = scaleImpl(t, t.getTypeImpl(), multiplier)


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