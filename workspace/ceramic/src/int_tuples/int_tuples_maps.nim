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
    if t.kind == nnkTupleConstr:
      ## Direct destructure — preserves compile-time info for const elements
      for child in t:
        let recurse = newCall(ident"mapLeavesWith", child, body)
        elems.add recurse
    else:
      ## Bracket access for variables / function returns
      for i in 0 ..< tType.len:
        let fieldAccess = nnkBracketExpr.newTree(t, newLit i)
        let recurse = newCall(ident"mapLeavesWith", fieldAccess, body)
        elems.add recurse
    result = nnkTupleConstr.newTree(elems)
    return

  result = body.replaceNodes(ident"it", t)

# ═══════════════════════════════════════════════════════════════════════
#  flatMapLeaves — flat leaf-wise tuple map (single pack)
# ═══════════════════════════════════════════════════════════════════════

proc flatMapLeavesImpl(tNode: NimNode; body: NimNode): NimNode {.compileTime.} =
  ## Build the flat pack for `tNode`, one node per leaf, `body` with `it`
  ## replaced by the leaf access. Returns the untyped tuple construction.

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

  proc isLeaf(t: NimNode): bool =
    (t.kind == nnkSym and $t == "int") or
    (t.kind == nnkBracketExpr and $t[0] == "Int")

  proc leafAccess(e, t: NimNode; idx: int): NimNode =
    if e.kind in {nnkTupleConstr, nnkPar} and e.len == t.len:
      e[idx]
    else:
      newTree(nnkBracketExpr, e, newLit(idx))

  proc collect(acc: var NimNode; e: NimNode; t: NimNode) =
    if t.kind == nnkTupleConstr:
      for idx in 0 ..< t.len:
        let fd = t[idx]
        let fa = leafAccess(e, t, idx)
        if isLeaf(fd):
          acc.add body.replaceNodes(ident"it", fa)
        else:
          collect(acc, fa, fd)
    else:
      acc.add body.replaceNodes(ident"it", e)

  let tType = tNode.getTypeImpl()
  # nnkPar keeps a single-item result scalar and packs multi-item results
  # like nnkTupleConstr, avoiding an explicit `if result.len == 1` branch.
  result = newNimNode(nnkPar)
  collect(result, tNode, tType)

macro flatMapLeaves*(t: IntOrIntTuple, body: untyped): untyped =
  ## Map every leaf of a (possibly nested) tuple to exactly one output
  ## element and emit the flat result as a single tuple construction.
  ##
  ## - `body` is evaluated once per leaf with `it` bound to the leaf
  ## - The output is flat, the input's nesting is collapsed
  ##
  ## Returns the flat tuple construction.
  ##
  ## CuTe analogue, `transform_apply` + `tuple_cat(a...)` (tuple_algorithms.hpp).
  ##
  ## Example spellings:
  ##
  ##   flatMapLeaves(5, it)                 → 5
  ##   flatMapLeaves((1, (2, 3)), it)       → (1, 2, 3)
  ##
  ##   flatMapLeaves((2, 3), it * 10)       → (20, 30)
  flatMapLeavesImpl(t, body)

# ═══════════════════════════════════════════════════════════════════════
#  mapDimensionsWith — Top-level only tuple map
# ═══════════════════════════════════════════════════════════════════════

macro mapDimensionsWith*(t: tuple; body: untyped): untyped =
  ## Apply `body` to each top-level element of tuple `t` (does NOT recurse into nested tuples).
  ## `it` binds to the current element.
  ##
  ## Example:
  ##   mapDimensionsWith((2, 4, 6)): it * 2  →  (4, 8, 12)
  let tt = getTypeInst(t)
  let n = tt.len

  proc subst(x: NimNode; i: int; ttup: NimNode): NimNode =
    if x.kind in {nnkIdent, nnkSym} and x.eqIdent("it"):
      result = nnkBracketExpr.newTree(ttup, newLit(i))
    else:
      result = x.copyNimTree()
      for j in 0 ..< x.len: result[j] = subst(x[j], i, ttup)

  var items: seq[NimNode]
  for i in 0 ..< n:
    items.add subst(body, i, t)
  # nnkTupleConstr: always preserve tuple structure (even for single-element results)
  result = nnkTupleConstr.newTree(items)
