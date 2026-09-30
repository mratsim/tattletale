# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros, std/typetraits
import ./int_tuples_datatypes
import ./int_tuples_transforms

# ═══════════════════════════════════════════════════════════════
#  zipDimensionsWith, zip the top-level elements with a binary op
# ═══════════════════════════════════════════════════════════════

macro zipDimensionsWith*[A, B: IntOrIntTuple](a: A; b: B; body: untyped): untyped =
  ## Zip top-level elements of tuples `a` and `b` pairwise via `body` (does NOT recurse into nested tuples).
  ## `it_a` / `it_b` bind to corresponding elements.
  ## Leftover elements from the longer tuple are appended unchanged.
  ##
  ## Example:
  ##   zipWith((2, 4), (10, 20)): it_a + it_b  →  (12, 24)
  ##   zipWith((2, 4, 6), (10, 20)): it_a + it_b  →  (12, 24, 6)
  let ta = getTypeInst(a)
  let tb = getTypeInst(b)
  let RA = if ta.kind == nnkTupleConstr: ta.len else: 1
  let RB = if tb.kind == nnkTupleConstr: tb.len else: 1
  let rMin = min(RA, RB)
  let rMax = max(RA, RB)

  proc subst(x: NimNode; i: int; la, lb: NimNode): NimNode =
    if x.kind in {nnkIdent, nnkSym} and x.eqIdent("it_a"):
      result = nnkBracketExpr.newTree(la, newLit(i))
    elif x.kind in {nnkIdent, nnkSym} and x.eqIdent("it_b"):
      result = nnkBracketExpr.newTree(lb, newLit(i))
    else:
      result = x.copyNimTree()
      for j in 0 ..< x.len:
        result[j] = subst(x[j], i, la, lb)

  result = newStmtList()
  var items: seq[NimNode]
  for i in 0 ..< rMax:
    let name = ident("__zw" & $i)
    if i < rMin:
      items.add name
      result.add newLetStmt(name, subst(body, i, a, b))
    elif i < RA:
      items.add nnkBracketExpr.newTree(a, newLit(i))
    else:
      items.add nnkBracketExpr.newTree(b, newLit(i))
  # nnkPar: single-item result stays scalar (avoids explicit `if result.len == 1`).
  # Multi-item: construct a tuple like nnkTupleConstr.
  result.add nnkPar.newTree(items)

# ═══════════════════════════════════════════════════════════════
#  zipLeavesWith, element-wise binary op for equal-structure tuples
# ═══════════════════════════════════════════════════════════════

template zipLeavesRecur*(a, b: typed; idx: static int; body: untyped): untyped =
  ## Internal: walk tuple from index `idx`, recurse into nested tuples.
  when idx < tupleLen(typeof(a)):
    when a[idx] is tuple:
      concat((zipLeavesRecur(a[idx], b[idx], 0, body),),
             zipLeavesRecur(a, b, idx + 1, body))
    else:
      block:
        let it_a {.inject.} = a[idx]
        let it_b {.inject.} = b[idx]
        when idx == tupleLen(typeof(a)) - 1:
          (body,)
        else:
          concat((body,), zipLeavesRecur(a, b, idx + 1, body))
  else:
    ()

template zipLeavesWith*(a, b: typed; body: untyped): untyped =
  ## Apply `body` to corresponding leaf pairs of equal-structure tuples.
  ## Inside `body`, `it` is a 2-tuple `(leaf_a, leaf_b)`.
  ## Use `it_a` for the leaf from `a` and `it_b` for the leaf from `b`.
  ##
  ## runnableExamples:
  ##   let r = zipLeavesWith((10, 10), (3, 2)): it_a - it_b
  ##   doAssert r == (7, 8)
  ##   let r2 = zipLeavesWith(((1, 2), 3), ((4, 5), 6)): it_a - it_b
  ##   doAssert r2 == ((-3, -3), -3)
  when a is tuple:
    zipLeavesRecur(a, b, 0, body)
  else:
    let it_a = a
    let it_b = b
    body
