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
#  zipModesWith — zip tuple top-level with `op`
# ═══════════════════════════════════════════════════════════════

macro zipModesWith*[A, B: IntOrIntTuple](a: A; b: B; body: untyped): untyped =
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

# ═══════════════════════════════════════════════════════════════════════════════
#  zipLeavesWith — element-wise binary op for equal-structure tuples
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
#  foldZipWith — fold(zipWith(`op`(it_a, it_b)))
# ═══════════════════════════════════════════════════════════════

template foldZipWith_recurse*(idx: static int; a, b: tuple; state: typed; body: untyped): auto =
  static: doAssert tupleLen(a) == tupleLen(b), "foldZipWith: tuples must have same rank"
  const N = tupleLen(a)
  let field = foldZipWith(a[idx], b[idx], state, body)
  when idx == N - 1:
    field
  else:
    foldZipWith_recurse(idx + 1, a, b, field, body)

template foldZipWith*(a, b: typed; startingAcc: typed; body: untyped): auto =
  ## Fold over paired leaves of a and b.
  ## Injects `acc`, `it_a`, `it_b`.
  when a is tuple and b is tuple:
    foldZipWith_recurse(0, a, b, startingAcc, body)
  else:
    block:
      let acc {.inject.} = startingAcc
      let it_a {.inject.} = a
      let it_b {.inject.} = b
      body

# ═══════════════════════════════════════════════════════════════
#  zip2_by — guided zip for rank-2 tuples
# ═══════════════════════════════════════════════════════════════

template zip2_by*(t: tuple; guide: int): auto =
  ## CuTe: zip2_by(t, guide) — tuple_algorithms.hpp
  ## Terminal guide: t must be a pair, returned as-is.
  t

template zip2_by*[V: static int](t: tuple; guide: Int[V]): auto =
  ## Terminal Int[N] guide.
  t

macro zip2_by_impl(t, guide: typed): untyped =
  let guideTyp = guide.getTypeInst()
  let guideLen = guideTyp.len
  let tLen = t.getTypeInst().len
  var firstParts = newNimNode(nnkTupleConstr)
  var secondParts = newNimNode(nnkTupleConstr)
  for i in 0 ..< guideLen:
    let ti = nnkBracketExpr.newTree(t, newLit(i))
    let gi = nnkBracketExpr.newTree(guide, newLit(i))
    if guideTyp[i].kind == nnkTupleConstr:
      let splitPair = newCall(bindSym"zip2_by_impl", ti, gi)
      firstParts.add nnkBracketExpr.newTree(splitPair, newLit(0))
      secondParts.add nnkBracketExpr.newTree(splitPair, newLit(1))
    else:
      firstParts.add nnkBracketExpr.newTree(ti, newLit(0))
      secondParts.add nnkBracketExpr.newTree(ti, newLit(1))
  for i in guideLen ..< tLen:
    secondParts.add nnkBracketExpr.newTree(t, newLit(i))
  result = nnkTupleConstr.newTree(firstParts, secondParts)

func zip2_by*[T: tuple, G: tuple](t: T; guide: G): auto {.inline, noInit.} =
  ## Guided zip: split flat tuple `t` into (first_parts, second_parts).
  ##
  ## For each i where guide[i] is a tuple: recurse zip2_by(t[i], guide[i]).
  ## For each i where guide[i] is scalar: t[i] must be a pair; its two
  ##   elements go to first_parts and second_parts respectively.
  ## Extra elements of t (beyond guide length) append to second_parts.
  ##
  ## Result: (first_parts_tuple, second_parts_tuple) — rank-2.
  ##
  ## CuTe: zip2_by(t, guide) — tuple_algorithms.hpp line ~739
  ## Used by tile_unzip → zipped_divide/zipped_product.
  zip2_by_impl(t, guide)