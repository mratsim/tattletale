# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros, std/typetraits
import ./int_tuples_compiletime
import ./int_tuples_datatypes
import ./int_tuples_streams
import ./int_tuples_transforms
import workspace/ceramic/src/macros/replace_nodes

# ═══════════════════════════════════════════════════════════════
#  zipDimensionsWith, zip the top-level elements with a binary op
# ═══════════════════════════════════════════════════════════════

macro zipDimensionsWith*[A, B: IntOrIntTuple](a: A, b: B, body: untyped): untyped =
  ## Zip top-level elements of tuples `a` and `b` pairwise via `body` (does NOT recurse into nested tuples).
  ## `it_a` / `it_b` bind to corresponding elements.
  ## Leftover elements from the longer tuple are appended unchanged.
  ##
  ## Contract:
  ## - the zip is top-level only, a nested tuple element is one element,
  ##   the body sees it whole
  ## - ranks may differ, the longer tuple's tail passes through unchanged
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

  result = newStmtList()
  var items: seq[NimNode]
  for i in 0 ..< rMax:
    let name = ident("__zw" & $i)
    if i < rMin:
      items.add name
      result.add newLetStmt(name, body.replaceNodesAt(("it_a", a), ("it_b", b), i))
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

macro zipLeavesWith*(a, b: typed, body: untyped): untyped =
  ## Apply `body` to corresponding leaf pairs of equal-structure tuples.
  ## Inside `body`, `it` is a 2-tuple `(leaf_a, leaf_b)`.
  ## Use `it_a` for the leaf from `a` and `it_b` for the leaf from `b`.
  ##
  ## Contract:
  ## - `a` and `b` have the same structure, one leaf pair at a time,
  ##   recursion-free, no runtime `let` in the emitted code
  ## - scalar inputs unwrap, the result is the mapped scalar
  ##
  ## Example:
  ##   let r = zipLeavesWith((10, 10), (3, 2)): it_a - it_b
  ##   doAssert r == (7, 8)
  ##   let r2 = zipLeavesWith(((1, 2), 3), ((4, 5), 6)): it_a - it_b
  ##   doAssert r2 == ((-3, -3), -3)
  var inputTuple = a.getTypeInst().isTupleTy()
  var builder = TupleBuilderNested.new(1)
  var aStream = a.tupleStream()
  var bStream = b.tupleStream()
  while not aStream.done():
    if bStream.done():
      error "zipLeavesWith: `b` has fewer elements than `a`", b
    let aEvent = aStream.next()
    let bEvent = bStream.next()
    if aEvent.kind != bEvent.kind:
      error "zipLeavesWith: `a` and `b` have different tuple structures", b
    case aEvent.kind
    of kLeaf:
      let mapped = body.replaceNodes(
        ("it_a", aEvent.leaf), ("it_b", bEvent.leaf),
        ("it", nnkPar.newTree(aEvent.leaf, bEvent.leaf)))
      builder.append(mapped, verbatim = false)
    else:
      builder.append(aEvent)
  if not bStream.done():
    error "zipLeavesWith: `b` has more elements than `a`", b
  let (resultTuple, _) = builder.emit(0, emitScalarForSize1 = not inputTuple)
  result = newStmtList(resultTuple)
