# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ./int_tuples_compiletime
import ./int_tuples_datatypes
import ./int_tuples_streams
import workspace/ceramic/src/macros/replace_nodes

# ═══════════════════════════════════════════════════════════════════════
#  mapLeavesWith, recursive leaf-wise tuple map
# ═══════════════════════════════════════════════════════════════════════

macro mapLeavesWith*(t: IntOrIntTuple, body: untyped): untyped =
  ## Recursively walk `t` (int | Int[N] | tuple) and apply `body` to
  ## every leaf.  Returns a value of the same shape with leaves transformed.
  ##
  ## Example spellings:
  ##
  ##   mapLeavesWith((1, (2, 3)), it * 10)  →  (10, (20, 30))
  ##   mapLeavesWith(5, it)                 →  5
  ##   mapLeavesWith((2, 3), it * 10)       →  (20, 30)
  let scalar = not t.isTupleTy() # Preserve 1 vs (1, ) input
  var builder = TupleBuilderNested.new(1, dropEmpty = false)
  for event in t.tupleStream():
    builder.onLeaves(event):
      builder.append(body.replaceNodes(("it", event.leaf)))
  result = builder.emit(0, emitScalarForSize1 = scalar)

# ═══════════════════════════════════════════════════════════════════════
#  flatMapLeaves, flat leaf-wise tuple map of a single pack
# ═══════════════════════════════════════════════════════════════════════

proc flatMapLeavesImpl(tNode: NimNode; body: NimNode): NimNode {.compileTime.} =
  ## Build the flat pack for `tNode`, one node per leaf, `body` with `it`
  ## replaced by the leaf access. Returns the untyped tuple construction.
  # nnkPar keeps a single-item result scalar and packs multi-item results
  # like nnkTupleConstr, avoiding an explicit `if result.len == 1` branch.
  result = newNimNode(nnkPar)
  for (leaf, _) in tNode.tupleStream().leaves():
    result.add body.replaceNodes(("it", leaf))

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
#  mapDimensionsWith, top-level-only tuple map
# ═══════════════════════════════════════════════════════════════════════

macro mapDimensionsWith*(t: tuple; body: untyped): untyped =
  ## Apply `body` to each top-level element of tuple `t` (does NOT recurse into nested tuples).
  ## `it` binds to the current element.
  ##
  ## Example:
  ##   mapDimensionsWith((2, 4, 6)): it * 2  →  (4, 8, 12)
  var builder = TupleBuilderFlat.new(1)
  for event in t.tupleDimsStream():
    builder.onLeaves(event):
      builder.append(body.replaceNodes(("it", event.leaf)))
  result = builder.emit(0)
