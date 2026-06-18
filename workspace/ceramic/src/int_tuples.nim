# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./int_tuples/int_tuples_filters
import std/macros, std/typetraits
import ./int_tuples/int_tuples_datatypes
import ./int_tuples/int_tuples_compiletime
import ./int_tuples/int_tuples_folds
import ./int_tuples/int_tuples_maps
import ./int_tuples/int_tuples_transforms
import ./int_tuples/int_tuples_zips

export int_tuples_datatypes
export int_tuples_filters
export int_tuples_compiletime
export int_tuples_folds
export int_tuples_maps
export int_tuples_transforms
export int_tuples_zips

# ═══════════════════════════════════════════════════════════════
#  makeIntTuple — wrap static ints / Int literals in Int[N]
# ═══════════════════════════════════════════════════════════════

#  Leaf procs — dispatch on exact type

template makeIntTupleLeaf(leaf: int): int =
  leaf

template makeIntTupleLeaf(leaf: static int): auto =
  Int[leaf]()

template makeIntTupleLeaf[V: static int](x: Int[V]): Int[V] =
  x

template makeIntTuple*(t: IntOrIntTuple): auto =
  ## Convert all `int` leaves in `t` (at any nesting depth) to `Int[N]()`.
  runnableExamples:
    let t = makeIntTuple((3, 4))
    doAssert t is (Int[3], Int[4])

  bind makeIntTupleLeaf
  mapLeavesWith(t):
    makeIntTupleLeaf(it)

# ═══════════════════════════════════════════════════════════════
#  Maps
# ═══════════════════════════════════════════════════════════════

template scaleBy*(t: IntOrIntTuple, scale: int): auto =
  t.mapLeavesWith():
    it * scale

template scaleBy*(t: IntOrIntTuple, scale: Int): auto =
  t.mapLeavesWith():
    it * scale

# ═══════════════════════════════════════════════════════════════
#  Prefix / suffix scans
# ═══════════════════════════════════════════════════════════════

template prefix_product*(shape: IntOrIntTuple): untyped =
  ## Cumulative left-to-right product scan.
  ##
  ## Each output element is the product of all input elements
  ## **before** that position. This is a **scan** (not a reduce),
  ## so the first output is always 1 (the multiplicative identity
  ## over zero elements).
  ##
  ## Examples:
  ##   prefix_product(5)        → 1         (scalar)
  ##   prefix_product((a,))     → (1,)
  ##   prefix_product((a, b))   → (1, a)
  ##   prefix_product((2,3,4))  → (1, 2, 6)
  ##   prefix_product((2,3,4))  === (1, 2, 6)
  ##
  ## Identical to CuTe's prefix_product / make_seq.
  ##
  ## See also: suffix_product, prefix_scanIt
  prefix_scanIt(shape, Int[1](), acc * it)

template suffix_product*(shape: IntOrIntTuple): untyped =
  ## Cumulative right-to-left product scan.
  ##
  ## Right-to-left variant of prefix_product:
  ##   suffix_product((a,b,c))  → (b*c, c, 1)
  ##   suffix_product((2,3,4))  → (12, 4, 1)
  ##
  ## The last output is always 1 (identity over zero elements).
  ##
  ## See also: prefix_product, suffix_scanIt
  suffix_scanIt(shape, Int[1](), acc * it)

# ═══════════════════════════════════════════════════════════════
#  Reductions
# ═══════════════════════════════════════════════════════════════

func product*(t: IntOrIntTuple): auto =
  ## Product of all leaves (like size for shapes). Returns Int[N] for all-Int, int otherwise.
  fold(t, Int[1](), acc * it)

func max*(t: IntOrIntTuple): auto =
  ## Maximum leaf value. Returns Int[N] for all-Int, int otherwise.
  fold(t, Int[low(int)](), max(acc, it))

func min*(t: IntOrIntTuple): auto =
  ## Minimum leaf value. Returns Int[N] for all-Int, int otherwise.
  fold(t, Int[high(int)](), min(acc, it))

# ═══════════════════════════════════════════════════════════════
#  product_each — product of each top-level tuple element
# ═══════════════════════════════════════════════════════════════

func product_each*(t: IntOrIntTuple): auto =
  ## For each top-level element of tuple `t`, compute the product of its leaves.
  ##
  ## Examples:
  ##   product_each(((2,2), (2,8)))  →  (4, 16)
  ##   product_each(5)                →  5
  ##
  mapModesWith(t): product(it)
