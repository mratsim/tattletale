# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for int_tuples.nim: Int[N], fold, product, max, min,
## prefix_scanIt, suffix_scanIt, prefix_product, suffix_product.

import std/macros
import workspace/ceramic/src/int_tuples {.all.}

# ═══════════════════════════════════════════════════════════════
#  fold tests
# ═══════════════════════════════════════════════════════════════

proc runFoldTests =
  block:
    let r = fold(5, 1, acc * it)
    doAssert r === 5
    doAssert typeof(r) is int

  block:
    let r = fold(5, 10, acc + it)
    doAssert r === 15

  block:
    let r = fold(Int[7](), 1, acc * it)
    doAssert r === 7
    doAssert typeof(r) is int

  block:
    let r = fold(Int[7](), 10, acc + it)
    doAssert r === 17

  block:
    let r = fold((2, 3, 4), 1, acc * it)
    doAssert r === 24

  block:
    let r = fold((Int[2](), Int[3](), Int[4]()), 1, acc * it)
    doAssert r === 24

  block:
    let d4 = 4
    let r = fold((Int[2](), d4, Int[3]()), 1, acc * it)
    doAssert r === 24
    doAssert typeof(r) is int

  block:
    let r = fold(((2, 3), 4), 1, acc * it)
    doAssert r === 24

  block:
    let d7 = 7
    let r = fold(((Int[2](), d7), Int[3]()), 1, acc * it)
    doAssert r === 42

  block:
    let r = fold((Int[7](),), 1, acc * it)
    doAssert r === 7

  block:
    let r = fold((1, 2, 3, 4, 5), 0, acc + it)
    doAssert r === 15

  block:
    let r = fold((3, 7, 2, 9, 1), 0, max(acc, it))
    doAssert r === 9
  echo "  Fold: 12 cases OK"

# ═══════════════════════════════════════════════════════════════
#  product tests
# ═══════════════════════════════════════════════════════════════

proc runProductTests =
  block:
    const r = product(42)
    doAssert r === 42
    doAssert typeof(r) is int

  block:
    const r = product(Int[5]())
    doAssert r === 5
    doAssert typeof(r) is Int

  block:
    let r = product((2, 3, 4))
    doAssert r === 24
    doAssert typeof(r) is int

  block:
    const r = product((Int[2](), Int[3](), Int[4]()))
    doAssert r === 24

  block:
    let d4 = 4
    let r = product((Int[2](), d4, Int[3]()))
    doAssert r === 24

  block:
    const r = product(((Int[2](), Int[3]()), Int[4]()))
    doAssert r === 24

  echo "  Product: 6 cases OK"

# ═══════════════════════════════════════════════════════════════
#  max tests
# ═══════════════════════════════════════════════════════════════

proc runMaxTests =
  block:
    let r = max(42)
    doAssert r === 42

  block:
    const r = max(Int[42]())
    doAssert r === 42

  block:
    let r = max((3, 7, 2, 9, 1))
    doAssert r === 9

  block:
    let r = max((Int[3](), Int[7](), Int[2](), Int[9](), Int[1]()))
    doAssert r === 9

  block:
    let d7 = 7; let d9 = 9
    let r = max((Int[3](), d7, Int[2](), d9, Int[1]()))
    doAssert r === 9

  block:
    let r = max((-5, -2, -9, -1))
    doAssert r === -1

  block:
    let r = max(((3, 7), (9, 1)))
    doAssert r === 9
  echo "  Max: 7 cases OK"

# ═══════════════════════════════════════════════════════════════
#  min tests
# ═══════════════════════════════════════════════════════════════

proc runMinTests =
  block:
    let r = min(42)
    doAssert r === 42

  block:
    const r = min(Int[42]())
    doAssert r === 42

  block:
    let r = min((3, 7, 2, 9, 1))
    doAssert r === 1

  block:
    let r = min((Int[3](), Int[7](), Int[2](), Int[9](), Int[1]()))
    doAssert r === 1

  block:
    let d7 = 7; let d9 = 9
    let r = min((Int[3](), d7, Int[2](), d9, Int[1]()))
    doAssert r === 1

  block:
    let r = min((-5, -2, -9, -1))
    doAssert r === -9

  block:
    let r = min(((3, 7), (9, 1)))
    doAssert r === 1
  echo "  Min: 7 cases OK"

# ═══════════════════════════════════════════════════════════════
#  Int[N] comparison overloads (==, <=, >=)
# ═══════════════════════════════════════════════════════════════

proc runIntCmpTests =
  block:
    doAssert Int[5]() === Int[5]()
    doAssert not (Int[5]() === Int[8]())
    doAssert Int[5]() === 5
    doAssert 5 === Int[5]()
    doAssert Int[5]() !== 8
    doAssert 8 !== Int[5]()
    doAssert Int[5]() <= 8
    doAssert Int[5]() >= 3
    doAssert Int[5]() <= Int[8]()
    doAssert Int[5]() >= Int[3]()
  echo "  Int[N] cmp: 10 checks OK"

# ═══════════════════════════════════════════════════════════════
#  prefix_scanIt / suffix_scanIt
# ═══════════════════════════════════════════════════════════════

proc runScanTests =
  block:
    doAssert prefix_scanIt(5, 1, acc * it) === 1
  block:
    doAssert prefix_scanIt(Int[5](), 1, acc * it) === 1
  block:
    doAssert prefix_scanIt((2, 3, 4), 1, acc * it) === (1, 2, 6)

  block:
    doAssert suffix_scanIt(5, 1, acc * it) === 1
  block:
    doAssert suffix_scanIt(Int[5](), 1, acc * it) === 1
  block:
    doAssert suffix_scanIt((2, 3, 4), 1, acc * it) === (12, 4, 1)

  block:
    # prefix_scanIt with max operation
    doAssert prefix_scanIt((3, 5, 2), low(int), max(acc, it)) === (low(int), 3, 5)
  block:
    # prefix_scanIt with addition (sum so far)
    doAssert prefix_scanIt((1, 2, 3), 0, acc + it) === (0, 1, 3)
  block:
    # suffix_scanIt single-element tuple with addition
    doAssert suffix_scanIt((5,), 10, acc + it) === (10,)
  block:
    # suffix_scanIt with sum
    doAssert suffix_scanIt((1, 2, 3), 0, acc + it) === (5, 3, 0)
  echo "  Scan: 10 cases OK"

# ═══════════════════════════════════════════════════════════════
#  prefix_product / suffix_product
# ═══════════════════════════════════════════════════════════════

proc runProductScanTests =
  block:
    doAssert prefix_product(5) === 1
  block:
    doAssert prefix_product(Int[5]()) === 1
  block:
    doAssert prefix_product((2, 3, 4)) === (1, 2, 6)
  block:
    let pp = prefix_product((Int[4](), Int[8]()))
    doAssert pp[0] === 1 and pp[1] === 4

  block:
    doAssert suffix_product(5) === 1
  block:
    doAssert suffix_product(Int[5]()) === 1
  block:
    doAssert suffix_product((2, 3, 4)) === (12, 4, 1)
  block:
    let sp = suffix_product((Int[4](), Int[8]()))
    doAssert sp[0] === 8 and sp[1] === 1
  # ── Nested tuple prefix_product ──
  block:
    doAssert prefix_product(((4, 1), (8, 8))) === ((1, 4), (4, 32))
  block:
    let mr = 4; let mpT = 8; let kc = 8
    let p = prefix_product(((mr, 1), (mpT, kc)))
    doAssert p[0][0] === 1 and p[0][1] == mr
    doAssert p[1][0] == mr and p[1][1] == mr * mpT
  # ── Nested tuple suffix_product ──
  block:
    doAssert suffix_product(((4, 1), (8, 8))) === ((64, 64), (8, 1))

  # ── prefix_scanIt with nested tuples ──
  block:
    let s = prefix_scanIt(((4, 1), (8, 8)), Int[1](), acc * it)
    doAssert s[0][0] === 1 and s[0][1] == 4
    doAssert s[1][0] == 4 and s[1][1] == 32


  # ── Nested tuple prefix_product ──
  block:
    doAssert prefix_product(((4, 1), (8, 8))) === ((1, 4), (4, 32))
  block:
    let mr = 4; let mpT = 8; let kc = 8
    let p = prefix_product(((mr, 1), (mpT, kc)))
    doAssert p[0][0] === 1 and p[0][1] == mr
    doAssert p[1][0] == mr and p[1][1] == mr * mpT
  # ── Nested tuple suffix_product ──
  block:
    doAssert suffix_product(((4, 1), (8, 8))) === ((64, 64), (8, 1))

  # ── suffix_scanIt with nested tuples ──
  block:
    let s = suffix_scanIt(((4, 1), (8, 8)), Int[1](), acc * it)
    doAssert s[0][0] === 64 and s[0][1] === 64
    doAssert s[1][0] === 8 and s[1][1] === 1

  # ── prefix_scanIt with addition (prefix sum) on nested tuples ──
  block:
    let s = prefix_scanIt(((1, 2), (3, 4)), 0, acc + it)
    doAssert s[0][0] === 0 and s[0][1] === 1
    doAssert s[1][0] === 3 and s[1][1] === 6

  # ── suffix_scanIt with addition (suffix sum) on nested tuples ──
  block:
    let s = suffix_scanIt(((1, 2), (3, 4)), 0, acc + it)
    doAssert s[0][0] === 9 and s[0][1] === 7
    doAssert s[1][0] === 4 and s[1][1] === 0

  echo "  Nested tuple scan: 8 cases OK"
  echo "  Product scan: 13 cases OK"
# ═══════════════════════════════════════════════════════════════
#  Mixed static/dynamic scans
#  (N, C, H, W) — N dynamic, C/H/W are Int[N] (static)
#  Since suffix folds from the right and all inner dims are
#  Int[N], the starting accumulator (int literal 1) stays
#  compile-time throughout, making ALL outputs Int[N].
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
#  IntOrIntTuple type class checks
# ═══════════════════════════════════════════════════════════════

proc runTypeClassTests =
  block:
    doAssert 2 is IntOrIntTuple
  block:
    doAssert (2, 3) is IntOrIntTuple
  block:
    doAssert ((2, 3), 4) is IntOrIntTuple
  block:
    doAssert (5,) is IntOrIntTuple
  block:
    doAssert "hello" isnot IntOrIntTuple
  block:
    doAssert 3.14 isnot IntOrIntTuple
  block:
    doAssert @[1, 2, 3] isnot IntOrIntTuple
  echo "  Type class: 7 checks OK"

# ═══════════════════════════════════════════════════════════════
#  product/max/min — convenience wrappers (fold-based)
# ═══════════════════════════════════════════════════════════════

proc runFoldWrapperTests =
  block:
    doAssert product(7) === 7
  block:
    doAssert product((3, 4, 5)) === 60
  block:
    doAssert product(((2, 3), 4)) === 24
  block:
    doAssert max(42) === 42
  block:
    doAssert max((3, 7, 2)) === 7
  block:
    doAssert min(42) === 42
  block:
    doAssert min((5, 3, 8)) === 3
  echo "  Fold wrappers: 7 cases OK"
# ═══════════════════════════════════════════════════════════════
#  flatten / concat — extra cases
# ═══════════════════════════════════════════════════════════════

proc runFlattenConcatTests =
  block:
    doAssert flatten((4, 8, 2)) === (4, 8, 2)
  block:
    doAssert flatten((4, (1, 8), 2)) === (4, 1, 8, 2)
  block:
    doAssert flatten(((1, 2), (3, (4, 5)), 6)) === (1, 2, 3, 4, 5, 6)
  block:
    doAssert flatten(42) === 42
  block:
    type Shape2d = (int, int)
    const s: Shape2d = (4, 8)
    doAssert flatten(s) === (4, 8)
  block:
    let t = (4, (1, 8), 2)
    doAssert flatten(t) === (4, 1, 8, 2)
  block:
    let a = 4; let b = (1, 8)
    doAssert flatten((a, b)) === (4, 1, 8)
  block:
    # flatten of proc return value
    proc foo(): (int, (int, int), (int, int, int)) =
      result[0] = 0; result[1][0] = 1; result[1][1] = 2
      result[2][0] = 3; result[2][1] = 4; result[2][2] = 5
    doAssert flatten(foo()) === (0, 1, 2, 3, 4, 5)
  block:
    # concat with type alias
    type Shape2d = (int, int)
    const s: Shape2d = (4, 8)
    doAssert concat(s, 1) === (4, 8, 1)
  echo "  Flatten/concat: 9 cases OK"

proc runMixedStaticDynamicTests =
  block:
    let dN = 2
    let shape = (dN, Int[3](), Int[8](), Int[8]())
    let stride = suffix_product(shape)
    doAssert stride[0] === 3*8*8 and stride[1] === 8*8 and stride[2] === 8 and stride[3] === 1
    doAssert not isConst(shape[0])
    doAssert isConst(shape[1])
    doAssert isConst(shape[2])
    doAssert isConst(shape[3])
    doAssert isConst(stride[0])
    doAssert isConst(stride[1])
    doAssert isConst(stride[2])
    doAssert isConst(stride[3])

  block:
    let dN = 2
    let shape = (dN, Int[3](), Int[8](), Int[8]())
    let stride = prefix_product(shape)
    doAssert stride[0] === 1 and stride[1] === dN and stride[2] === dN*3 and stride[3] === dN*3*8
  echo "  Mixed static/dynamic: 2 cases OK"

# ═══════════════════════════════════════════════════════════════
#  zip2_by — guided zip for rank-2 tuples
# ═══════════════════════════════════════════════════════════════
proc runZip2ByTests =
  block:
    ## Terminal guide — pair pass-through
    const t = (Int[2](), Int[3]())
    const guide = 99
    let r = zip2_by(t, guide)
    doAssert r === (Int[2](), Int[3]())
  block:
    ## Tuple guide with 2 terminals — basic split
    let t = ((Int[2](), Int[3]()), (Int[4](), Int[5]()))
    let guide = (1, 2)
    let r = zip2_by(t, guide)
    doAssert r === ((Int[2](), Int[4]()), (Int[3](), Int[5]()))
  block:
    ## Nested guide — guide = (X, (X, X))
    let t = ((Int[2](), Int[3]()), ((Int[4](), Int[5]()), (Int[6](), Int[7]())))
    let guide = (1, (2, 3))
    let r = zip2_by(t, guide)
    let expected = ((Int[2](), (Int[4](), Int[6]())), (Int[3](), (Int[5](), Int[7]())))
    doAssert r === expected
  block:
    ## Guide shorter than t — trailing goes to group 1
    let t = ((Int[2](), Int[3]()), (Int[4](), Int[5]()), Int[6]())
    let guide = (1, 2)
    let r = zip2_by(t, guide)
    let expected = ((Int[2](), Int[4]()), (Int[3](), Int[5](), Int[6]()))
    doAssert r === expected
  block:
    ## Guide matches t exactly — no trailing
    let t = ((Int[2](), Int[3]()), (Int[4](), Int[5]()))
    let guide = (1, 2)
    let r = zip2_by(t, guide)
    let expected = ((Int[2](), Int[4]()), (Int[3](), Int[5]()))
    doAssert r === expected
  block:
    ## MoYe.jl tuple_alg test: chars as stand-ins for Int[N]
    let t = ((1, 10), ((2, 20), (3, 30)), 100)
    let guide = (0, (0, 0))
    let r = zip2_by(t, guide)
    let expected = ((1, (2, 3)), (10, (20, 30), 100))
    doAssert r === expected, "got " & $r & " expected " & $expected
  block:
    ## Rank-1 input (single pair)
    const t = (Int[2](), Int[3]())
    const guide = 0  # terminal
    let r = zip2_by(t, guide)
    doAssert r === (Int[2](), Int[3]())
  # ── zip2_by doc examples ──
  block:
    # Flat scalar guide: each t[i] is a pair, split pair-wise
    doAssert zip2_by(((Int[2](), Int[3]()), (Int[4](), Int[5]())), (1, 2)) ===
      ((Int[2](), Int[4]()), (Int[3](), Int[5]()))
  block:
    # Mixed guide: scalar splits a pair, tuple recurses into sub-tuple
    doAssert zip2_by(((Int[2](), Int[3]()), ((Int[4](), Int[5]()), (Int[6](), Int[7]()))), (1, (2, 3))) ===
      ((Int[2](), (Int[4](), Int[6]())), (Int[3](), (Int[5](), Int[7]())))
  block:
    # Guide shorter than t — trailing appended to group 1
    doAssert zip2_by(((Int[2](), Int[3]()), (Int[4](), Int[5]()), Int[99]()), (1, 2)) ===
      ((Int[2](), Int[4]()), (Int[3](), Int[5](), Int[99]()))

  echo "  zip2_by: 10 cases OK"
#  Run all
# ═══════════════════════════════════════════════════════════════


proc runMapZipWithTests =
  block:
    let r = mapModesWith((2, 4, 6)): it * 2
    doAssert r === (4, 8, 12)
  block:
    let r = mapModesWith((Int[2](), Int[4]())): it * 3
    doAssert r === (6, 12)
  block:
    # product_each: product of each top-level element
    let r = mapModesWith(((2,2), (2,8))): product(it)
    doAssert r === (4, 16)
  block:
    let r = zipModesWith((2, 4), (10, 20)): it_a + it_b
    doAssert r === (12, 24)
  block:
    let r = zipModesWith((2, 4, 6), (10, 20)): it_a + it_b
    doAssert r === (12, 24, 6)
  block:
    let r = zipModesWith((2, 4), (10, 20, 30)): it_a + it_b
    doAssert r === (12, 24, 30)
  block:
    let r = zipModesWith((7, 10, 15), (3, 4, 6)): ceil_div(it_a, it_b)
    doAssert r === (3, 3, 3)
  echo "  mapModesWith/zipModesWith: 7 checks OK"

proc runMapLeavesWithPlainIntTests =
  ## mapLeavesWith on scalar int — literal, let, const, proc chain.
  block:
    ## int literal — doubled
    let r = mapLeavesWith(42, it * 2)
    doAssert r === 84
    doAssert typeof(r) is int

  block:
    ## int literal — identity
    let r = mapLeavesWith(99, it)
    doAssert r === 99

  block:
    ## let variable
    let x = 10
    let r = mapLeavesWith(x, it + 5)
    doAssert r === 15

  block:
    ## const variable
    const y = 7
    let r = mapLeavesWith(y, it * 3)
    doAssert r === 21

  block:
    ## proc chain — compile-time foldable
    func double(x: int): int = x * 2
    let r = mapLeavesWith(double(5), it + 1)
    doAssert r === 11

  block:
    ## proc chain — runtime
    proc add(a, b: int): int = a + b
    let v = 3
    let r = mapLeavesWith(add(v, 4), it * 10)
    doAssert r === 70
  echo "  mapLeavesWith (plain int): 6 cases OK"

proc runTests =
  echo "── Int tuples ──"
  runFoldTests()
  runProductTests()
  runMaxTests()
  runMinTests()
  runIntCmpTests()
  runScanTests()
  runProductScanTests()
  runTypeClassTests()
  runFoldWrapperTests()

  runFlattenConcatTests()
  runMixedStaticDynamicTests()
  runZip2ByTests()
  runMapZipWithTests()
  runMapLeavesWithPlainIntTests()
  echo "ALL INT_TUPLES TESTS PASSED"

when isMainModule:
  runTests()
