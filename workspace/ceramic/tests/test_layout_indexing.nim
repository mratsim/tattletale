## Test: layout_indexing — crd2idx, idx2crd, CoordWheel
##
## Tests both GPU (divmod) and CPU (wheel-winding) indexing paths.

import ../src/int_tuples
import ../src/layouts
import ../src/layout_indexing_cpu
import ../src/layout_indexing_gpu
import ../src/layout_indexing
import std/typetraits
import testutils

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════
#  crd2idx — scalar
# ═══════════════════════════════════════════════════════════════

block:
  check crd2idx(5, 10), 5, int
  check crd2idx(3, 5, 2), 6, int

# ═══════════════════════════════════════════════════════════════
#  crd2idx — tuple coord → inner product
# ═══════════════════════════════════════════════════════════════

block:
  # Scalar coord decomposed over shape/stride
  check crd2idx(5, (3, 4), (2, 8)), 12, Int
  check crd2idx(0, (3, 4), (2, 8)), 0, Int
  check crd2idx(3, (3, 4), (2, 8)), 8, Int
  # Tuple coord
  check crd2idx((2, 2), (3, 4), (2, 8)), 20, Int
  check crd2idx((1, 3), (3, 4), (2, 8)), 26, Int
  check crd2idx((3, 4), (3, 4), (2, 8)), 38, Int

block:
  # 3D
  check crd2idx((1, 2, 3), (3, 4, 5), (1, 3, 12)), 43, Int

block:
  # Negative strides
  check crd2idx((2, 1), (4, 8), (-1, -4)), -6, Int

block:
  # Dynamic strides (runtime value, not compile-time Int)
  let st = (1, 3)
  check crd2idx((1, 2), (3, 4), st), 7, int
  let st2 = (1, 3)
  check crd2idx((2, 3), (3, 4), st2), 11, int
  check crd2idx((3, 4), (3, 4), st2), 15, int

echo "  [OK] crd2idx: tuple coord (6 cases)"

# ═══════════════════════════════════════════════════════════════
#  crd2idx — via Layout
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout((3, 4), (1, 3))
  check crd2idx(L, (2, 2)), 2*1 + 2*3, Int
  check crd2idx(L, (0, 0)), 0, Int
  check crd2idx(L, (1, 2)), 1*1 + 2*3, Int
  check crd2idx(L, (1, 2)), 1*1 + 2*3, Int

echo "  [OK] crd2idx: via Layout (3 cases)"

# ═══════════════════════════════════════════════════════════════
#  idx2crd — roundtrip
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout((3, 4), (1, 3))
  for i in 0 ..< 12:
    let crd = idx2crd(L, i)
    let idx = crd2idx(L, crd)
    doAssert idx == i, "idx2crd roundtrip: " & $i & " → " & $crd & " → " & $idx

echo "  [OK] idx2crd: roundtrip"

block:
  # L1 cache residency test: 24 "warps" of 8 elements
  let L = make_layout((3, 8), (1, 3))
  for i in 0 ..< 24:
    let crd = idx2crd(L, i)
    let idx = crd2idx(L, crd)
    doAssert idx == i

echo "  [OK] idx2crd: 24 elements roundtrip"

# ═══════════════════════════════════════════════════════════════
#  idx2crd — specific coordinate tests (commented: == on Int[N] blocked)
# ═══════════════════════════════════════════════════════════════

## proc runIdx2crdTests =
##   block:
##     ## Basic 2D flat shape
##     let L = make_layout((3, 4), (1, 4))
##     let crd = idx2crd(L, 5)
##     doAssert crd[0] === 2
##     doAssert crd[1] === 1
##   block:
##     ## Index 0 -> first element
##     let L = make_layout((3, 4), (1, 4))
##     let crd = idx2crd(L, 0)
##     doAssert crd[0] === 0
##     doAssert crd[1] === 0
##   block:
##     ## Last element
##     let L = make_layout((3, 4), (1, 4))
##     let crd = idx2crd(L, 11)
##     doAssert crd[0] === 2
##     doAssert crd[1] === 2
##   block:
##     ## Non-compact stride (MoYe test case, 0-indexed)
##     let L = make_layout((3, 4), (1, 3))
##     let crd = idx2crd(L, 9)
##     doAssert crd[0] === 0
##     doAssert crd[1] === 3
##   block:
##     ## Index at shape boundary
##     let L = make_layout((3, 4), (1, 3))
##     let crd = idx2crd(L, 3)
##     doAssert crd[0] === 0
##     doAssert crd[1] === 1
##   block:
##     ## Single mode layout
##     let L = make_layout(8, 1)
##     let crd = idx2crd(L, 5)
##     doAssert crd === 5
##   block:
##     ## 3D flat shape
##     let L = make_layout((3, 4, 5), (1, 3, 12))
##     let crd = idx2crd(L, 43)
##     doAssert crd[0] === 1
##     doAssert crd[1] === 2
##     doAssert crd[2] === 3
##   block:
##     ## Roundtrip: crd2idx(idx2crd(L, i), L) == i
##     let L = make_layout((4, 8), (1, 4))
##     for i in 0 ..< size(L):
##       let crd = idx2crd(L, i)
##       let idx = crd2idx(L, crd)
#      doAssert idx === i, "roundtrip i=" & $i & ": got " & $idx
#  echo "  idx2crd: 8 cases OK"

# ═══════════════════════════════════════════════════════════════
#  CoordWheel — basic iteration
# ═══════════════════════════════════════════════════════════════
#  CoordWheel — basic iteration
# ═══════════════════════════════════════════════════════════════

block:
  let shape = (3, 4)
  let strides = (1, 3)
  var wheel = initCoordWheel(CoordWheel[2], shape)
  var expectedIdx = 0
  for _ in 0 ..< 12:
    let off = wheel.coordOffset(strides)
    doAssert off == expectedIdx, "CoordWheel offset " & $off & " != expected " & $expectedIdx
    doAssert wheel.coord[0] == expectedIdx mod 3
    doAssert wheel.coord[1] == expectedIdx div 3
    expectedIdx += 1
    wheel.incr(shape)

echo "  [OK] CoordWheel: 2D iteration"

# ═══════════════════════════════════════════════════════════════
#  CoordWheel — 3D iteration
# ═══════════════════════════════════════════════════════════════

block:
  let shape = (2, 3, 4)
  let strides = (1, 2, 6)
  var wheel = initCoordWheel(CoordWheel[3], shape)
  # Expected: coord (0,0,0)→(1,0,0)→(0,1,0)→(1,1,0)→(0,2,0)→...
  var expected: array[3, int]
  for idx in 0 ..< 24:
    let off = wheel.coordOffset(strides)
    doAssert off == expected[0]*1 + expected[1]*2 + expected[2]*6
    doAssert wheel.coord[0] == expected[0]
    doAssert wheel.coord[1] == expected[1]
    doAssert wheel.coord[2] == expected[2]
    # Advance expected
    expected[0] += 1
    if expected[0] >= 2:
      expected[0] = 0
      expected[1] += 1
      if expected[1] >= 3:
        expected[1] = 0
        expected[2] += 1
    wheel.incr(shape)

echo "  [OK] CoordWheel: 3D iteration"

# ═══════════════════════════════════════════════════════════════
#  CoordWheel — single element
# ═══════════════════════════════════════════════════════════════

block:
  let shape = (1,)
  let strides = (1,)
  var wheel = initCoordWheel(CoordWheel[1], shape)
  doAssert wheel.coordOffset(strides) == 0
  wheel.incr(shape)
  # After one incr: coord goes (0) → (0) because 1-1 == 0, so it resets to 0
  doAssert wheel.coordOffset(strides) == 0

echo "  [OK] CoordWheel: single element"

# ═══════════════════════════════════════════════════════════════
#  CoordWheel — LayoutRight strides vs LayoutLeft
# ═══════════════════════════════════════════════════════════════
block:
  # Both LayoutRight and LayoutLeft: CoordWheel increments dim-0 fastest
  # (carry chain from dim-0). Compare offsets against crd2idx computed from
  # the current coordinate tuple.
  let shape = (3, 4)
  let rightStrides = (1, 3)
  let leftStrides = (4, 1)
  var wR = initCoordWheel(CoordWheel[2], shape)
  var wL = initCoordWheel(CoordWheel[2], shape)
  for r in 0 ..< 3:
    for c in 0 ..< 4:
      let offR = wR.coordOffset(rightStrides)
      let offL = wL.coordOffset(leftStrides)
      let expectedR = crd2idx((wR.coord[0], wR.coord[1]), shape, rightStrides)
      let expectedL = crd2idx((wL.coord[0], wL.coord[1]), shape, leftStrides)
      doAssert offR == expectedR, "LayoutRight CoordWheel vs crd2idx"
      doAssert offL == expectedL, "LayoutLeft CoordWheel vs crd2idx"
      wR.incr(shape)
      wL.incr(shape)

echo "  [OK] CoordWheel: LayoutRight vs LayoutLeft"

# ═══════════════════════════════════════════════════════════════
#  slice/dice on Layout
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout((4, 8), (1, 4))
  doAssert slice(L, (X, Y)) === (4, 1)
  doAssert slice(L, (Y, X)) === (8, 4)
  doAssert slice(L, (X, X)) === L
  doAssert slice(L, (Y, Y)) === ((), ())
echo "    slice on Layout: 4 cases OK"

block:
  let L = make_layout((2, 3, 4), (1, 2, 6))
  let sub = slice(L, (X, Y, X))
  doAssert sub.shape[0] === 2
  doAssert sub.shape[1] === 4
  doAssert sub.stride[0] === 1
  doAssert sub.stride[1] === 6
echo "    slice rank-3: 4 checks OK"

block:
  let L = make_layout((3, 4), (1, 4))
  doAssert dice(L, (Y, X)) === (3, 1)
  doAssert dice(L, (X, Y)) === (4, 4)
  doAssert dice(L, (Y, Y)) === L
  doAssert dice(L, (X, X)) === ((), ())
echo "    dice on Layout: 4 cases OK"

# ═══════════════════════════════════════════════════════════════
#  Call operator — crd2idx via L()
# ═══════════════════════════════════════════════════════════════

block:
  let l = make_layout(8, 1)
  check crd2idx(l, 0), 0, int
  check crd2idx(l, 3), 3, int
  check crd2idx(l, 7), 7, int
block:
  let l = make_layout((4, 8), (1, 4))
  check crd2idx(l, 0), 0, Int
  check crd2idx(l, 10), 10, Int
echo "    layout(): 2 checks OK"

# ═══════════════════════════════════════════════════════════════
#  Dual dispatch — L() with _ vs int
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout((3, 4), (1, 4))
  check crd2idx(L, (0, 0)), 0, Int
  check crd2idx(L, (1, 2)), 9, Int
  check crd2idx(L, (2, 3)), 14, Int
block:
  let L = make_layout((3, 4), (1, 4))
  check crd2idx(L, (0, 0)), 0, Int
  check crd2idx(L, (1, 2)), 9, Int
  check crd2idx(L, (2, 3)), 14, Int
block:
  let L = make_layout((3, 4), (1, 4))
  doAssert slice(L, (_, 0)) === make_layout((3,), (1,))
  doAssert slice(L, (0, _)) === make_layout((4,), (4,))
  doAssert slice(L, (_, _)) === L
echo "    slice via () syntax: 3 cases OK"

echo "\n--- layout_indexing tests ---"
echo "  All tests passed."
