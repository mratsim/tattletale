## Test: layout_indexing — crd2idx, idx2crd, CoordWheel
##
## Tests both GPU (divmod) and CPU (wheel-winding) indexing paths.

import ../src/int_tuples
import ../src/layouts
import ../src/layout_indexing_cpu
import ../src/layout_indexing_gpu

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════
#  crd2idx — scalar
# ═══════════════════════════════════════════════════════════════

block:
  doAssert crd2idx(5, 10) == 5
  doAssert crd2idx(3, 5, 2) == 6

# ═══════════════════════════════════════════════════════════════
#  crd2idx — tuple coord → inner product
# ═══════════════════════════════════════════════════════════════

block:
  # From test_layouts.nim: crd2idx with (coord, shape, stride)
  doAssert crd2idx(5, (3, 4), (2, 8)) == 12  # 5 = (1,1): (1*2 + 1*8) wrong... let me think
  # Actually 5 as int coord decomposed over (3,4),(2,8):
  #   5 mod 3 = 2, 5 div 3 = 1
  #   2*2 + 1*8 = 4 + 8 = 12 ✓
  doAssert crd2idx(0, (3, 4), (2, 8)) == 0
  doAssert crd2idx(3, (3, 4), (2, 8)) == 8
  # Tuple coord
  doAssert crd2idx((2, 2), (3, 4), (2, 8)) == 20
  doAssert crd2idx((1, 3), (3, 4), (2, 8)) == 26
  doAssert crd2idx((3, 4), (3, 4), (2, 8)) == 38

block:
  # 3D
  doAssert crd2idx((1, 2, 3), (3, 4, 5), (1, 3, 12)) == 43

block:
  # Negative strides
  doAssert crd2idx((2, 1), (4, 8), (-1, -4)) == -6

block:
  # Dynamic strides (runtime value, not compile-time Int)
  let st = (1, 3)
  doAssert crd2idx((1, 2), (3, 4), st) == 7
  let st2 = (1, 3)
  doAssert crd2idx((2, 3), (3, 4), st2) == 11
  doAssert crd2idx((3, 4), (3, 4), st2) == 15

echo "  [OK] crd2idx: tuple coord (6 cases)"

# ═══════════════════════════════════════════════════════════════
#  crd2idx — via Layout
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout((3, 4), (1, 3))
  doAssert crd2idx(L, (2, 2)) == 2*1 + 2*3
  doAssert crd2idx(L, (0, 0)) == 0
  doAssert L((1, 2)) == 1*1 + 2*3
  doAssert L[1, 2] == 1*1 + 2*3

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
#  idx2crd — rank-1
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout(12, 1)
  for i in 0 ..< 12:
    let crd = idx2crd(L, i)
    doAssert crd == i
    doAssert crd2idx(L, crd) == i

echo "  [OK] idx2crd: rank-1"

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

echo "\n--- layout_indexing tests ---"
echo "  All tests passed."
