# Test: crd2idx, coordinate to logical offset
# Run: nim c -r workspace/ceramic/tests/layout/t_crd2idx.nim
#
# Consolidates the crd2idx tests.
#
# Coverage:
# - tuple and scalar coordinates, nested shapes
# - alias-typed layouts, roundtrips, the call operator
# - coordinate strides: semimodule evaluation over CoordStride strides
# - all vectors cross-checked against pycute

import std/macros
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/tests/layout/layouts_testutils

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════
#  crd2idx — tuple coord → inner product
# ═══════════════════════════════════════════════════════════════

block:
  # Scalar coord decomposed over shape/stride
  check crd2idx(make_layout((3, 4), (2, 8)), 5), 12, Int
  check crd2idx(make_layout((3, 4), (2, 8)), 0), 0, Int
  check crd2idx(make_layout((3, 4), (2, 8)), 3), 8, Int
  # Tuple coord
  check crd2idx(make_layout((3, 4), (2, 8)), (2, 2)), 20, Int
  check crd2idx(make_layout((3, 4), (2, 8)), (1, 3)), 26, Int
  # A keep-mark coordinate has no index, the term drops from the sum
  # (pycute: idx2crd(None, shape) == 0s like shape)
  check crd2idx(make_layout((4, 8), (1, 4)), (2, X())), 2, Int
  check crd2idx(make_layout((4, 8), (1, 4)), (X(), 3)), 12, Int
  check crd2idx(make_layout((4, 8), (1, 4)), (X(), X())), 0, Int
  check crd2idx(make_layout((3, 4), (2, 8)), (3, 4)), 38, Int

block:
  # 3D
  check crd2idx(make_layout((3, 4, 5), (1, 3, 12)), (1, 2, 3)), 43, Int

block:
  # Negative strides
  check crd2idx(make_layout((4, 8), (-1, -4)), (2, 1)), -6, Int

block:
  # Dynamic strides (runtime value, not compile-time Int)
  let st = (1, 3)
  check crd2idx(make_layout((3, 4), st), (1, 2)), 7, int
  let st2 = (1, 3)
  check crd2idx(make_layout((3, 4), st2), (2, 3)), 11, int
  check crd2idx(make_layout((3, 4), st2), (3, 4)), 15, int

proc runCrd2idxAliasLayoutTests =
  ## Alias-typed layouts and layout-valued calls hold the destructured
  ## shape and stride bindings, the binding must reach the offset
  block:
    proc mkLayout(): auto =
      make_layout((4, 8), (1, 4))
    let L = mkLayout()
    check crd2idx(L, (2, 5)), 22, Int
    check crd2idx(mkLayout(), (2, 5)), 22, Int

proc runCrd2idxNamedTupleCoordTests =
  ## Named tuple types from call-typed coordinates, the stream walks
  ## them after field reduction, nested and grouped identifiers alike
  block:
    proc coordN(): tuple[a: tuple[x, y: int], b: int] =
      ((2, 1), 3)
    check crd2idx(make_layout(((2, 2), 3), ((1, 2), 4)), coordN()), 16, Int
    proc coord4(): tuple[p, q, r, s: int] =
      (1, 2, 0, 1)
    check crd2idx(make_layout((2, 3, 4, 2), (1, 2, 6, 24)), coord4()), 29, Int

runCrd2idxAliasLayoutTests()
runCrd2idxNamedTupleCoordTests()

echo "  [OK] crd2idx: tuple coord (9 cases)"

# ═══════════════════════════════════════════════════════════════
#  crd2idx — scalar coord into nested shape/stride
# ═══════════════════════════════════════════════════════════════

proc runCrd2idxNestedLeafTests =
  ## Scalar coord over nested shape and stride.
  ## The inner product reads the (coord, stride) leaf pairs in order,
  ## single-leaf collapse included.
  block:
    # Static nested shape and stride
    check crd2idx(make_layout(((3, 4),), ((2, 8),)), 5), 12, Int
    check crd2idx(make_layout(((3, 4),), ((2, 8),)), 7), 18, Int
    # Single-leaf nested collapses to the plain inner product
    check crd2idx(make_layout(((10,),), ((2,),)), 3), 6, Int
    # Deeply nested 1-tuple wrappers
    check crd2idx(make_layout((((3, 4),),), (((2, 8),),)), 7), 18, Int
  block:
    # Dynamic nested shape and stride
    let s = ((3, 4),)
    let st = ((2, 8),)
    check crd2idx(make_layout(s, st), 5), 12, int
    let s1 = ((10,),)
    let st1 = ((2,),)
    check crd2idx(make_layout(s1, st1), 3), 6, int
  block:
    # Nested layout indexed through the crd2idx(L, coord) wrapper
    let l = make_layout(((3, 4),), ((2, 8),))
    check crd2idx(l, 5), 12, Int
    check crd2idx(l, 7), 18, Int

runCrd2idxNestedLeafTests()
echo "  [OK] crd2idx: nested leaf shapes (8 cases)"

# ═══════════════════════════════════════════════════════════════
#  crd2idx — via Layout
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout((3, 4), (1, 3))
  check crd2idx(L, (2, 2)), 2*1 + 2*3, Int
  check crd2idx(L, (0, 0)), 0, Int
  check crd2idx(L, (1, 2)), 1*1 + 2*3, Int

echo "  [OK] crd2idx: via Layout (3 cases)"

# ═══════════════════════════════════════════════════════════════
#  crd2idx — roundtrips
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout((3, 4), (1, 3))
  for i in 0 ..< 12:
    let crd = idx2crd(L, i)
    let idx = crd2idx(L, crd)
    doAssert idx == i, "idx2crd roundtrip: " & $i & " → " & $crd & " → " & $idx

block:
  # L1 cache residency test: 24 "warps" of 8 elements
  let L = make_layout((3, 8), (1, 3))
  for i in 0 ..< 24:
    let crd = idx2crd(L, i)
    let idx = crd2idx(L, crd)
    doAssert idx == i

echo "  [OK] crd2idx: roundtrips"

# ═══════════════════════════════════════════════════════════════
#  crd2idx: coordinate strides
#
#  A coordinate-strided layout maps to a coordinate, the sum runs
#  in the semimodule: leaf * CoordStride contributes scale*leaf
#  to each codomain mode. Vectors cross-checked against pycute.
# ═══════════════════════════════════════════════════════════════

block:
  # single-axis atom: 5@0 over an 8-row shape
  # pycute has tv(3) == 15@0, a scalar coord folds to a static atom
  let t1 = make_layout((8,), (ism_coord_strides.E(5, 0),))
  let g1 = crd2idx(t1, 3)
  doAssert g1 === ism_coord_strides.E(15, 0)
  # two-axis atoms: 2@0 and 3@1 (pycute: (2,1) -> (4,3), (3,7) -> (6,21))
  let t2 = make_layout((4, 8), (ism_coord_strides.E(2, 0), ism_coord_strides.E(3, 1)))
  let a = 2; let b = 1
  let g2 = crd2idx(t2, (a, b))
  check g2[0], 4, int
  check g2[1], 3, int
  let c1 = 3; let c2 = 7
  let g3 = crd2idx(t2, (c1, c2))
  check g3[0], 6, int
  check g3[1], 21, int
  # a zero coordinate folds to the zero atom
  let z = 0
  let g0 = crd2idx(t2, (z, z))
  check g0[0], 0, int
  check g0[1], 0, int

echo "  [OK] crd2idx: coordinate stride atoms (3 cases)"

block:
  # the opA TVCrdLayout, tile-flat thread map over (M=32, K=16)
  # with pycute vectors, the tile coordinate per (thread, value) index
  let tv = make_layout(
    (((2, 2, 2, 2, 2), 2), (4, 2)),
    (((ism_coord_strides.E(2, 1), ism_coord_strides.E(1, 0),
       ism_coord_strides.E(2, 0), ism_coord_strides.E(4, 1),
       ism_coord_strides.E(4, 0)), ism_coord_strides.E(1, 1)),
      (ism_coord_strides.E(8, 0), ism_coord_strides.E(8, 1))))
  let a1 = 1; let a2 = 0; let a3 = 1; let a4 = 0; let a5 = 1
  let v = 1
  let t = 2; let r = 1
  let g1 = crd2idx(tv, (((a1, a2, a3, a4, a5), v), (t, r)))
  check g1[0], 22, int
  check g1[1], 11, int
  let b1 = 0; let b2 = 1; let b3 = 0; let b4 = 1; let b5 = 0
  let u = 3
  let g2 = crd2idx(tv, (((b1, b2, b3, b4, b5), v), (u, r)))
  check g2[0], 25, int
  check g2[1], 13, int
  let o = 0
  let g3 = crd2idx(tv, (((a1, a1, a1, a1, a1), v), (o, o)))
  check g3[0], 7, int
  check g3[1], 7, int
  let g4 = crd2idx(tv, (((o, o, o, o, o), o), (a1, o)))
  check g4[0], 8, int
  check g4[1], 0, int
  let g5 = crd2idx(tv, (((a1, o, o, o, a1), v), (u, r)))
  check g5[0], 28, int
  check g5[1], 11, int
  # a keep-mark leaf contributes zero, the other modes stand
  let g6 = crd2idx(tv, (((_, a2, a3, a4, a5), v), (t, r)))
  check g6[0], 22, int
  check g6[1], 9, int
  # a static coordinate folds into a single static atom
  const c = (((1, 0, 1, 0, 1), 1), (2, 1))
  let gs = crd2idx(tv, c)
  doAssert gs === ism_coord_strides.E(22, 0) + ism_coord_strides.E(11, 1)

echo "  [OK] crd2idx: TVCrdLayout vectors (7 cases)"

# ═══════════════════════════════════════════════════════════════
#  crd2idx — call operator dual dispatch
# ═══════════════════════════════════════════════════════════════

block:
  let l = make_layout(8, 1)
  check crd2idx(l, 0), 0, Int
  check crd2idx(l, 3), 3, Int
  check crd2idx(l, 7), 7, Int
block:
  let l = make_layout((4, 8), (1, 4))
  check crd2idx(l, 0), 0, Int
  check crd2idx(l, 10), 10, Int
block:
  let L = make_layout((3, 4), (1, 4))
  check crd2idx(L, (0, 0)), 0, Int
  check crd2idx(L, (1, 2)), 9, Int
  check crd2idx(L, (2, 3)), 14, Int
echo "  [OK] crd2idx: call operator (3 cases)"

echo "\n--- crd2idx tests ---"
echo "  All tests passed."
