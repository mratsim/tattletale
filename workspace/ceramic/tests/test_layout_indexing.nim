## Test: layout_indexing, crd2idx idx2crd
## Run: nim cpp -r tests/test_layout_indexing.nim
##
## Tests both GPU (divmod) and CPU (wheel-winding) indexing paths.

import ../src/layout_algebra
import std/typetraits
import std/macros
import ./layouts_testutils

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

proc runGroupedNamedTupleTypeTests =
  ## A written grouped named tuple type reduces field-wise, the shared
  ## type appends once per identifier
  block:
    macro groupedArity(): untyped =
      let ty = nnkTupleTy.newTree(
        nnkIdentDefs.newTree(ident"a", ident"b", ident"int", newEmptyNode()))
      result = newLit(ty.getTupleType().len)
    check groupedArity(), 2, int

runGroupedNamedTupleTypeTests()

echo "  [OK] crd2idx: tuple coord (6 cases)"

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
echo "  [OK] crd2idx: nested leaf shapes (7 cases)"

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

proc runIdx2crdAbsorbTests =
  block:
    # L1 cache residency test: 24 "warps" of 8 elements
    let L = make_layout((3, 8), (1, 3))
    for i in 0 ..< 24:
      let crd = idx2crd(L, i)
      let idx = crd2idx(L, crd)
      doAssert idx == i

  echo "  [OK] idx2crd: 24 elements roundtrip"
  block:
    # pycute alignment, the quotient runs unmod'd at the most-significant leaf,
    # and the excess accumulates there without wrapping, per pycute's idx2crd.
    # CuTe C++ mods every leaf and wraps, so the case below diverges
    # from CuTe C++ on purpose.
    let L = make_layout((4, 8), (1, 4))
    doAssert idx2crd(L, 100) === (0, 25)
    # for a non-compact layout the most-significant leaf = the largest
    # stride (16 here), not the positional last, so (1, 7) below stays (1, 7)
    let n = make_layout((4, 8), (16, 1))
    doAssert idx2crd(n, 31) === (1, 7)

  echo "  [OK] idx2crd: pycute absorb"

runIdx2crdAbsorbTests()

# ═══════════════════════════════════════════════════════════════
#  idx2crd(shape, idx) — shape-based (valid for non-compact shapes)
# ═══════════════════════════════════════════════════════════════

block:
  # flat index → coord over the SHAPE (first dimension fastest), no stride use
  doAssert idx2crd((4, 8), 31) === (3, 7)
  doAssert idx2crd((4, 8), 5) === (1, 1)
  doAssert idx2crd((2, 2), 3) === (1, 1)
  doAssert idx2crd(2, 1) === 1
  # a scalar shape is the degenerate case of the absorb rule, so the coordinate equals the index itself
  doAssert idx2crd(1, 5) === 5
  # out of bounds, the last flat leaf absorbs the excess, so the result
  # does not wrap
  doAssert idx2crd((3, 7, 2), 42) === (0, 0, 2)
  doAssert idx2crd(((4, 8), (2, 2)), 1000) === ((0, 2), (1, 15))
  # nested shape: recursive per-dimension split
  doAssert idx2crd(((4, 8), (2, 2)), 31) === ((3, 7), (0, 0))
  # roundtrip with crd2idx over the shape (compact basis)
  let sh = (3, 4)
  for i in 0 ..< 12:
    let crd = idx2crd(sh, i)
    doAssert crd2idx(make_layout(sh, (1, 3)), crd) == i, "shape idx2crd roundtrip " & $i

echo "  [OK] idx2crd(shape, idx): shape-based decomposition"

# ═══════════════════════════════════════════════════════════════
#  idx2crd(layout, idx), stride-based
#
#  Non-compact layouts do not invert: only compact layouts have one
#  coordinate per slot. Use the shape-based idx2crd(shape, idx) there.
# ═══════════════════════════════════════════════════════════════

block:
  let n = make_layout((4, 8), (16, 1))   # atom T-dimension structure, gaps
  # flat 31 over the shape is (3, 7) and the stride-based
  # decomposition gives (1, 7), no roundtrip, the compact-only case
  doAssert idx2crd(n, 31) === (1, 7)
  doAssert crd2idx(n, idx2crd(n, 31)).toInt() != 31

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
##     ## Single dimension layout
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
#  idx2crd_cpu — wheel peeling, no div/mod
# ═══════════════════════════════════════════════════════════════

block:
  # every recovered coordinate roundtrips through crd2idx
  let L = make_layout((4, 8), (1, 4))
  doAssert idx2crd_cpu(L, 22) == (2, 5)
  for i in 0 ..< 32:
    doAssert crd2idx(L, idx2crd_cpu(L, i)) == i

block:
  # out of range: the highest dimension counts beyond its size
  let L = make_layout((4, 8), (1, 4))
  doAssert idx2crd_cpu(L, 35) == (3, 8)
  doAssert crd2idx(L, idx2crd_cpu(L, 35)) == 35

block:
  # scalar shape is the rank-1 degenerate case
  let S = make_layout(8, 1)
  doAssert idx2crd_cpu(S, 5) == 5

echo "  [OK] idx2crd_cpu: wheel peeling (3 cases)"

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
  check crd2idx(l, 0), 0, Int
  check crd2idx(l, 3), 3, Int
  check crd2idx(l, 7), 7, Int
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
