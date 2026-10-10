# Test: layout_indexing, idx2crd and slice/dice
# Run: nim c -r workspace/ceramic/tests/layout/t_layout_indexing.nim
#
# crd2idx tests live in t_crd2idx.nim. This file covers the idx2crd
# directions (GPU divmod, shape-based, CPU wheel).
# Also covered: the non-compact inexactness and slice/dice on Layout.

import std/macros
import std/typetraits
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/tests/layout/layouts_testutils

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════
#  idx2crd — pycute absorb
# ═══════════════════════════════════════════════════════════════

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
  # roundtrip with crd2idx over the shape (compact basis), see t_crd2idx.nim
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
#  Dual dispatch — L() with _ vs int, slice form
# ═══════════════════════════════════════════════════════════════

block:
  let L = make_layout((3, 4), (1, 4))
  doAssert slice(L, (_, 0)) === make_layout((3,), (1,))
  doAssert slice(L, (0, _)) === make_layout((4,), (4,))
  doAssert slice(L, (_, _)) === L
echo "    slice via () syntax: 3 cases OK"

echo "\n--- layout_indexing tests ---"
echo "  All tests passed."
