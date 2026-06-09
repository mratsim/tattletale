# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for CuTe-compatible int_tuples + layouts.
##
## Convention:
##   const C2 = 2; C4 = 4 — compile-time Int[N] (static)
##   let  d2 = 2; d4 = 4 — runtime int (dynamic)
##
## Reference:
##   - CuTe C++: layout.hpp, composition.cpp
##   - Python: tensor-layouts/tests/layouts.py
##   - POC: poc_coalesce.nim, poc_flatten.nim

import std/macros
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra

# ═══════════════════════════════════════════════════════════════
#  make_layout — shape + stride, const correctness, stride order
# ═══════════════════════════════════════════════════════════════

proc runMakeLayoutTests* =
  let d1 = 1

  # ═══════════════════════════════════════════════════════════════
  #  1. Scalar layouts — (int, int) → Layout[int, int]
  # ═══════════════════════════════════════════════════════════════

  block:  # (1,0)
    doAssert make_layout(1, 0) === (1, 0)

  block:  # (1,1)
    doAssert make_layout(1, 1) === (1, 1)

  # ═══════════════════════════════════════════════════════════════
  #  2. Const vs let scalars — verify isConst behavior
  # ═══════════════════════════════════════════════════════════════

  block:
    const C4 = 4
    let l = make_layout(C4, 0)
    doAssert l === (4, 0)
    doAssert isConst(l.shape)
    doAssert isConst(l.stride)

  block:
    const C4 = 4
    let l = make_layout(C4)
    doAssert l === (4, 1)
    doAssert isConst(l.shape)
    doAssert isConst(l.stride)

  block:
    let d4 = 4
    doAssert make_layout(d4, 0) === (4, 0)

  # ═══════════════════════════════════════════════════════════════
  #  3. Const tuple — all fields Int[N]
  # ═══════════════════════════════════════════════════════════════

  block:
    const C2 = 2; const C4 = 4
    let l = make_layout((C2, C4), (1, 2))
    doAssert l === ((2, 4), (1, 2))
    doAssert isConst(l.shape[0])
    doAssert isConst(l.shape[1])
    doAssert isConst(l.stride[0])
    doAssert isConst(l.stride[1])

  block:
    const C2 = 2; const C4 = 4
    let l = make_layout((C2, C4))
    doAssert l === ((2, 4), (1, 2))
    doAssert isConst(l.shape[0])
    doAssert isConst(l.shape[1])
    doAssert isConst(l.stride[0])
    doAssert isConst(l.stride[1])

  # ═══════════════════════════════════════════════════════════════
  #  4. Mixed const/let tuple
  # ═══════════════════════════════════════════════════════════════

  block:
    const C2 = 2; const C4 = 4; let d2 = 2
    let l = make_layout((d2, C4), (1, 2))
    doAssert l === ((2, 4), (1, 2))
    doAssert not isConst(l.shape[0])
    doAssert isConst(l.shape[1])
    doAssert isConst(l.stride[0])
    doAssert isConst(l.stride[1])

  block:
    const C2 = 2; const C4 = 4; let d2 = 2
    let l = make_layout((d2, C4))
    doAssert l === ((2, 4), (1, 2))
    doAssert not isConst(l.shape[0])
    doAssert isConst(l.shape[1])
    doAssert isConst(l.stride[0])     # always 1 (Int[1]) for column-major
    doAssert not isConst(l.stride[1])

  # ═══════════════════════════════════════════════════════════════
  #  5. Let tuple — all dynamic
  # ═══════════════════════════════════════════════════════════════

  block:
    let d2 = 2; let d4 = 4
    let l = make_layout((d2, d4), (1, 2))
    doAssert l === ((2, 4), (1, 2))
    doAssert not isConst(l.shape[0])
    doAssert not isConst(l.shape[1])
    doAssert isConst(l.stride[0])
    doAssert isConst(l.stride[1])

  block:
    let d2 = 2; let d4 = 4
    let l = make_layout((d2, d4))
    doAssert l === ((2, 4), (1, 2))
    doAssert not isConst(l.shape[0])
    doAssert not isConst(l.shape[1])
    doAssert isConst(l.stride[0])     # always 1 (Int[1]) for column-major
    doAssert not isConst(l.stride[1])

  # ═══════════════════════════════════════════════════════════════
  #  6. Tuple assigned to variable — loses static info
  # ═══════════════════════════════════════════════════════════════

  block:
    const C2 = 2; const C4 = 4
    let shapeTuple = (C2, C4)
    let l = make_layout(shapeTuple, (1, 2))
    doAssert l === ((2, 4), (1, 2))
    doAssert not isConst(l.shape[0])
    doAssert not isConst(l.shape[1])
    doAssert isConst(l.stride[0])
    doAssert isConst(l.stride[1])

  # ═══════════════════════════════════════════════════════════════
  #  7. Tuple from function — runtime, all dynamic
  # ═══════════════════════════════════════════════════════════════

  block:
    proc mkShape(): auto = (2, 4)
    doAssert make_layout(mkShape(), (1, 2)) === ((2, 4), (1, 2))

  # ═══════════════════════════════════════════════════════════════
  #  8. static() — folds to nnkTupleConstr, preserves const
  # ═══════════════════════════════════════════════════════════════

  block:
    proc mkShape(): auto = (2, 4)
    let l = make_layout(static(mkShape()), (1, 2))
    doAssert l === ((2, 4), (1, 2))
    doAssert isConst(l.shape[0])
    doAssert isConst(l.shape[1])

  # ═══════════════════════════════════════════════════════════════
  #  9. const variable from function — also folds to nnkTupleConstr
  # ═══════════════════════════════════════════════════════════════

  block:
    proc mkShape(): auto = (2, 4)
    const shape = mkShape()
    let l = make_layout(shape, (1, 2))
    doAssert l === ((2, 4), (1, 2))
    doAssert isConst(l.shape[0])
    doAssert isConst(l.shape[1])

  # ═══════════════════════════════════════════════════════════════
  #  10. Recursive (nested) tuple — all Int[N] via literals
  # ═══════════════════════════════════════════════════════════════

  block:
    let l = make_layout(((2, 2), (2, 2)), ((1, 4), (8, 32)))
    doAssert l === (((2, 2), (2, 2)), ((1, 4), (8, 32)))
    doAssert isConst(l.shape[0][0])
    doAssert isConst(l.shape[0][1])
    doAssert isConst(l.shape[1][0])
    doAssert isConst(l.shape[1][1])

  # ═══════════════════════════════════════════════════════════════
  #  11. Recursive tuple via variable — all dynamic
  # ═══════════════════════════════════════════════════════════════

  block:
    let nestedShape = ((2, 2), (2, 2))
    let nestedStride = ((1, 4), (8, 32))
    let l = make_layout(nestedShape, nestedStride)
    doAssert l === (((2, 2), (2, 2)), ((1, 4), (8, 32)))
    doAssert not isConst(l.shape[0][0])
    doAssert not isConst(l.shape[0][1])
    doAssert not isConst(l.shape[1][0])
    doAssert not isConst(l.shape[1][1])

  # ═══════════════════════════════════════════════════════════════
  #  12. isConst standalone checks
  # ═══════════════════════════════════════════════════════════════

  block:
    const C4 = 4
    let d4 = 4
    let lConst = make_layout(C4, 0)
    let lLet = make_layout(d4, 0)
    doAssert isConst(lConst.shape)
    doAssert not isConst(lLet.shape)

  block:
    const C2 = 2; const C4 = 4
    let d2 = 2
    let lMixed = make_layout((d2, C4), (1, 2))
    doAssert not isConst(lMixed.shape[0])
    doAssert isConst(lMixed.shape[1])

  # ═══════════════════════════════════════════════════════════════
  #  13. Stride-order construction — LayoutLeft / LayoutRight
  # ═══════════════════════════════════════════════════════════════

  block:  # LayoutLeft (default): leftmost mode contiguous
    doAssert make_layout((4, 8), LayoutLeft) === ((4, 8), (1, 4))

  block:  # LayoutRight: rightmost mode contiguous
    doAssert make_layout((4, 8), LayoutRight) === ((4, 8), (8, 1))

  block:  # LayoutLeft on 3D
    doAssert make_layout((3, 4, 5), LayoutLeft) === ((3, 4, 5), (1, 3, 12))

  block:  # LayoutRight on 3D
    doAssert make_layout((3, 4, 5), LayoutRight) === ((3, 4, 5), (20, 5, 1))

  block:  # LayoutLeft mixed: static int literals, no pre-conversion
    let dN = 2
    const C3 = 3; const C8 = 8
    let l = make_layout((dN, C3, C8), LayoutLeft)
    doAssert l === ((2, 3, 8), (1, 2, 6))
    doAssert not isConst(l.shape[0])
    doAssert isConst(l.shape[1])
    doAssert isConst(l.shape[2])
    doAssert isConst(l.stride[0])     # always 1 (Int[1]) for column-major
    doAssert not isConst(l.stride[1])
    doAssert not isConst(l.stride[2])

  block:  # LayoutLeft mixed: static int literals, no pre-conversion
    let dN = 2
    const C3 = 3; const C8 = 8
    let l = make_layout((C8, C3, dN), LayoutLeft)
    doAssert l === ((8, 3, 2), (1, 8, 24))
    doAssert isConst(l.shape[0])
    doAssert isConst(l.shape[1])
    doAssert not isConst(l.shape[2])
    doAssert isConst(l.stride[0])
    doAssert isConst(l.stride[1])
    doAssert isConst(l.stride[2])

  block:  # LayoutRight mixed: static int literals, no pre-conversion
    let dN = 2
    const C3 = 3; const C8 = 8
    let l = make_layout((dN, C3, C8), LayoutRight)
    doAssert l === ((2, 3, 8), (24, 8, 1))
    doAssert not isConst(l.shape[0])
    doAssert isConst(l.shape[1])
    doAssert isConst(l.shape[2])
    doAssert isConst(l.stride[0])
    doAssert isConst(l.stride[1])
    doAssert isConst(l.stride[2])

  block:  # Custom strides mixed: all-static shape, explicit strides
    let dN = 2
    const C3 = 3; const C8 = 8
    let l = make_layout((dN, C3, C8), (1, 1, 1))
    doAssert l === ((2, 3, 8), (1, 1, 1))
    doAssert not isConst(l.shape[0])
    doAssert isConst(l.shape[1])
    doAssert isConst(l.shape[2])
    doAssert isConst(l.stride[0])
    doAssert isConst(l.stride[1])
    doAssert isConst(l.stride[2])

  echo "  make_layout: 23 cases OK"

# ═══════════════════════════════════════════════════════════════
#  Phase 2: flatten
# ═══════════════════════════════════════════════════════════════

proc runFlattenTests* =
  block:
    doAssert flatten(5) === 5

  block:
    let f1 = flatten((1, 2, 3))
    doAssert f1[0] === 1 and f1[1] === 2 and f1[2] === 3

  block:
    let f2 = flatten((1, (2, 3), 4))
    doAssert f2[0] === 1 and f2[1] === 2 and f2[2] === 3 and f2[3] === 4

  block:
    let f3 = flatten(((1, 2), (3, (4, 5))))
    doAssert f3[0] === 1 and f3[1] === 2 and f3[2] === 3 and f3[3] === 4 and f3[4] === 5

  block:
    let d4 = 4
    let f4 = flatten((d4, (5, 6)))
    doAssert f4[0] === d4 and f4[1] === 5 and f4[2] === 6

  block:
    let l = make_layout(((2,2),(2,2)), ((1,4),(8,32)))
    let f5 = flatten(l.shape)
    doAssert f5[0] === 2 and f5[1] === 2 and f5[2] === 2 and f5[3] === 2
  echo "  Flatten: 6 cases OK"

# ═══════════════════════════════════════════════════════════════
#  Phase 2: concat
# ═══════════════════════════════════════════════════════════════

proc runConcatTests* =
  block:
    # concat int + tuple
    let c1 = concat(1, (4, 8))
    doAssert c1[0] === 1 and c1[1] === 4 and c1[2] === 8

  block:
    # concat tuple + int
    let c2 = concat((4, 8), 1)
    doAssert c2[0] === 4 and c2[1] === 8 and c2[2] === 1

  block:
    # concat tuple + tuple
    let c3 = concat((4,), (8, 2))
    doAssert c3[0] === 4 and c3[1] === 8 and c3[2] === 2

  block:
    # concat int + int
    let c4 = concat(4, 8)
    doAssert c4[0] === 4 and c4[1] === 8

  block:
    # concat with dynamic variables
    let a = (4, 8)
    let b = 1
    let c5 = concat(a, b)
    doAssert c5 === (4, 8, 1)

  block:
    # concat int with dynamic tuple
    let x = 1
    let y = (4, 8)
    let c6 = concat(x, y)
    doAssert c6 === (1, 4, 8)

  block:
    # concat with Layout shapes (scalar Int[N] + scalar Int[N])
    let a = make_layout(3, 1)
    let b = make_layout(4, 3)
    let c7 = concat(a.shape, b.shape)
    doAssert c7[0] === 3 and c7[1] === 4

  block:
    # concat with Int[N] + dynamic tuple
    const C1 = 1
    let dt = (4, 8)
    let c8 = concat(C1, dt)
    doAssert c8[0] === 1 and c8[1] === 4 and c8[2] === 8
  echo "  Concat: 8 cases OK"

# ═══════════════════════════════════════════════════════════════
#  Size tests (ported from Python tensor-layouts test suite)
# ═══════════════════════════════════════════════════════════════

proc runSizeTests* =
  block:
    let l = make_layout(31, 1)
    doAssert size(l) === 31

  block:
    let l = make_layout((64, 32), (1, 128))
    doAssert size(l) === 64 * 32

  block:
    let l = make_layout((3, 8, 8, 8), (1, 3, 24, 192))
    doAssert size(l) === 3 * 8 * 8 * 8

  block:
    let l = make_layout((2, 2, 2, 2, 2), (160, 80, 40, 20, 10))
    doAssert size(l) === 32
  echo "  Size: 4 Python reference cases OK"

# ═══════════════════════════════════════════════════════════════
#  Cosize tests (ported from Python tensor-layouts test suite)
# ═══════════════════════════════════════════════════════════════
#
#  ⚠ Known discrepancies between implementations of cosize on
#  COMPOSED layouts (make_layout(l1, l2)):
#
#   1. CuTe C++ — uses hierarchical (nested) cosize. For a composed
#      layout Layout<A,B>, cosize ≈ cosize(A) * cosize(B) effectively,
#      which is incorrect when the outer layout has non-trivial stride.
#
#   2. Meta tensor-layouts (Python) — enumerates ALL offsets to compute
#      max(L(i)) + 1.  This is O(size(L)) but is the only correct
#      definition for composed layouts.  CuTe's cosize(ComposedLayout)
#      bug is explicitly documented in the Python source.
#
#   3. Our Nim (flat affine) — uses the closed-form
#      1 + sum((sh_i - 1) * |st_i|) for pure affine layouts, which
#      matches Python's affine fast-path and CuTe's rank-1 cosize.
#      We DO NOT support ComposedLayout / Swizzle — our layouts are
#      always flat/affine, so the sum formula is correct.
#
#  Example cosize values for composed layouts:
#
#   Layout                    Affine sum   Cute hier   Python enum (correct)
#   ───────────────────────   ──────────   ──────────   ─────────────────────
#   make_layout(4:1,          (4-1)*1 +    cosize(4:1)  enumerate:
#               (2,2):(1,2))   (2-1)*1 +    ×             0+0=0, 2+0=2,
#                              (2-1)*2 +    cosize(       4+0=4, 6+0=6,
#                              1 = 6        (2,2):(1,2)   0+1=1, 2+1=3,
#                                          = 4 * 4 = 16   4+1=5, 6+1=5,
#                                                         0+2=2, ...
#                                                         → max=9, cosize=10
#
#   The sum formula (ours and Python's affine) gives cosize=6,
#   CuTe hierarchical product gives 16, Python enumeration gives 10.
#   All three disagree.  CuTe's product is WRONG per the Python docs;
#   enumeration is the only universally correct method.
#
#  For our pure affine layouts the sum formula IS correct —
#  we never create ComposedLayout. The complement post-condition
#  check (1) from CuTe test_complement cannot be replicated without
#  either hierarchical product (wrong) or enumeration (expensive),
#  so we only check "doesn't crash" for complement.
# ═══════════════════════════════════════════════════════════════

proc runCosizeTests* =
  let d1 = 1
  block:
    # cosize(Layout((64, 32), (1, 128))) == 4032
    let l = make_layout((64, 32), (1, 128))
    doAssert d1 * cosize(l) === 4032

  block:
    # cosize(Layout((3, 8, 8, 8), (1, 3, 24, 192))) == 1536
    let l = make_layout((3, 8, 8, 8), (1, 3, 24, 192))
    doAssert d1 * cosize(l) === 1536

  block:
    # cosize(Layout((2, 2, 2, 2, 2), (160, 80, 40, 20, 10))) == 311
    let l = make_layout((2, 2, 2, 2, 2), (160, 80, 40, 20, 10))
    doAssert d1 * cosize(l) === 311

  block:
    # cosize(Layout(4, -1)) == 4 (uses abs stride)
    let l = make_layout(4, -1)
    doAssert d1 * cosize(l) === 4

  block:
    # cosize(Layout((2, 4), (4, -1))) == 8
    let l = make_layout((2, 4), (4, -1))
    doAssert d1 * cosize(l) === 8

  block:
    # cosize(Layout((2, 2), (-1, -2))) == 4
    let l = make_layout((2, 2), (-1, -2))
    doAssert d1 * cosize(l) === 4
  echo "  Cosize: 6 Python reference cases OK"

# ═══════════════════════════════════════════════════════════════
#  filter_zeros tests (ported from Python tensor-layouts test suite)
#  Python's `filter` only removes stride-0 modes, which is
#  exactly what filter_zeros → coalesce achieves.
# ═══════════════════════════════════════════════════════════════

proc runFilterZerosTests* =
  let d1 = 1
  block:
    let l = make_layout((64, 8, 8, 128), (8, 1, 0, 512))
    let f = filter_zeros(l)
    let fFlat = flatten(f.shape)
    doAssert d1 * fFlat[0] === 64
    doAssert d1 * fFlat[1] === 8
    doAssert d1 * fFlat[2] === 1   # stride-0 mode became size-1
    doAssert d1 * fFlat[3] === 128

  block:
    let l = make_layout((3, 8, 8, 8), (16, 0, 0, 0))
    let f = filter_zeros(l)
    let fFlat = flatten(f.shape)
    doAssert d1 * fFlat[0] === 3
    doAssert d1 * fFlat[1] === 1  # stride-0 → size-1
    doAssert d1 * fFlat[2] === 1
    doAssert d1 * fFlat[3] === 1
  echo "  filter_zeros: 2 Python reference cases OK"

# ═══════════════════════════════════════════════════════════════
#  $ — stringify
# ═══════════════════════════════════════════════════════════════

proc runStringifyTests* =
  block:
    doAssert $make_layout(4, 1) == "4:1"
  block:
    let l = make_layout((4, 8), (1, 4))
    doAssert $l == "(4, 8):(1, 4)"
  block:
    let l = make_layout(31, 1)
    doAssert $l == "31:1"
  echo "  Stringify: 3 cases OK"

# ═══════════════════════════════════════════════════════════════
#  rank — number of modes
# ═══════════════════════════════════════════════════════════════

proc runRankTests* =
  block:
    doAssert rank(make_layout(4, 1)) === 1
  block:
    doAssert rank(make_layout((4, 8), (1, 4))) === 2
  block:
    doAssert rank(make_layout(Int[4](), Int[1]())) === 1
  echo "  Rank: 3 cases OK"

# ═══════════════════════════════════════════════════════════════
#  isCompact — compactness checks
# ═══════════════════════════════════════════════════════════════

proc runIsCompactTests* =
  block:
    doAssert isCompact(make_layout((4, 8), (1, 4)))
  block:
    doAssert not isCompact(make_layout((4, 8), (8, 1)))
  block:
    const C4 = 4; const C8 = 8
    doAssert isCompact(make_layout((C4, C8), (1, 4)))
  block:
    let d4 = 4; let d8 = 8
    doAssert isCompact(make_layout((d4, d8), (1, 4)))
  echo "  isCompact: 4 cases OK"

  block:  # Function call returning static int
    func double(x: static int): static int = x * 2
    doAssert make_layout(double(3)) === (6, 1)
    doAssert isConst(make_layout(double(3)).shape)
    doAssert isConst(make_layout(double(3)).stride)
# ═══════════════════════════════════════════════════════════════
#  congruent — structural shape comparison
# ═══════════════════════════════════════════════════════════════

proc runPredicateTests* =
  # ═══════════════════════════════════════════════════════════════
  #  Predicates — congruent, weakly_congruent, compatible
  # ═══════════════════════════════════════════════════════════════

  block:
    doAssert congruent((2, 3), (4, 5))
    doAssert not congruent((2, 3), (4, 5, 6))
    doAssert congruent(((2, 3), 4), ((5, 6), 7))
  block:
    doAssert weakly_congruent(6, (2, 3))
    doAssert not weakly_congruent((2, 3), 6)
    doAssert weakly_congruent((2, 3), (4, 5))
  block:
    doAssert compatible(24, (4, 6))
    doAssert not compatible((4, 6), 24)
    doAssert compatible((2, 2, 3), (4, 3))
    doAssert not compatible(24, 32)
    doAssert compatible(24, ((2, 3), 4))
  block:
    doAssert not can_group_a_into_b((2, 3, 4), (4, 3))

  # ═══════════════════════════════════════════════════════════════
  #  Edge cases from .bak tests
  # ═══════════════════════════════════════════════════════════════
  block:
    # congruent: scalar vs 1-tuple (different structure)
    doAssert not congruent(3, (3,))
  block:
    # congruent: different values, same flat structure
    doAssert congruent((3, 128, 128), (1, 256, 64))
  block:
    # congruent: nested tuples matching (2,(3,4)) vs (5,(6,7))
    doAssert congruent((2, (3, 4)), (5, (6, 7)))
  block:
    # weakly_congruent: scalar matches any structure depth
    doAssert weakly_congruent(1, ((2, 3), (4, 5)))
  block:
    # weakly_congruent: same nesting
    doAssert weakly_congruent((2, (3, 4)), (5, (6, 7)))
  block:
    # weakly_congruent: A flatter than B
    doAssert weakly_congruent((2, 3), (5, (6, 7)))
  block:
    # weakly_congruent: A deeper than B fails
    doAssert not weakly_congruent((2, (3, 4)), (5, 6))
  block:
    # compatible: group into nested target
    doAssert compatible(24, ((2, 2), 6))
  block:
    # not compatible: nested shape into flat target fails
    doAssert not compatible(((2, 2), 3), (4, 3))
  echo "  Predicates: 21 checks OK"

# ═══════════════════════════════════════════════════════════════
#  crd2idx — coordinate to index
# ═══════════════════════════════════════════════════════════════

proc runCrd2IdxTests* =
  block:
    doAssert crd2idx(5, (3, 4), (2, 8)) === 12
  block:
    doAssert crd2idx(0, (3, 4), (2, 8)) === 0
  block:
    doAssert crd2idx(3, (3, 4), (2, 8)) === 8
  block:
    doAssert crd2idx((2, 2), (3, 4), (2, 8)) === 20
  block:
    doAssert crd2idx((1, 3), (3, 4), (2, 8)) === 26
  block:
    doAssert crd2idx((3, 4), (3, 4), (2, 8)) === 38
  block:
    # 3D coordinate lookup: 1*1 + 2*3 + 3*12 = 43
    doAssert crd2idx((1, 2, 3), (3, 4, 5), (1, 3, 12)) === 43
  block:
    # negative stride: 2*-1 + 1*-4 = -6
    doAssert crd2idx((2, 1), (4, 8), (-1, -4)) === -6
  block:
    let st = col_major_strides((3, 4))
    doAssert crd2idx((1, 2), (3, 4), st) === 7
  block:
    let st = col_major_strides((3, 4))
    doAssert crd2idx((2, 3), (3, 4), st) === 11
  block:
    let st = col_major_strides((3, 4))
    doAssert crd2idx((3, 4), (3, 4), st) === 15
  echo "  crd2idx: 11 cases OK"

# ═══════════════════════════════════════════════════════════════
#  layout[] — call operator
# ═══════════════════════════════════════════════════════════════

proc runCallOperatorTests* =
  block:
    let l = make_layout(8, 1)
    doAssert l[0] === 0
    doAssert l[3] === 3
    doAssert l[7] === 7
  block:
    let l = make_layout((4, 8), (1, 4))
    doAssert l[0] === 0
    doAssert l[10] === 10
  echo "  layout[]: 2 checks OK"

# ═══════════════════════════════════════════════════════════════
#  col_major_strides
# ═══════════════════════════════════════════════════════════════

proc runColMajorStridesTests* =
  block:
    doAssert col_major_strides(4) === 1
  block:
    doAssert col_major_strides((4, 8)) === (1, 4)
  block:
    let cm = col_major_strides((Int[4](), Int[8]())); doAssert cm[0] === 1 and cm[1] === 4
  block:
    doAssert col_major_strides((3, 4, 5)) === (1, 3, 12)
  block:
    let d3 = 3; let d4 = 4; let d5 = 5
    doAssert col_major_strides((d3, d4, d5)) === (1, 3, 12)
  echo "  col_major_strides: 5 cases OK"

# ═══════════════════════════════════════════════════════════════
#  NCHW mixed static/dynamic layout
#  (N, C, H, W) — N dynamic, C/H/W static
#  make_layout deduces column-major strides from the shape
# ═══════════════════════════════════════════════════════════════

proc runNCHWTests* =
  block:
    let dN = 2
    let sh = (dN, Int[3](), Int[8](), Int[8]())
    let st = col_major_strides(sh)
    let l = make_layout(sh, st)
    doAssert l === ((2, 3, 8, 8), (1, 2, 6, 48))
    doAssert not isConst(l.shape[0])
    doAssert isConst(l.shape[1])
    doAssert isConst(l.shape[2])
    doAssert isConst(l.shape[3])
    doAssert isConst(l.stride[0])     # always 1 (Int[1]) for column-major
    doAssert not isConst(l.stride[1])
    doAssert not isConst(l.stride[2])
    doAssert not isConst(l.stride[3])
  echo "  NCHW mixed static/dynamic: 1 case OK"
# ═══════════════════════════════════════════════════════════════
#  zip — interleave corresponding modes pairwise
# ═══════════════════════════════════════════════════════════════

proc runZipTests* =
  block:
    # Interleave two rank-2 layouts
    let a = make_layout((2, 2), (1, 2))
    let b = make_layout((3, 4), (4, 1))
    let R = zip(a, b)
    doAssert rank(R) === 2
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === ((2, 3), (1, 4)), "zip mode0: " & $m0
    doAssert m1 === ((2, 4), (2, 1)), "zip mode1: " & $m1
  block:
    # Single-element interleave (rank-1 with rank-1)
    let a = make_layout(4, 1)
    let b = make_layout(3, 1)
    let R = zip(a, b)
    doAssert rank(R) === 2
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === (4, 1), "zip m0: " & $m0
    doAssert m1 === (3, 1), "zip m1: " & $m1
  echo "  zip: 2 cases OK"


# ═══════════════════════════════════════════════════════════════
#  tile_unzip — unzip logical_divide/product result into tiles+rest
# ═══════════════════════════════════════════════════════════════
proc runTileUnzipTests* =
  block:
    ## Rank-1 layout / rank-1 tiler (both Layout terminals)
    let L = make_layout(8, 1)
    let tiler = make_layout(2, 1)
    let divided = logical_divide(L, tiler)
    let unzipped = tile_unzip(divided, tiler)
    doAssert rank(unzipped) === 2
  block:
    ## Rank-2 layout / tuple tiler (2 ints) — Python reference
    # zipped_divide(Layout((4,8)), (2,4)) -> Layout(((2,4),(2,2)), ((1,4),(2,16)))
    let L = make_layout((4, 8), (1, 4))
    let tiler = (2, 4)
    let divided = logical_divide(L, tiler)
    let unzipped = tile_unzip(divided, tiler)
    let m0 = mode(unzipped, 0)
    let m1 = mode(unzipped, 1)
    doAssert rank(unzipped) === 2, "rank: " & $rank(unzipped)
    doAssert m0 === ((2, 4), (1, 4)), "m0: " & $m0
    doAssert m1 === ((2, 2), (2, 16)), "m1: " & $m1
  block:
    ## Rank-2 layout / rank-2 tuple of Layouts
    let L = make_layout((4, 8), (1, 4))
    let tiler = (make_layout(2, 1), make_layout(4, 1))
    let divided = logical_divide(L, tiler)
    let unzipped = tile_unzip(divided, tiler)
    doAssert rank(unzipped) === 2
    doAssert size(mode(unzipped, 0)) === 8
    doAssert size(mode(unzipped, 1)) === 4
  block:
    ## Rank-2 layout / rank-1 tiler (partial tiler)
    let L = make_layout((4, 8), (1, 4))
    let tiler = (2,)
    let divided = logical_divide(L, tiler)
    let unzipped = tile_unzip(divided, tiler)
    doAssert rank(unzipped) === 2
  echo "  tile_unzip: 4 cases OK"
# ═══════════════════════════════════════════════════════════════
#  Run all
# ═══════════════════════════════════════════════════════════════
proc runTests* =
  echo "--- make_layout ---"
  runMakeLayoutTests()
  echo "--- Flatten ---"
  runFlattenTests()
  echo "--- Concat ---"
  runConcatTests()
  echo "--- Size ---"
  runSizeTests()
  echo "--- Cosize ---"
  runCosizeTests()
  echo "--- filter_zeros ---"
  runFilterZerosTests()
  echo "--- Stringify ---"
  runStringifyTests()
  echo "--- Rank ---"
  runRankTests()
  echo "--- isCompact ---"
  runIsCompactTests()
  echo "--- Predicates ---"
  runPredicateTests()
  echo "--- crd2idx ---"
  runCrd2IdxTests()
  echo "--- layout[] ---"
  runCallOperatorTests()
  echo "--- col_major_strides ---"
  runColMajorStridesTests()
  echo "--- NCHW ---"
  runNCHWTests()
  echo "--- zip ---"
  runTileUnzipTests()
  echo "--- mapModesWith/zipModesWith ---"
  block:
    # map: double each mode's stride
    let a = make_layout((2, 4), (1, 2))
    let r = mapModesWith(a):
      make_layout(it.shape, it.stride * 2)
    doAssert r.shape === (2, 4) and r.stride === (2, 4)
  block:
    # map over single-mode layout
    let a = make_layout(3, 5)
    let r = mapModesWith(a):
      make_layout(it.shape, it.stride * 10)
    doAssert r.shape === 3 and r.stride === 50
  block:
    # map over hierarchical layout: scale each mode's stride
    let a = make_layout(((2,2),(2,8)), ((1,4),(2,8)))
    let r = mapModesWith(a):
      make_layout(it.shape, it.stride.scaleBy(2))
    doAssert r.shape === ((2,2),(2,8))
    doAssert r.stride === ((2,8),(4,16))
  block:
    # zipWith: pairwise — take stride from it_b (rightover writes over)
    let a = make_layout((2, 4), (1, 2))
    let b = make_layout((3, 5), (10, 20))
    let r = zipModesWith(a, b):
      make_layout(it_a.shape, it_b.stride)
    doAssert r.shape === (2, 4) and r.stride === (10, 20)
  block:
    # zipWith: a shorter (leftover b appended)
    let a = make_layout((2,), (1,))
    let b = make_layout(((2,2),(2,8)), ((1,4),(2,8)))
    let r = zipModesWith(a, b):
      make_layout(it_a.shape, it_b.stride)
    doAssert rank(r) == 2
    # mode 0 = zip result: shape from a, stride from b mode 0
    doAssert mode(r, 0).shape === 2
    doAssert mode(r, 0).stride === (1, 4)
    # mode 1 = leftover from b (unchanged)
    doAssert mode(r, 1).shape === (2, 8)
  echo "  5 checks OK"

  echo "--- group ---"
  block:
    let a = make_layout((2, 3, 5, 7))
    let b = a.group(0, 2)
    doAssert rank(b) === 3
  block:
    let a = make_layout((2, 3, 5, 7))
    let b = a.group(0, 2)
    let c = b.group(1, 3)
    doAssert rank(c) === 2
  # block: -- blocked by tuple hash collision, pending https://github.com/nim-lang/Nim/pull/25889
  #   ## From start (B=0, E=3) — groups 3 elements into sub-tuple
  #   let a = make_layout((2, 3, 5, 7))
  #   let b = a.group(0, 3)
  #   doAssert rank(b) === 2
  block:
    let a = make_layout((10, 20, 30, 40))
    let b = a.group(1, 3)
    doAssert rank(b) === 3
  block:
    let t = (2, 3, 5, 7)
    doAssert group(t, 0, 2)[0] === (2, 3)
    doAssert group(t, 1, 3)[1] === (3, 5)
  echo "    group: 4 cases OK"

when isMainModule:
  runTests()
