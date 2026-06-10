# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "callOperator".}

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
import workspace/ceramic/src/layouts {.all.}
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

  # ── Nested tuple make_layout ──
  block:
    let l = make_layout(((4, 1), (8, 8)))
    doAssert $l == "((4, 1), (8, 8)):((1, 4), (4, 32))"
  block:
    let l = make_layout(((1, 4), (8, 4)))
    doAssert $l == "((1, 4), (8, 4)):((1, 1), (4, 32))"
  block:
    let mr = 4; let mpT = 8; let kc = 8
    let l = make_layout(((mr, 1), (mpT, kc)))
    doAssert l[((0, 0), (0, 0))] == 0
    doAssert l[((0, 0), (1, 0))] == mr
    doAssert l[((0, 0), (0, 1))] == mr * mpT
    doAssert l[((1, 0), (0, 0))] == 1
  block:
    let l = make_layout(((4, 1), (8, 8)), LayoutRight)
    doAssert $l == "((4, 1), (8, 8)):((64, 64), (8, 1))"

  echo "  Nested make_layout: 4 cases OK"

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
#  idx2crd — index to coordinate (on Layout)
# ═══════════════════════════════════════════════════════════════

proc runIdx2crdTests* =
  block:
    ## Basic 2D flat shape
    let L = make_layout((3, 4), (1, 4))
    let crd = idx2crd(L, 5)
    doAssert crd[0] === 2
    doAssert crd[1] === 1
  block:
    ## Index 0 -> first element
    let L = make_layout((3, 4), (1, 4))
    let crd = idx2crd(L, 0)
    doAssert crd[0] === 0
    doAssert crd[1] === 0
  block:
    ## Last element
    let L = make_layout((3, 4), (1, 4))
    let crd = idx2crd(L, 11)
    doAssert crd[0] === 2
    doAssert crd[1] === 2
  block:
    ## Non-compact stride (MoYe test case, 0-indexed)
    let L = make_layout((3, 4), (1, 3))
    let crd = idx2crd(L, 9)
    doAssert crd[0] === 0
    doAssert crd[1] === 3
  block:
    ## Index at shape boundary
    let L = make_layout((3, 4), (1, 3))
    let crd = idx2crd(L, 3)
    doAssert crd[0] === 0
    doAssert crd[1] === 1
  block:
    ## Single mode layout
    let L = make_layout(8, 1)
    let crd = idx2crd(L, 5)
    doAssert crd === 5
  block:
    ## 3D flat shape
    let L = make_layout((3, 4, 5), (1, 3, 12))
    let crd = idx2crd(L, 43)
    doAssert crd[0] === 1
    doAssert crd[1] === 2
    doAssert crd[2] === 3
  block:
    ## Roundtrip: crd2idx(idx2crd(L, i), L) == i
    let L = make_layout((4, 8), (1, 4))
    for i in 0 ..< size(L):
      let crd = idx2crd(L, i)
      let idx = crd2idx(L, crd)
      doAssert idx === i, "roundtrip i=" & $i & ": got " & $idx
  echo "  idx2crd: 8 cases OK"
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
#  zipModes — interleave corresponding modes pairwise
# ═══════════════════════════════════════════════════════════════

proc runZipTests* =
  block:
    # Interleave two rank-2 layouts
    let a = make_layout((2, 2), (1, 2))
    let b = make_layout((3, 4), (4, 1))
    let R = zipModes(a, b)
    doAssert rank(R) === 2
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === ((2, 3), (1, 4)), "zip mode0: " & $m0
    doAssert m1 === ((2, 4), (2, 1)), "zip mode1: " & $m1
  block:
    # Single-element interleave (rank-1 with rank-1)
    let a = make_layout(4, 1)
    let b = make_layout(3, 1)
    let R = zipModes(a, b)
    doAssert rank(R) === 2
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === (4, 1), "zip m0: " & $m0
    doAssert m1 === (3, 1), "zip m1: " & $m1
  echo "  zipModes: 2 cases OK"


# ═══════════════════════════════════════════════════════════════
#  mapLeavesWith — apply body to each leaf (shape, stride) pair
# ═══════════════════════════════════════════════════════════════
proc runMapLeavesWithTests* =
  block:
    ## Flat: simple
    let a = make_layout((2, 3))
    let b = mapLeavesWith(a): (it_sh * 2, it_st)
    doAssert b.shape === (4, 6)
    doAssert b.stride === (1, 2)
  block:
    ## Flat: let indirection
    let factor = 10
    let a = make_layout((2, 3))
    let b = mapLeavesWith(a): (it_sh * factor, it_st)
    doAssert b.shape === (20, 30)
    doAssert b.stride === (1, 2)
  block:
    ## Flat: multi-step
    let a = make_layout((2, 3))
    let b = mapLeavesWith(a):
      let x = it_sh * 10
      let y = it_st + 7
      (x * y, x div y)
    doAssert b.shape === (160, 270)
    doAssert b.stride === (2, 3)
  block:
    ## Nested 2-level
    let a = make_layout(((2, 3), 5), ((1, 2), 6))
    let b = mapLeavesWith(a): (it_sh * 2, it_st * 2)
    doAssert b.shape === ((4, 6), 10)
    doAssert b.stride === ((2, 4), 12)
  block:
    ## Nested 3-level
    let a = make_layout(((7, (8, 9)), 10), ((1, (2, 3)), 4))
    let b = mapLeavesWith(a): (it_sh * 3, it_st)
    doAssert b.shape === ((21, (24, 27)), 30)
    doAssert b.stride === ((1, (2, 3)), 4)
  block:
    ## Flat: dynamic shapes (let variables, not literals)
    let d2 = 2
    let d3 = 3
    let d1 = 1
    let d2st = 2
    let a = make_layout((d2, d3), (d1, d2st))
    let b = mapLeavesWith(a): (it_sh * 10, it_st)
    doAssert b.shape === (20, 30)
    doAssert b.stride === (1, 2)
  block:
    ## Nested 2-level: dynamic shapes
    let d2 = 2
    let d3 = 3
    let d5 = 5
    let d1 = 1
    let d2st = 2
    let d6st = 6
    let a = make_layout(((d2, d3), d5), ((d1, d2st), d6st))
    let b = mapLeavesWith(a): (it_sh * 2, it_st * 2)
    doAssert b.shape === ((4, 6), 10)
    doAssert b.stride === ((2, 4), 12)
  echo "    mapLeavesWith: 7 cases OK"

# ═══════════════════════════════════════════════════════════════
#  upcast / downcast — ref: tensor-layouts/tests/layouts.py + MoYe.jl
# ═══════════════════════════════════════════════════════════════

proc runUpcastDowncastTests* =
  # ── Python tensor-layouts: test_upcast_simple_stride1 ──
  block:
    ## upcast divides innermost (stride-1) shape by n.
    ## (32, 32):(32, 1) → upcast<16> → (32, 2):(2, 1)
    let a = make_layout((32, 32), (32, 1))
    let b = a.upcast(16)
    doAssert b.shape === (32, 2), "shape: " & $b.shape
    doAssert b.stride === (2, 1), "stride: " & $b.stride
  # ── Python: test_upcast_hierarchical_value_mode ──
  block:
    ## upcast handles nested value modes.
    ## SM75_U32x4_LDSM_N dst_layout_bits: (32, (32, 4)):(32, (1, 1024))
    ## upcast<16> → (32, (2, 4)):(2, (1, 64))
    let a = make_layout((32, (32, 4)), (32, (1, 1024)))
    let b = a.upcast(16)
    doAssert b.shape === (32, (2, 4)), "shape: " & $b.shape
    doAssert b.stride === (2, (1, 64)), "stride: " & $b.stride
  # ── Python: test_upcast_transpose_layout ──
  block:
    ## upcast handles transpose layouts (innermost stride > 1).
    ## SM75_U16x2_LDSM_T dst_layout_bits:
    ##   ((4, 8), (16, 2)):((256, 16), (1, 128))
    ## upcast<16> → ((4, 8), (1, 2)):((16, 1), (1, 8))
    let a = make_layout(((4, 8), (16, 2)), ((256, 16), (1, 128)))
    let b = a.upcast(16)
    doAssert b.shape === ((4, 8), (1, 2)), "shape: " & $b.shape
    doAssert b.stride === ((16, 1), (1, 8)), "stride: " & $b.stride
  # ── Python: test_upcast_identity ──
  block:
    ## upcast<1> returns the same layout.
    let a = make_layout((4, 8), (8, 1))
    let b = a.upcast(1)
    doAssert b.shape === (4, 8)
    doAssert b.stride === (8, 1)
  # ── Python: test_upcast_broadcast_stride ──
  block:
    ## upcast preserves stride-0 (broadcast) modes unchanged.
    let a = make_layout((4, 8), (0, 1))
    let b = a.upcast(4)
    doAssert b.stride[0] == 0
    doAssert b.shape[0] == 4
  # ── Python: test_downcast_simple ──
  block:
    ## downcast multiplies stride-1 shape by n, other strides by n.
    ## (32, 2):(2, 1) → downcast<16> → (32, 32):(32, 1)
    let a = make_layout((32, 2), (2, 1))
    let b = a.downcast(16)
    doAssert b.shape === (32, 32), "shape: " & $b.shape
    doAssert b.stride === (32, 1), "stride: " & $b.stride
  # ── Python: test_upcast_downcast_roundtrip ──
  block:
    ## downcast(upcast(layout, n), n) recovers original (innermost size >= n).
    let l1 = make_layout((32, 32), (32, 1))
    let r1 = l1.upcast(16).downcast(16)
    doAssert r1.shape === l1.shape, "r1 shape: " & $r1.shape
    doAssert r1.stride === l1.stride
  block:
    let l2 = make_layout((32, (32, 4)), (32, (1, 1024)))
    let r2 = l2.upcast(16).downcast(16)
    doAssert r2.shape === l2.shape, "r2 shape: " & $r2.shape
    doAssert r2.stride === l2.stride
  # ── Python: test_downcast_upcast_roundtrip ──
  block:
    ## upcast(downcast(layout, n), n) recovers the original.
    let l1 = make_layout((32, 2), (2, 1))
    let r1 = l1.downcast(4).upcast(4)
    doAssert r1.shape === l1.shape, "r1 shape: " & $r1.shape
    doAssert r1.stride === l1.stride
  block:
    let l2 = make_layout((4, 8), (8, 1))
    let r2 = l2.downcast(4).upcast(4)
    doAssert r2.shape === l2.shape, "r2 shape: " & $r2.shape
    doAssert r2.stride === l2.stride
  # ── MoYe.jl: recast (array.jl) ──
  block:
    ## MoYe: recast(Int32, a) on Int8 layout(4,3) → layout(16,3):(1,16).
    ## sizeof(Int32)/sizeof(Int8)=4 → downcast<4> on (4,3):(1,4)
    ## Leaf0 sh=4,st=1: |1|==1 → (16,1)     Leaf1 sh=3,st=4: |4|>1 → (3,16)
    let a = make_layout((4, 3))
    let b = a.downcast(4)
    doAssert b.shape === (16, 3), "shape: " & $b.shape
    doAssert b.stride === (1, 16)
  block:
    ## MoYe: Float32 layout(4,3) recast to Float64 → (2,3):(1,2).
    ## sizeof(Float64)/sizeof(Float32)=2 → upcast<2>
    ## Leaf0 sh=4,st=1: ceil_div(4,ceil_div(2,1))=2, stride=1  → (2,1)
    ## Leaf1 sh=3,st=4: ceil_div(3,ceil_div(2,4))=3, stride=2  → (3,2)
    let a = make_layout((4, 3))
    let b = a.upcast(2)
    doAssert b.shape === (2, 3), "shape: " & $b.shape
    doAssert b.stride === (1, 2), "stride: " & $b.stride
  block:
    ## MoYe: Float32 layout(4,3) recast to Float16 → (8,3):(1,8).
    ## sizeof(Float16)/sizeof(Float32)=0.5 → downcast<2>
    ## Leaf0 sh=4,st=1: |1|==1 → (8,1)     Leaf1 sh=3,st=4: |4|>1 → (3,8)
    let a = make_layout((4, 3))
    let b = a.downcast(2)
    doAssert b.shape === (8, 3), "shape: " & $b.shape
    doAssert b.stride === (1, 8), "stride: " & $b.stride
  # ── Own: stride-2 / broadcast / N=1 ──
  block:
    ## upcast: stride-2 gapped layout
    let a = make_layout(8, 2)
    let b = a.upcast(4)
    doAssert b.shape === 4, "shape: " & $b.shape
    doAssert b.stride === 1, "stride: " & $b.stride
  block:
    ## upcast: broadcast stride 0 unchanged
    let a = make_layout(8, 0)
    let b = a.upcast(4)
    doAssert b.shape === 8
    doAssert b.stride === 0
  block:
    ## downcast: stride-2 (stride multiplies)
    let a = make_layout(8, 2)
    let b = a.downcast(4)
    doAssert b.shape === 8
    doAssert b.stride === 8
  block:
    ## downcast: broadcast stride 0 unchanged
    let a = make_layout(8, 0)
    let b = a.downcast(4)
    doAssert b.shape === 8
    doAssert b.stride === 0
  block:
    ## upcast nested layout
    let a = make_layout(((2, 4), 8), ((1, 2), 4))
    let b = a.upcast(2)
    # leaf (2,1): ceil_div(2, ceil_div(2,1))=1, ceil_div(1,2)=1  → (1,1)
    # leaf (4,2): ceil_div(4, ceil_div(2,2))=4, ceil_div(2,2)=1  → (4,1)
    # leaf (8,4): ceil_div(8, ceil_div(2,4))=8, ceil_div(4,2)=2  → (8,2)
    doAssert b.shape === ((1, 4), 8), "shape: " & $b.shape
    doAssert b.stride === ((1, 1), 2), "stride: " & $b.stride
  block:
    ## downcast nested layout
    let a = make_layout(((2, 4), 8), ((1, 2), 4))
    let b = a.downcast(2)
    # leaf (2,1): |1|==1 → (4,1)
    # leaf (4,2): |2|!=1 → (4,4)
    # leaf (8,4): |4|!=1 → (8,8)
    doAssert b.shape === ((4, 4), 8), "shape: " & $b.shape
    doAssert b.stride === ((1, 4), 8), "stride: " & $b.stride
  block:
    ## upcast dynamic strides
    let d8 = 8; let d1 = 1; let d2 = 2
    let a = make_layout((d8, d1), (d1, d2))
    let b = a.upcast(4)
    doAssert b.shape === (8, 1)
    doAssert b.stride === (1, 1)
  block:
    ## downcast dynamic stride 1 (shape expands)
    let d8 = 8; let d1 = 1
    let a = make_layout(d8, d1)
    let b = a.downcast(4)
    doAssert b.shape === 32
    doAssert b.stride === 1
  echo "    upcast/downcast: 24 cases OK"


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
  echo "--- idx2crd ---"
  runIdx2crdTests()
  echo "--- layout[] ---"
  runCallOperatorTests()
  echo "--- col_major_strides ---"
  runColMajorStridesTests()
  echo "--- NCHW ---"
  runNCHWTests()
  echo "--- zipModes ---"
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

  echo "--- groupModes ---"
  block:
    ## Python test_append_prepend_replace_group: group(Layout((2,3,5,7)), 0, 2)
    let a = make_layout((2, 3, 5, 7))
    doAssert a.shape === (2, 3, 5, 7)
    doAssert a.stride === (1, 2, 6, 30)
    let b = a.groupModes(0, 2)
    doAssert b.shape === ((2, 3), 5, 7)
    doAssert b.stride === ((1, 2), 6, 30)
    let c = b.groupModes(1, 3)
    doAssert c.shape === ((2, 3), (5, 7))
    doAssert c.stride === ((1, 2), (6, 30))
  block:
    ## group with non-identity strides
    let a = make_layout((10, 20, 30, 40))
    let b = a.groupModes(1, 3)
    doAssert rank(b) === 3
  # block: -- blocked by tuple hash collision, pending https://github.com/nim-lang/Nim/pull/25889
  #   ## From start (B=0, E=3) — groups 3 elements into sub-tuple
  #   let a = make_layout((2, 3, 5, 7))
  #   let b = a.groupModes(0, 3)
  #   doAssert rank(b) === 2
  echo "    groupModes: 8 checks OK"

  echo "--- padRight/padLeft ---"
  block:
    ## padRight: append identity modes
    let a = make_layout((3, 4))
    let b = padRight(a, 3)
    doAssert b.shape === (3, 4, 1)
    doAssert b.stride === (1, 3, 0)
    doAssert rank(b) === 3
  block:
    ## padLeft: prepend identity modes
    let a = make_layout((3, 4))
    let b = padLeft(a, 3)
    doAssert b.shape === (1, 3, 4)
    doAssert b.stride === (0, 1, 3)
    doAssert rank(b) === 3
  block:
    ## padRight no-op (already at target rank)
    let a = make_layout((3, 4))
    let b = padRight(a, 2)
    doAssert b.shape === (3, 4)
    doAssert b.stride === (1, 3)
  block:
    ## padLeft no-op (already at target rank)
    let a = make_layout((3, 4))
    let b = padLeft(a, 2)
    doAssert b.shape === (3, 4)
    doAssert b.stride === (1, 3)
  block:
    ## padRight on scalar layout
    let a = make_layout(5)
    let b = padRight(a, 3)
    doAssert b.shape === (5, 1, 1)
    doAssert b.stride === (1, 0, 0)
  block:
    ## padLeft on scalar layout
    let a = make_layout(5)
    let b = padLeft(a, 3)
    doAssert b.shape === (1, 1, 5)
    doAssert b.stride === (0, 0, 1)
  echo "    padRight/padLeft: 6 cases OK"
  echo "--- takeModes/selectModes ---"
  block:
    let a = make_layout((2, 3, 5, 7))
    let b = a.takeModes(1, 3)
    doAssert b.shape === (3, 5)
    doAssert b.stride === (2, 6)
  block:
    let a = make_layout((2, 3, 5, 7))
    let b = a.selectModes(0, 3)
    doAssert b.shape === (2, 7)
    doAssert b.stride === (1, 30)
  block:
    let a = make_layout((2, 3, 5, 7))
    let b = a.takeModes(0, 1)
    doAssert b.shape === 2
    doAssert b.stride === 1
  block:
    let a = make_layout((2, 3, 5, 7))
    let b = a.selectModes(2)
    doAssert b.shape === 5
    doAssert b.stride === 6
  block:
    ## const indirection for takeModes
    const B = 1
    const E = 3
    let a = make_layout((2, 3, 5, 7))
    let b = a.takeModes(B, E)
    doAssert b.shape === (3, 5)
    doAssert b.stride === (2, 6)
  block:
    ## const indirection for selectModes
    const I0 = 0
    const I3 = 3
    let a = make_layout((2, 3, 5, 7))
    let b = a.selectModes(I0, I3)
    doAssert b.shape === (2, 7)
    doAssert b.stride === (1, 30)
  echo "    takeModes/selectModes: 6 cases OK"

  echo "--- mapLeavesWith ---"
  runMapLeavesWithTests()
  echo "--- upcast/downcast ---"
  runUpcastDowncastTests()

when isMainModule:
  runTests()