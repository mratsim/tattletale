# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0)
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Test: make_layout, shape + stride, const correctness, stride order
# Run: nim c -r workspace/ceramic/tests/layout/t_make_layout.nim

import std/macros, std/typetraits
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layouts {.all.}
import workspace/ceramic/tests/layout/layouts_testutils

# ═══════════════════════════════════════════════════════════════
#  make_layout — shape + stride, const correctness, stride order
# ═══════════════════════════════════════════════════════════════

proc runMakeLayoutTests =
  let d1 = 1

  # ═══════════════════════════════════════════════════════════════
  #  1. Scalar layouts — (int, int) → Layout[int, int]
  # ═══════════════════════════════════════════════════════════════

  block:  # (1,0)
    doAssert make_layout(1, 0) === (1, 0)

  block:  # (1,1)
    doAssert make_layout(1, 1) === (1, 1)

  # ═══════════════════════════════════════════════════════════════
  #  2. Const vs let scalars — folding verified by type
  # ═══════════════════════════════════════════════════════════════

  block:
    const C4 = 4
    let l = make_layout(C4, 0)
    doAssert l === (4, 0)
    check l.shape, 4, Int[4]
    check l.stride, 0, Int[0]

  block:
    const C4 = 4
    let l = make_layout(C4)
    doAssert l === (4, 1)
    check l.shape, 4, Int[4]
    check l.stride, 1, Int[1]

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
    check l.shape[0], 2, Int[2]
    check l.shape[1], 4, Int[4]
    check l.stride[0], 1, Int[1]
    check l.stride[1], 2, Int[2]

  block:
    const C2 = 2; const C4 = 4
    let l = make_layout((C2, C4))
    doAssert l === ((2, 4), (1, 2))
    check l.shape[0], 2, Int[2]
    check l.shape[1], 4, Int[4]
    check l.stride[0], 1, Int[1]
    check l.stride[1], 2, Int[2]

  # ═══════════════════════════════════════════════════════════════
  #  4. Mixed const/let tuple
  # ═══════════════════════════════════════════════════════════════

  block:
    const C2 = 2; const C4 = 4; let d2 = 2
    let l = make_layout((d2, C4), (1, 2))
    echo l
    echo typeof(l)
    doAssert l === ((2, 4), (1, 2))
    check l.shape[0], 2, int
    check l.shape[1], 4, Int[4]
    check l.stride[0], 1, Int[1]
    check l.stride[1], 2, Int[2]

  block:
    const C2 = 2; const C4 = 4; let d2 = 2
    let l = make_layout((d2, C4))
    doAssert l === ((2, 4), (1, 2))
    check l.shape[0], 2, int
    check l.shape[1], 4, Int[4]
    check l.stride[0], 1, Int[1]      # always 1 (Int[1]) for column-major
    check l.stride[1], 2, int

  # ═══════════════════════════════════════════════════════════════
  #  5. Let tuple — all dynamic
  # ═══════════════════════════════════════════════════════════════

  block:
    let d2 = 2; let d4 = 4
    let l = make_layout((d2, d4), (1, 2))
    doAssert l === ((2, 4), (1, 2))
    check l.shape[0], 2, int
    check l.shape[1], 4, int
    check l.stride[0], 1, Int[1]
    check l.stride[1], 2, Int[2]

  block:
    let d2 = 2; let d4 = 4
    let l = make_layout((d2, d4))
    doAssert l === ((2, 4), (1, 2))
    check l.shape[0], 2, int
    check l.shape[1], 4, int
    check l.stride[0], 1, Int[1]      # always 1 (Int[1]) for column-major
    check l.stride[1], 2, int

  # ═══════════════════════════════════════════════════════════════
  #  6. Tuple assigned to variable — loses static info
  # ═══════════════════════════════════════════════════════════════

  block:
    const C2 = 2; const C4 = 4
    let shapeTuple = (C2, C4)
    let l = make_layout(shapeTuple, (1, 2))
    doAssert l === ((2, 4), (1, 2))
    check l.shape[0], 2, int
    check l.shape[1], 4, int
    check l.stride[0], 1, Int[1]
    check l.stride[1], 2, Int[2]

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
    check l.shape[0], 2, Int[2]
    check l.shape[1], 4, Int[4]

  # ═══════════════════════════════════════════════════════════════
  #  9. const variable from function — also folds to nnkTupleConstr
  # ═══════════════════════════════════════════════════════════════

  block:
    proc mkShape(): auto = (2, 4)
    const shape = mkShape()
    let l = make_layout(shape, (1, 2))
    doAssert l === ((2, 4), (1, 2))
    check l.shape[0], 2, Int[2]
    check l.shape[1], 4, Int[4]

  # ═══════════════════════════════════════════════════════════════
  #  10. Recursive (nested) tuple — all Int[N] via literals
  # ═══════════════════════════════════════════════════════════════

  block:
    let l = make_layout(((2, 2), (2, 2)), ((1, 4), (8, 32)))
    doAssert l === (((2, 2), (2, 2)), ((1, 4), (8, 32)))
    check l.shape[0][0], 2, Int[2]
    check l.shape[0][1], 2, Int[2]
    check l.shape[1][0], 2, Int[2]
    check l.shape[1][1], 2, Int[2]

  # ═══════════════════════════════════════════════════════════════
  #  11. Recursive tuple via variable — all dynamic
  # ═══════════════════════════════════════════════════════════════

  block:
    let nestedShape = ((2, 2), (2, 2))
    let nestedStride = ((1, 4), (8, 32))
    let l = make_layout(nestedShape, nestedStride)
    doAssert l === (((2, 2), (2, 2)), ((1, 4), (8, 32)))
    check l.shape[0][0], 2, int
    check l.shape[0][1], 2, int
    check l.shape[1][0], 2, int
    check l.shape[1][1], 2, int

  # ═══════════════════════════════════════════════════════════════
  #  12. Folding checks on standalone layouts
  # ═══════════════════════════════════════════════════════════════

  block:
    const C4 = 4
    let d4 = 4
    let lConst = make_layout(C4, 0)
    let lLet = make_layout(d4, 0)
    check lConst.shape, 4, Int[4]
    check lLet.shape, 4, int

  block:
    const C2 = 2; const C4 = 4
    let d2 = 2
    let lMixed = make_layout((d2, C4), (1, 2))
    check lMixed.shape[0], 2, int
    check lMixed.shape[1], 4, Int[4]

  # ═══════════════════════════════════════════════════════════════
  #  13. Stride-order construction — LayoutLeft / LayoutRight
  # ═══════════════════════════════════════════════════════════════

  block:  # LayoutLeft (default): leftmost dimension contiguous
    doAssert make_layout((4, 8), LayoutLeft) === ((4, 8), (1, 4))

  block:  # LayoutRight: rightmost dimension contiguous
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
    check l.shape[0], 2, int
    check l.shape[1], 3, Int[3]
    check l.shape[2], 8, Int[8]
    check l.stride[0], 1, Int[1]      # always 1 (Int[1]) for column-major
    check l.stride[1], 2, int
    check l.stride[2], 6, int

  block:  # LayoutLeft mixed: static int literals, no pre-conversion
    let dN = 2
    const C3 = 3; const C8 = 8
    let l = make_layout((C8, C3, dN), LayoutLeft)
    doAssert l === ((8, 3, 2), (1, 8, 24))
    check l.shape[0], 8, Int[8]
    check l.shape[1], 3, Int[3]
    check l.shape[2], 2, int
    check l.stride[0], 1, Int[1]
    check l.stride[1], 8, Int[8]
    check l.stride[2], 24, Int[24]

  block:  # LayoutRight mixed: static int literals, no pre-conversion
    let dN = 2
    const C3 = 3; const C8 = 8
    let l = make_layout((dN, C3, C8), LayoutRight)
    doAssert l === ((2, 3, 8), (24, 8, 1))
    check l.shape[0], 2, int
    check l.shape[1], 3, Int[3]
    check l.shape[2], 8, Int[8]
    check l.stride[0], 24, Int[24]
    check l.stride[1], 8, Int[8]
    check l.stride[2], 1, Int[1]

  block:  # Custom strides mixed: all-static shape, explicit strides
    let dN = 2
    const C3 = 3; const C8 = 8
    let l = make_layout((dN, C3, C8), (1, 1, 1))
    doAssert l === ((2, 3, 8), (1, 1, 1))
    check l.shape[0], 2, int
    check l.shape[1], 3, Int[3]
    check l.shape[2], 8, Int[8]
    check l.stride[0], 1, Int[1]
    check l.stride[1], 1, Int[1]
    check l.stride[2], 1, Int[1]

  echo "  make_layout: 23 cases OK"

  # ── Nested tuple make_layout ──
  block:
    let l = make_layout(((4, 1), (8, 8)))
    doAssert l.shape === ((4, 1), (8, 8)) and l.stride === ((1, 4), (4, 32))
  block:
    let l = make_layout(((1, 4), (8, 4)))
    doAssert l.shape === ((1, 4), (8, 4)) and l.stride === ((1, 1), (4, 32))
  block:
    let mr = 4; let mpT = 8; let kc = 8
    let l = make_layout(((mr, 1), (mpT, kc)))
    check crd2idx(l, ((0, 0), (0, 0))), 0, int
    check crd2idx(l, ((0, 0), (1, 0))), mr, int
    check crd2idx(l, ((0, 0), (0, 1))), mr * mpT, int
    check crd2idx(l, ((1, 0), (0, 0))), 1, int
  block:
    let l = make_layout(((4, 1), (8, 8)), LayoutRight)
    doAssert l.shape === ((4, 1), (8, 8)) and l.stride === ((64, 64), (8, 1))

  echo "  Nested make_layout: 4 cases OK"

  # ═══════════════════════════════════════════════════════════════
  #  product(l.shape) — shape extracted from layout
  # ═══════════════════════════════════════════════════════════════
  block:
    let l = make_layout((Int[2](), Int[4]()), (Int[1](), Int[2]()))
    let r = product(l.shape)
    doAssert r === 8

  block:
    let d1 = 1; let d4dyn = 4
    let l = make_layout((Int[2](), d4dyn), (d1, Int[2]()))
    let r = product(l.shape)
    doAssert r === 8
  echo "  product(l.shape): 2 cases OK"

  # ═══════════════════════════════════════════════════════════════
  #  Runtime strides — custom strides that are not compile-time
  # ═══════════════════════════════════════════════════════════════

  block:  # all-dynamic strides
    let d2 = 2; let d4 = 4
    let s1 = 1; let s2 = 2
    let l = make_layout((d2, d4), (s1, s2))
    doAssert l === ((2, 4), (1, 2))
    check l.stride[0], 1, int
    check l.stride[1], 2, int
    check crd2idx(l, (1, 1)), 3, int
    check crd2idx(l, (0, 3)), 6, int

  block:  # mixed static/dynamic strides
    const C4 = 4; let d2 = 2
    let s1 = 1
    let l = make_layout((d2, C4), (s1, 2))
    doAssert l === ((2, 4), (1, 2))
    check l.stride[0], 1, int
    check l.stride[1], 2, Int[2]
    check crd2idx(l, (1, 1)), 3, int

  block:  # runtime scalar stride, nonzero
    let d4 = 4
    let ds = 3
    let l = make_layout(d4, ds)
    doAssert l === (4, 3)
    check l.stride, 3, int
    check crd2idx(l, 2), 6, int

  block:  # runtime nested strides
    let mr = 4; let mpT = 8; let kc = 8
    let sa = 1; let sb = 4; let sc = 8; let sd = 32
    let l = make_layout(((mr, 1), (mpT, kc)), ((sa, sb), (sc, sd)))
    check l.stride, ((1, 4), (8, 32)), ((int, int), (int, int))
    check crd2idx(l, ((0, 0), (0, 0))), 0, int
    check crd2idx(l, ((1, 0), (0, 0))), 1, int
    check crd2idx(l, ((0, 1), (0, 0))), 4, int
    check crd2idx(l, ((0, 0), (1, 0))), 8, int
    check crd2idx(l, ((0, 0), (0, 1))), 32, int

  echo "  Runtime strides: 4 cases OK"

  # ═══════════════════════════════════════════════════════════════
  #  Strided by 2 — 2D layouts whose strides are 2, non-contiguous
  # ═══════════════════════════════════════════════════════════════

  block:  # all-static shape and strides
    const C4 = 4; const C8 = 8
    let l = make_layout((C4, C8), (2, 2))
    doAssert l === ((4, 8), (2, 2))
    check l.stride[0], 2, Int[2]
    check l.stride[1], 2, Int[2]
    check crd2idx(l, (0, 0)), 0, Int
    check crd2idx(l, (1, 0)), 2, Int
    check crd2idx(l, (3, 0)), 6, Int
    check crd2idx(l, (0, 1)), 2, Int
    check crd2idx(l, (1, 1)), 4, Int
    check crd2idx(l, (3, 2)), 10, Int

  block:  # runtime shape, static strides of 2
    let d4 = 4; let d8 = 8
    let l = make_layout((d4, d8), (2, 2))
    check l.shape[0], 4, int
    check l.stride[0], 2, Int[2]
    check l.stride[1], 2, Int[2]
    check crd2idx(l, (2, 1)), 6, Int
    check crd2idx(l, (3, 3)), 12, Int

  block:  # runtime strides of 2
    let d4 = 4; let d8 = 8
    let s2 = 2
    let l = make_layout((d4, d8), (s2, s2))
    check l.stride[0], 2, int
    check l.stride[1], 2, int
    check crd2idx(l, (1, 0)), 2, int
    check crd2idx(l, (3, 2)), 10, int

  block:  # mixed: stride 2 on one dimension, runtime on the other
    const C8 = 8
    let d4 = 4
    let s2 = 2
    let l = make_layout((d4, C8), (2, s2))
    check l.stride[0], 2, Int[2]
    check l.stride[1], 2, int
    check crd2idx(l, (3, 1)), 8, int
    check crd2idx(l, (1, 4)), 10, int

  echo "  Strided by 2: 4 cases OK"

proc runGivenStrideTests =
  echo "--- make_layout given strides (verbatim congruent, base expansion) ---"

  # ═══════════════════════════════════════════════════════════════
  #  Congruent given strides pass through verbatim, gaps,
  #  stride-1 and stride-0 modes included
  # ═══════════════════════════════════════════════════════════════

  block:  # given strides kept as written
    let l = make_layout((4, 8), (8, 1))
    check l.shape, (4, 8), (Int[4], Int[8])
    check l.stride, (8, 1), (Int[8], Int[1])

  block:  # gaps: non-compact strides stay
    let l = make_layout((4, 8), (2, 5))
    check l.stride, (2, 5), (Int[2], Int[5])

  block:  # stride 1s stay
    let l = make_layout((2, 3), (1, 1))
    check l.stride, (1, 1), (Int[1], Int[1])

  block:  # stride 0 stays
    let l = make_layout((2, 3), (0, 1))
    check l.stride, (0, 1), (Int[0], Int[1])

  block:  # nested congruent strides stay nested verbatim
    let l = make_layout((2, (3, 4)), (1, (2, 8)))
    check l.stride, (1, (2, 8)), (Int[1], (Int[2], Int[8]))

  # ═══════════════════════════════════════════════════════════════
  #  A scalar stride seeds prefix products, pycute prefix_product
  # ═══════════════════════════════════════════════════════════════

  block:  # scalar broadcast
    let l = make_layout((2, 3), 4)
    check l.stride, (4, 8), (Int[4], Int[8])

  block:  # scalar broadcast of 1
    let l = make_layout((2, 3), 1)
    check l.stride, (1, 2), (Int[1], Int[2])

  # ═══════════════════════════════════════════════════════════════
  #  A scalar stride under a tuple mode is that mode's base,
  #  pycute prefix_product(((2,3),4), (8,1)) == ((8,16),1)
  # ═══════════════════════════════════════════════════════════════

  block:  # base expansion
    let l = make_layout(((2, 3), 4), (8, 1))
    check l.shape, ((2, 3), 4), ((Int[2], Int[3]), Int[4])
    check l.stride, ((8, 16), 1), ((Int[8], Int[16]), Int[1])

  block:  # per-mode bases
    let l = make_layout(((2, 3), (4, 5)), (1, 100))
    check l.stride, ((1, 2), (100, 400)), ((Int[1], Int[2]), (Int[100], Int[400]))

  block:  # dynamic base, static tail
    let s = 8
    let l = make_layout(((2, 3), 4), (s, 1))
    check l.stride[0][0], s, int
    check l.stride[0][1], 16, int
    check l.stride[1], 1, Int[1]

  echo "  Given strides: 10 cases OK"

proc runEvalOnceShadowingTests =
  echo "--- evalOnceAs shadowing (genSym'd aliases) ---"
  block:
    let a = make_layout((3,))
    let b = make_layout((2, 4))
    doAssert toInt(a.shape[0]) == 3, "no shadowing: layout (3,)"
    doAssert toInt(b.shape[0]) == 2, "no shadowing: layout (2,4) shape[0]"
    doAssert toInt(b.shape[1]) == 4, "no shadowing: layout (2,4) shape[1]"
    echo "    make_layout: 3 cases OK"

when isMainModule:
  runMakeLayoutTests()
  runGivenStrideTests()
  runEvalOnceShadowingTests()
