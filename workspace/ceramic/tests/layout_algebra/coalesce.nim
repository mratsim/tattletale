# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Run: nim c -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip workspace/ceramic/tests/layout_algebra/coalesce.nim

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/tests/layouts_testutils

proc assertNestedCoalesceIsAstNoop(L: Layout, preserveTrailing: static bool = false) =
  ## Coalescing an already-coalesced layout is a no-op.
  doAssert coalesce(L, preserveTrailing) ===
    coalesce(coalesce(L, preserveTrailing), preserveTrailing)


# ── Scalar rank-1 ────────────────────────────────────────────
# rank-1 layouts keep their trailing size-1 dimension, only rank > 1
# layouts drop it, so Layout(1,1) coalesces to itself 1:1.
proc runCoalesceScalarTests =
  block:
    let l = make_layout(1, 0)
    let c = coalesce(l)
    doAssert c === (1, 0)
  block:
    let l = make_layout(1, 1)
    let c = coalesce(l)
    doAssert c === (1, 1)
  block:
    let l = make_layout(1, 2)
    let c = coalesce(l)
    doAssert c === (1, 2)
  block:
    let l = make_layout(1, 5)
    let c = coalesce(l)
    doAssert c === (1, 5)
  echo "  Scalar: 4 cases OK"

# ── Column-major contiguous ──────────────────────────────────
proc runCoalesceColMajorTests =
  const C2 = 2
  const C4 = 4
  let l1 = make_layout((C2, C4), (1, 2))
  let c1 = coalesce(l1)
  doAssert c1 === (8, 1)
  const C6 = 6
  let l2 = make_layout((C2, C4, C6), (1, 2, 8))
  let c2 = coalesce(l2)
  doAssert c2 === (48, 1)
  echo "  Column-major contiguous: 2 cases OK"

# ── Size-1 dimensions ────────────────────────────────────────
proc runCoalesceSize1Tests =
  block:
    let l = make_layout((1, 8), (1, 1))
    let c = coalesce(l)
    doAssert c === (8, 1)
  block:
    let l = make_layout((1, 1, 8), (1, 1, 1))
    let c = coalesce(l)
    doAssert c === (8, 1)
  block:
    let l = make_layout((1, 8, 1), (1, 1, 1))
    let c = coalesce(l)
    doAssert c === (8, 1)
  echo "  Size-1 dimensions: 3 cases OK"

# ── Stride-0 dimensions ──────────────────────────────────────
proc runCoalesceStride0Tests =
  block:
    let l = make_layout((4, 1), (1, 0))
    let c = coalesce(l)
    doAssert c === (4, 1)
  block:
    let l = make_layout((1, 2), (0, 2))
    let c = coalesce(l)
    doAssert c === (2, 2)
  block:
    let l = make_layout((2, 1), (3, 0))
    let c = coalesce(l)
    doAssert c === (2, 3)
  echo "  Stride-0 dimensions: 3 cases OK"

# ── Non-contiguous strides ───────────────────────────────────
proc runCoalesceNonContigTests =
  block:
    let l = make_layout((4, 8), (1, 4))
    let c = coalesce(l)
    doAssert c === (32, 1)
  block:
    let l = make_layout((4, 8), (1, 5))
    let c = coalesce(l)
    doAssert c === ((4, 8), (1, 5))
  block:
    let l = make_layout((3, 4, 5), (1, 3, 12))
    let c = coalesce(l)
    doAssert c === (60, 1)
  echo "  Non-contiguous strides: 3 cases OK"

# ── Mixed ────────────────────────────────────────────────────
proc runCoalesceMixedTests =
  block:
    let l = make_layout((6, 7, 4), (1, 6, 42))
    let c = coalesce(l)
    doAssert c === (168, 1)
  block:
    let l = make_layout((2, 6), (1, 9))
    let c = coalesce(l)
    doAssert c === ((2, 6), (1, 9))
  block:
    let l = make_layout((3, 4, 5), (1, 1, 12))
    let c = coalesce(l)
    doAssert c === ((3, 4, 5), (1, 1, 12))
  echo "  Mixed: 3 cases OK"

# ── Dynamic shapes ───────────────────────────────────────────
proc runCoalesceDynamicTests =
  block:
    let d8 = 8
    let l = make_layout(d8, 1)
    let c = coalesce(l)
    doAssert c === (8, 1)
  block:
    let d12 = 12
    let l = make_layout(d12, 2)
    let c = coalesce(l)
    doAssert c === (12, 2)
  echo "  Dynamic shapes: 2 cases OK"

# ── Nested leaves ────────────────────────────────────────────
# Coalesce over nested shape and stride layouts, leafwise merge
# with the single-leaf collapse kept exact.
proc runCoalesceNestedLeafTests =
  block:
    # Nested 1-tuple wrappers collapse to the rank-1 layout of their single leaf
    let l = make_layout(((4,),), ((2,),))
    let c = coalesce(l)
    doAssert c === (4, 2)
  block:
    # Nested single-leaf size-1 shape (inactive dimension)
    let l = make_layout(((1,),), ((0,),))
    let c = coalesce(l)
    doAssert c === (1, 0)
  block:
    # Nested coalescable pair
    let l = make_layout(((4, 2),), ((1, 4),))
    let c = coalesce(l)
    doAssert c === (8, 1)
  block:
    # Nested non-contiguous pair stays split
    let l = make_layout(((2, 3),), ((1, 4),))
    let c = coalesce(l)
    doAssert c === ((2, 3), (1, 4))
  block:
    # Mixed nesting depths and dynamic leaves
    let d8 = 8
    let l = make_layout(((d8, 2),), ((1, 4),))
    let c = coalesce(l)
    doAssert c === ((8, 2), (1, 4))
  echo "  Nested leaves: 5 cases OK"

# ═══════════════════════════════════════════════════════════════
#  Anti-regressions: constant-folding and single-evaluation
#  corner cases (layouts_testutils.check)
# ═══════════════════════════════════════════════════════════════

# ── coalesce over a const layout ─────────────────────────────
proc runCoalesceConstantFixtureTests =
  # inline coalesce folds at compile time and stays Int[N]
  block:
    let r = coalesce(make_layout((2, ceil_div(16, 8)), (2, 8)))
    check r.shape, (2, 2), (Int[2], Int[2])
    check r.stride, (2, 8), (Int[2], Int[8])
  # const context folds identically
  block:
    const c = coalesce(make_layout((2, ceil_div(16, 8)), (2, 8)))
    check c.shape, (2, 2), (Int[2], Int[2])
    check c.stride, (2, 8), (Int[2], Int[8])
  echo "  const layout: 2 cases OK"

# ── coalesce inside compose, trailing size-1 dimension ───────
proc runCoalesceTrailingSizeOneTests =
  ## Coalesce of the compose LHS: a chain that reaches a trailing
  ## size-1 dimension's stride absorbs it, a chain that stops short
  ## leaves the Int[DynamicSentinel] marker untouched by the arithmetic.
  block:
    let r = compose(make_layout((4, 1), (1, 4)), make_layout(4, 1))
    check r.shape, 4, Int[4]
    check r.stride, 1, Int[1]
  block:
    let r = compose(make_layout((2, 1), (1, 3)), make_layout(2, 1))
    check r.shape, 2, Int[2]
    check r.stride, 1, Int[1]
  echo "  compose trailing size-1: 2 cases OK"

# ── coalesce over runtime layouts ─────────────────────────────
proc runCoalesceRuntimeLayoutTests =
  ## Runtime leaves are invisible at compile time: never merged.
  ## The layout argument is evaluated exactly once.
  block:
    var buildCount = 0
    proc countedLayout(n, m: int): auto =
      inc buildCount
      make_layout((n, m), (1, n))
    let r = coalesce(countedLayout(4, 8))
    doAssert buildCount == 1,
      "coalesce must evaluate a layout-valued argument exactly once"
    doAssert r.shape === (4, 8)
    doAssert r.stride === (1, 4)
  block:
    let n = 4
    let s = 8
    let r = coalesce(make_layout(n, s))
    doAssert r.shape === 4
    doAssert r.stride === 8
    doAssert r(3) == 24
  block:
    # merging needs static strides, a runtime leaf splits the chains,
    # the static leaves on each side merge on their own values
    let n = 2
    let r = coalesce(make_layout((4, n, 8), (1, 4, 8)))
    doAssert r.shape === (4, 2, 8)
    doAssert r.stride === (1, 4, 8)
    let rs = coalesce(make_layout((4, 2, 8), (1, 4, 8)))
    check rs.shape, 64, Int[64]
    check rs.stride, 1, Int[1]
  echo "  runtime layouts: 3 cases OK"

# ── coalesce preserveTrailing ─────────────────────────────────
proc runCoalescePreserveTrailingTests =
  ## preserveTrailing keeps a trailing size-1 dimension as an Int[DynamicSentinel] marker
  ## unless the chain absorbs it.
  block:
    let r = coalesce(make_layout((2, 1), (1, 3)), true)
    check r.shape, (2, DynamicSentinel), (Int[2], Int[DynamicSentinel])
    check r.stride, (1, 3), (Int[1], Int[3])
  block:
    let r = coalesce(make_layout((4, 1), (1, 4)), true)
    check r.shape, 4, Int[4]
    check r.stride, 1, Int[1]
  block:
    let r = coalesce(make_layout((1, 1), (0, 0)), true)
    check r.shape, DynamicSentinel, Int[DynamicSentinel]
    check r.stride, 0, Int[0]
  echo "  preserveTrailing: 3 cases OK"

# ── all-size-1 layouts ────────────────────────────────────────
proc runCoalesceOneLeafTests =
  # an all-size-1 layout keeps its last stride, coalesce(1:4) == 1:4
  block:
    let r = coalesce(make_layout(64, 1))
    check r.shape, 64, Int[64]
    check r.stride, 1, Int[1]
  block:
    let r = coalesce(make_layout(1, 0))
    check r.shape, 1, Int[1]
    check r.stride, 0, Int[0]
  block:
    let r = coalesce(make_layout(1, 4))
    check r.shape, 1, Int[1]
    check r.stride, 4, Int[4]
  block:
    let r = coalesce(make_layout((1, 1), (0, 0)))
    check r.shape, 1, Int[1]
    check r.stride, 0, Int[0]
  echo "  all-size-1: 4 cases OK"

proc runCoalesceIdentityTests =
  block:
    assertNestedCoalesceIsAstNoop(make_layout((2, 3, 4), (1, 8, 64)))
  block:
    let m = 4
    let n = 8
    let s = 2
    assertNestedCoalesceIsAstNoop(make_layout((m, n), (1, s)))
  block:
    assertNestedCoalesceIsAstNoop(make_layout((4, 1), (1, 4)))
  block:
    assertNestedCoalesceIsAstNoop(make_layout((2, 1), (1, 3)), true)
  block:
    let r = coalesce(make_layout((2, 4), (1, 2)))
    check r.shape, 8, Int[8]
    check r.stride, 1, Int[1]
  echo "  already-coalesced: 5 cases OK"

proc runCoalesceTests =
  echo "\n── Coalesce ──"
  runCoalesceScalarTests()
  runCoalesceColMajorTests()
  runCoalesceSize1Tests()
  runCoalesceStride0Tests()
  runCoalesceNonContigTests()
  runCoalesceMixedTests()
  runCoalesceDynamicTests()
  runCoalesceNestedLeafTests()
  echo "\n── Anti-regressions ──"
  runCoalesceConstantFixtureTests()
  runCoalesceTrailingSizeOneTests()
  runCoalesceRuntimeLayoutTests()
  runCoalescePreserveTrailingTests()
  runCoalesceOneLeafTests()
  runCoalesceIdentityTests()
  echo "\nALL TESTS PASSED"

when isMainModule:
  runCoalesceTests()
