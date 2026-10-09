# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Run: nim c -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip workspace/ceramic/tests/layout_algebra/t_compose.nim

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_constructors

template check(got: untyped, expected: typed, expectedType: typedesc) =
  ## Check with constant-folding assertion
  block:
    let tmp = got
    type TmpType = typeof(tmp)
    when TmpType is expectedType:
      doAssert tmp === expected
    else:
      {.error: "[ttt] Please check constant-folding: type is " &
          $TmpType & ", expected " & $expectedType.}

proc assertCompositionProperty[Sh1, St1, Sh2, St2](a: Layout[Sh1, St1], b: Layout[Sh2, St2]) =
  ## Check that (a ∘ b)(x) == a(b(x))
  let r = compose(a, b)
  doAssert compatible(b.shape, r.shape),
    "compatible(" & $b.shape & ", " & $r.shape & ")"
  for x in 0 ..< size(b):
    doAssert r(x) === a(b(x)),
      "(a ∘ b)(" & $x & ")=" & $r(x) & " a(b(" & $x & "))=" & $a(b(x))

const flat   = make_layout((4, 8), (1, 32))
const nested = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))

# ── exact values ─────────────────────────────────────────────
proc runComposeExactValueTests =
  block:
    let c = compose(make_layout(8, 2), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 2, Int[2]
  block:
    let c = compose(make_layout(8, 2), make_layout(4, 2))
    check c.shape, 4, Int[4]
    check c.stride, 4, Int[4]
  block:
    let c = compose(make_layout((4, 8), (1, 4)), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout((4, 8), (1, 4)), make_layout(4, 4))
    check c.shape, 4, Int[4]
    check c.stride, 4, Int[4]
  block:
    let c = compose(make_layout(16, 1), make_layout((4, 4), (1, 4)))
    check c.shape, (4, 4), (Int[4], Int[4])
    check c.stride, (1, 4), (Int[1], Int[4])
  block:
    let c = compose(make_layout(((6, 1), (32, 512)), ((1, 192), (6, 192))), make_layout(200, 1))
    check c.shape, 200, Int[200]
    check c.stride, 1, Int[1]
  # rank-1 LHS with hierarchical RHS preserves the nesting
  block:
    let c = compose(make_layout(64, 1), make_layout(((2, 2), (2, 8)), ((1, 4), (2, 8))))
    check c.shape, ((2, 2), (2, 8)), ((Int[2], Int[2]), (Int[2], Int[8]))
    check c.stride, ((1, 4), (2, 8)), ((Int[1], Int[4]), (Int[2], Int[8]))
  # coalescable rank-2 LHS coalesces to rank-1 inside compose
  block:
    let c = compose(make_layout((8, 8), (1, 8)), make_layout(((2, 2), (2, 8)), ((1, 4), (2, 8))))
    check c.shape, ((2, 2), (2, 8)), ((Int[2], Int[2]), (Int[2], Int[8]))
    check c.stride, ((1, 4), (2, 8)), ((Int[1], Int[4]), (Int[2], Int[8]))
  block:
    let c = compose(make_layout(16, 2), make_layout((4, 4), (1, 4)))
    check c.shape, (4, 4), (Int[4], Int[4])
    check c.stride, (2, 8), (Int[2], Int[8])
  block:
    let c = compose(make_layout((4, 4), (1, 4)), make_layout((2, 2), (1, 2)))
    check c.shape, (2, 2), (Int[2], Int[2])
    check c.stride, (1, 2), (Int[1], Int[2])
  block:
    let c = compose(make_layout((6, 2), (8, 2)), make_layout((4, 3), (3, 1)))
    check c.shape, ((2, 2), 3), ((Int[2], Int[2]), Int[3])
    check c.stride, ((24, 2), 8), ((Int[24], Int[2]), Int[8])
  block:
    let c = compose(make_layout((10, 2), (16, 4)), make_layout((5, 4), (1, 5)))
    check c.shape, (5, (2, 2)), (Int[5], (Int[2], Int[2]))
    check c.stride, (16, (80, 4)), (Int[16], (Int[80], Int[4]))
  # tuple LHS + scalar RHS
  block:
    let c = compose(make_layout((4, 8), (1, 4)), make_layout(10, 1))
    check c.shape, 10, Int[10]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout((4, 8), (1, 4)), make_layout(8, 1))
    check c.shape, 8, Int[8]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout((4, 8), (1, 4)), make_layout(5, 1))
    check c.shape, 5, Int[5]
    check c.stride, 1, Int[1]
  echo "  exact values: 14 cases OK"

# ── fixtures ─────────────────────────────────────────────────
proc runComposeFixtureTests =
  block:
    # rank-1 LHS x flat RHS, the strides multiply through the LHS stride
    let r = compose(make_layout(32, 1), flat)
    check r.shape, (4, 8), (Int[4], Int[8])
    check r.stride, (1, 32), (Int[1], Int[32])
  block:
    # coalescable rank-2 LHS x nested RHS coalesces to rank-1 first
    let r = compose(make_layout((8, 8), (1, 8)), nested)
    check r.shape, ((4, 8), (2, 2)), ((Int[4], Int[8]), (Int[2], Int[2]))
    check r.stride, ((16, 1), (8, 64)), ((Int[16], Int[1]), (Int[8], Int[64]))
  block:
    # non-coalescable rank-2 LHS x flat RHS, B(i) = r + 32c, A at r + 32c = r + 64c
    let r = compose(make_layout((16, 8), (1, 32)), flat)
    check r.shape, (4, 8), (Int[4], Int[8])
    check r.stride, (1, 64), (Int[1], Int[64])
  block:
    # rank-1 LHS x nested RHS
    let r = compose(make_layout(32, 1), nested)
    check r.shape, ((4, 8), (2, 2)), ((Int[4], Int[8]), (Int[2], Int[2]))
    check r.stride, ((16, 1), (8, 64)), ((Int[16], Int[1]), (Int[8], Int[64]))
  block:
    # non-coalescable rank-2 LHS x nested RHS
    let r = compose(make_layout((16, 8), (1, 32)), nested)
    check r.shape, ((4, 8), (2, 2)), ((Int[4], Int[8]), (Int[2], Int[2]))
    check r.stride, ((32, 1), (8, 128)), ((Int[32], Int[1]), (Int[8], Int[128]))
  echo "  fixtures: 5 cases OK"

# ── simple rank-1 ────────────────────────────────────────────
proc runComposeSimpleTests =
  block:
    let c = compose(make_layout(1, 0), make_layout(1, 0))
    check c.shape, 1, Int[1]
    check c.stride, 0, Int[0]
  block:
    let c = compose(make_layout(1, 0), make_layout(1, 1))
    check c.shape, 1, Int[1]
    check c.stride, 0, Int[0]
  block:
    let c = compose(make_layout(1, 1), make_layout(1, 0))
    check c.shape, 1, Int[1]
    check c.stride, 0, Int[0]
  block:
    let c = compose(make_layout(1, 1), make_layout(1, 1))
    check c.shape, 1, Int[1]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout(4, 1), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout(4, 2), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 2, Int[2]
  block:
    let c = compose(make_layout(4, 1), make_layout(4, 2))
    check c.shape, 4, Int[4]
    check c.stride, 2, Int[2]
  block:
    let c = compose(make_layout(4, 0), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 0, Int[0]
  block:
    let c = compose(make_layout(4, 1), make_layout(4, 0))
    check c.shape, 4, Int[4]
    check c.stride, 0, Int[0]
  block:
    let c = compose(make_layout(1, 0), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 0, Int[0]
  block:
    let c = compose(make_layout(4, 1), make_layout(1, 0))
    check c.shape, 1, Int[1]
    check c.stride, 0, Int[0]
  block:
    let c = compose(make_layout(4, 1), make_layout(2, 1))
    check c.shape, 2, Int[2]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout(4, 2), make_layout(2, 1))
    check c.shape, 2, Int[2]
    check c.stride, 2, Int[2]
  block:
    let c = compose(make_layout(4, 1), make_layout(2, 2))
    check c.shape, 2, Int[2]
    check c.stride, 2, Int[2]
  block:
    let c = compose(make_layout(4, 2), make_layout(2, 2))
    check c.shape, 2, Int[2]
    check c.stride, 4, Int[4]
  assertCompositionProperty(make_layout(1, 0), make_layout(1, 0))
  assertCompositionProperty(make_layout(1, 0), make_layout(1, 1))
  assertCompositionProperty(make_layout(1, 1), make_layout(1, 0))
  assertCompositionProperty(make_layout(1, 1), make_layout(1, 1))
  assertCompositionProperty(make_layout(4, 1), make_layout(4, 1))
  assertCompositionProperty(make_layout(4, 2), make_layout(4, 1))
  assertCompositionProperty(make_layout(4, 1), make_layout(4, 2))
  assertCompositionProperty(make_layout(4, 0), make_layout(4, 1))
  assertCompositionProperty(make_layout(4, 1), make_layout(4, 0))
  assertCompositionProperty(make_layout(1, 0), make_layout(4, 1))
  assertCompositionProperty(make_layout(4, 1), make_layout(1, 0))
  assertCompositionProperty(make_layout(4, 1), make_layout(2, 1))
  assertCompositionProperty(make_layout(4, 2), make_layout(2, 1))
  assertCompositionProperty(make_layout(4, 1), make_layout(2, 2))
  assertCompositionProperty(make_layout(4, 2), make_layout(2, 2))
  echo "  simple: 15 cases OK"

# ── multi-dimension ──────────────────────────────────────────
proc runComposeMultiDimensionTests =
  block:
    let c = compose(make_layout((4, 3), (1, 1)), make_layout(12, 1))
    check c.shape, (4, 3), (Int[4], Int[3])
    check c.stride, (1, 1), (Int[1], Int[1])
  block:
    let c = compose(make_layout(12, 1), make_layout((4, 3), (1, 1)))
    check c.shape, (4, 3), (Int[4], Int[3])
    check c.stride, (1, 1), (Int[1], Int[1])
  block:
    let c = compose(make_layout(12, 2), make_layout((4, 3), (1, 1)))
    check c.shape, (4, 3), (Int[4], Int[3])
    check c.stride, (2, 2), (Int[2], Int[2])
  block:
    let c = compose(make_layout(12, 1), make_layout((4, 3), (3, 1)))
    check c.shape, (4, 3), (Int[4], Int[3])
    check c.stride, (3, 1), (Int[3], Int[1])
  block:
    let c = compose(make_layout(12, 2), make_layout((4, 3), (3, 1)))
    check c.shape, (4, 3), (Int[4], Int[3])
    check c.stride, (6, 2), (Int[6], Int[2])
  block:
    let c = compose(make_layout(12, 1), make_layout((2, 3), (2, 4)))
    check c.shape, (2, 3), (Int[2], Int[3])
    check c.stride, (2, 4), (Int[2], Int[4])
  block:
    let c = compose(make_layout((4, 3), (3, 1)), make_layout(12, 1))
    check c.shape, (4, 3), (Int[4], Int[3])
    check c.stride, (3, 1), (Int[3], Int[1])
  block:
    let c = compose(make_layout((4, 3), (3, 1)), make_layout(6, 2))
    check c.shape, (2, 3), (Int[2], Int[3])
    check c.stride, (6, 1), (Int[6], Int[1])
  block:
    let c = compose(make_layout((4, 3), (1, 4)), make_layout((4, 3), (1, 4)))
    check c.shape, (4, 3), (Int[4], Int[3])
    check c.stride, (1, 4), (Int[1], Int[4])
  block:
    let c = compose(make_layout((4, 3), (1, 4)), make_layout((6, 2), (2, 1)))
    check c.shape, (6, 2), (Int[6], Int[2])
    check c.stride, (2, 1), (Int[2], Int[1])
  block:
    let c = compose(make_layout((4, 3), (1, 4)), make_layout((4, 3), (3, 1)))
    check c.shape, (4, 3), (Int[4], Int[3])
    check c.stride, (3, 1), (Int[3], Int[1])
  block:
    let c = compose(make_layout((4, 3), (3, 1)), make_layout((4, 3), (1, 4)))
    check c.shape, (4, 3), (Int[4], Int[3])
    check c.stride, (3, 1), (Int[3], Int[1])
  assertCompositionProperty(make_layout(12, 1), make_layout((4, 3), (1, 1)))
  assertCompositionProperty(make_layout(12, 2), make_layout((4, 3), (1, 1)))
  assertCompositionProperty(make_layout(12, 1), make_layout((4, 3), (3, 1)))
  assertCompositionProperty(make_layout(12, 2), make_layout((4, 3), (3, 1)))
  assertCompositionProperty(make_layout(12, 1), make_layout((2, 3), (2, 4)))
  assertCompositionProperty(make_layout((4, 3), (3, 1)), make_layout(6, 2))
  assertCompositionProperty(make_layout((4, 3), (1, 4)), make_layout((4, 3), (1, 4)))
  assertCompositionProperty(make_layout((4, 3), (1, 4)), make_layout((6, 2), (2, 1)))
  assertCompositionProperty(make_layout((4, 3), (1, 4)), make_layout((4, 3), (3, 1)))
  assertCompositionProperty(make_layout((4, 3), (3, 1)), make_layout((4, 3), (1, 4)))
  echo "  multi-dimension: 12 cases OK"

# ── remainder ────────────────────────────────────────────────
proc runComposeRemainderTests =
  # the LHS leftover after B is consumed
  block:
    let c = compose(make_layout(1, 0), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 0, Int[0]
  block:
    let c = compose(make_layout(1, 1), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout(4, 1), make_layout(4, 2))
    check c.shape, 4, Int[4]
    check c.stride, 2, Int[2]
  block:
    let c = compose(make_layout((4, 3), (3, 1)), make_layout(24, 1))
    check c.shape, (4, 6), (Int[4], Int[6])
    check c.stride, (3, 1), (Int[3], Int[1])
  block:
    let c = compose(make_layout((4, 3), (3, 1)), make_layout(8, 1))
    check c.shape, (4, 2), (Int[4], Int[2])
    check c.stride, (3, 1), (Int[3], Int[1])
  block:
    # coalesce drops trailing size-1 stride-0 modes
    let c = compose(make_layout((4, 3, 1), (3, 1, 0)), make_layout(24, 1))
    check c.shape, (4, 6), (Int[4], Int[6])
    check c.stride, (3, 1), (Int[3], Int[1])
  block:
    let c = compose(make_layout((4, 3, 1), (3, 1, 0)), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 3, Int[3]
  block:
    let c = compose(make_layout((4, 6, 8, 10), (2, 3, 5, 7)), make_layout(6, 12))
    check c.shape, (2, 3), (Int[2], Int[3])
    check c.stride, (9, 5), (Int[9], Int[5])
  block:
    let c = compose(make_layout((8, 8), (8, 1)), make_layout(2, 3))
    check c.shape, 2, Int[2]
    check c.stride, 24, Int[24]
  block:
    let c = compose(make_layout((8, 8), (8, 1)), make_layout(3, 3))
    check c.shape, 3, Int[3]
    check c.stride, 24, Int[24]
  block:
    let c = compose(make_layout(3, 1), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout((48, 24, 5), (1, 128, 3072)), make_layout(32, 1))
    check c.shape, 32, Int[32]
    check c.stride, 1, Int[1]
  assertCompositionProperty(make_layout(1, 0), make_layout(4, 1))
  assertCompositionProperty(make_layout(1, 1), make_layout(4, 1))
  assertCompositionProperty(make_layout(4, 1), make_layout(4, 2))
  assertCompositionProperty(make_layout((4, 3), (3, 1)), make_layout(24, 1))
  assertCompositionProperty(make_layout((4, 3), (3, 1)), make_layout(8, 1))
  assertCompositionProperty(make_layout((4, 3, 1), (3, 1, 0)), make_layout(24, 1))
  assertCompositionProperty(make_layout((4, 3, 1), (3, 1, 0)), make_layout(4, 1))
  assertCompositionProperty(make_layout((4, 6, 8, 10), (2, 3, 5, 7)), make_layout(6, 12))
  assertCompositionProperty(make_layout((8, 8), (8, 1)), make_layout(2, 3))
  assertCompositionProperty(make_layout((8, 8), (8, 1)), make_layout(3, 3))
  assertCompositionProperty(make_layout(3, 1), make_layout(4, 1))
  assertCompositionProperty(make_layout((48, 24, 5), (1, 128, 3072)), make_layout(32, 1))
  echo "  remainder: 12 cases OK"

# ── nested RHS ───────────────────────────────────────────────
proc runComposeNestedTests =
  block:
    let c = compose(make_layout(((4, 2),), ((1, 16),)), make_layout((4, 2), (2, 1)))
    check c.shape, ((2, 2), 2), ((Int[2], Int[2]), Int[2])
    check c.stride, ((2, 16), 1), ((Int[2], Int[16]), Int[1])
  block:
    let c = compose(make_layout((2, 2), (2, 1)), make_layout((2, 2), (2, 1)))
    check c.shape, (2, 2), (Int[2], Int[2])
    check c.stride, (1, 2), (Int[1], Int[2])
  block:
    let c = compose(make_layout((4, 8, 2), (1, 4, 32)), make_layout((2, 2, 2), (2, 8, 1)))
    check c.shape, (2, 2, 2), (Int[2], Int[2], Int[2])
    check c.stride, (2, 8, 1), (Int[2], Int[8], Int[1])
  block:
    let c = compose(make_layout((4, 8, 2), (2, 8, 1)), make_layout((2, 2, 2), (1, 8, 2)))
    check c.shape, (2, 2, 2), (Int[2], Int[2], Int[2])
    check c.stride, (2, 16, 4), (Int[2], Int[16], Int[4])
  block:
    let c = compose(make_layout((4, 8, 2), (2, 8, 1)), make_layout((4, 2, 2), (2, 8, 1)))
    check c.shape, (4, 2, 2), (Int[4], Int[2], Int[2])
    check c.stride, (4, 16, 2), (Int[4], Int[16], Int[2])
  # scalar LHS, flat and nested RHS compose leafwise exactly,
  # a rank-1 nested RHS shape flattens
  block:
    let c = compose(make_layout(16, 1), make_layout((4, 2), (1, 16)))
    check c.shape, (4, 2), (Int[4], Int[2])
    check c.stride, (1, 16), (Int[1], Int[16])
  block:
    let c = compose(make_layout(16, 1), make_layout(((4, 2),), ((1, 16),)))
    check c.shape, (4, 2), (Int[4], Int[2])
    check c.stride, (1, 16), (Int[1], Int[16])
  block:
    let c = compose(make_layout(16, 1), make_layout(((4, 2), (2,)), ((1, 16), (32,))))
    check c.shape, ((4, 2), 2), ((Int[4], Int[2]), Int[2])
    check c.stride, ((1, 16), 32), ((Int[1], Int[16]), Int[32])
  block:
    let c = compose(make_layout(8, 2), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 2, Int[2]
  block:
    # a runtime-bound shape must match the literal form
    let d16 = 16
    let c = compose(make_layout(d16, 1), make_layout(((4, 2),), ((1, 16),)))
    check c.shape, (4, 2), (Int[4], Int[2])
    check c.stride, (1, 16), (Int[1], Int[16])
  assertCompositionProperty(make_layout((2, 2), (2, 1)), make_layout((2, 2), (2, 1)))
  assertCompositionProperty(make_layout((4, 8, 2), (1, 4, 32)), make_layout((2, 2, 2), (2, 8, 1)))
  assertCompositionProperty(make_layout((4, 8, 2), (2, 8, 1)), make_layout((2, 2, 2), (1, 8, 2)))
  assertCompositionProperty(make_layout((4, 8, 2), (2, 8, 1)), make_layout((4, 2, 2), (2, 8, 1)))
  echo "  nested: 10 cases OK"

# ── negative strides ─────────────────────────────────────────
proc runComposeNegStrideTests =
  block:
    let c = compose(make_layout(4, -1), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, -1, Int[-1]
  block:
    let c = compose(make_layout(4, 1), make_layout(4, -1))
    check c.shape, 4, Int[4]
    check c.stride, -1, Int[-1]
  block:
    let c = compose(make_layout(4, -1), make_layout(4, -1))
    check c.shape, 4, Int[4]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout(4, 1), make_layout(4, -2))
    check c.shape, 4, Int[4]
    check c.stride, -2, Int[-2]
  block:
    let c = compose(make_layout((4, 4), (-1, 1)), make_layout(2, 1))
    check c.shape, 2, Int[2]
    check c.stride, -1, Int[-1]
  block:
    let c = compose(make_layout((4, 4), (-1, 1)), make_layout((2, 4, 2), (1, 4, 2)))
    check c.shape, (2, 4, 2), (Int[2], Int[4], Int[2])
    check c.stride, (-1, 1, -2), (Int[-1], Int[1], Int[-2])
  assertCompositionProperty(make_layout(4, -1), make_layout(4, 1))
  assertCompositionProperty(make_layout(4, 1), make_layout(4, -1))
  assertCompositionProperty(make_layout(4, -1), make_layout(4, -1))
  assertCompositionProperty(make_layout(4, 1), make_layout(4, -2))
  assertCompositionProperty(make_layout((4, 4), (-1, 1)), make_layout(2, 1))
  assertCompositionProperty(make_layout((4, 4), (-1, 1)), make_layout((2, 4, 2), (1, 4, 2)))
  echo "  negative strides: 6 cases OK"

# ── tiler composition ────────────────────────────────────────
proc runComposeTilerTests =
  # int tiler elements compose each dimension with (N):(1),
  # the first N positions
  block:
    let c = compose(make_layout((32, 8), (1, 32)), (16, 2))
    check c.shape, (16, 2), (Int[16], Int[2])
    check c.stride, (1, 32), (Int[1], Int[32])
  # a `_` tiler element passes the dimension through whole
  block:
    let c = compose(make_layout((4, 8), (1, 4)), (_, 2))
    check c.shape, (4, 2), (Int[4], Int[2])
    check c.stride, (1, 4), (Int[1], Int[4])
  # leftover dimensions drop
  block:
    let c = compose(make_layout((4, 8, 2), (1, 4, 32)), (2, 4))
    check c.shape, (2, 4), (Int[2], Int[4])
    check c.stride, (1, 4), (Int[1], Int[4])
  block:
    let c = compose(make_layout((4, 8, 2), (1, 4, 32)), (2, _, _))
    check c.shape, (2, 8, 2), (Int[2], Int[8], Int[2])
    check c.stride, (1, 4, 32), (Int[1], Int[4], Int[32])
  # a Layout tiler element composes its dimension as a (T, V) atom
  block:
    let c = compose(make_layout((4, 8), (1, 4)), (make_layout(2, 2), _))
    check c.shape, (2, 8), (Int[2], Int[8])
    check c.stride, (2, 4), (Int[2], Int[4])
  block:
    let c = compose(make_layout((16, 8), (1, 16)), (make_layout((2, 4), (1, 4)), _))
    check c.shape, ((2, 4), 8), ((Int[2], Int[4]), Int[8])
    check c.stride, ((1, 4), 16), ((Int[1], Int[4]), Int[16])
  # a Layout tiler element on a later mode composes with its own mode,
  # the composed mode pair lands in the layout's flat domain
  block:
    let c = compose(make_layout((4, 8), (1, 4)), (_, make_layout(2, 2)))
    check c.shape, (4, 2), (Int[4], Int[2])
    check c.stride, (1, 8), (Int[1], Int[8])
  # static tiler literals promote through makeIntTuple forms too
  block:
    let c = compose(make_layout((32, 8), (1, 32)), makeIntTuple((16, 2)))
    check c.shape, (16, 2), (Int[16], Int[2])
    check c.stride, (1, 32), (Int[1], Int[32])
  # a runtime int stride keeps the consume runtime
  block:
    let dS = 2
    let c = compose(make_layout((32, 8), (1, dS)), (16, 4))
    check c.shape, (16, 4), (Int[16], Int[4])
    check c.stride, (1, dS), (Int[1], int)
  # a const stride leaf promotes through makeIntTuple,
  # the whole result comes out Int-typed
  block:
    const dS = 2
    let c = compose(make_layout((32, 8), (1, dS)), (16, 4))
    check c.shape, (16, 4), (Int[16], Int[4])
    check c.stride, (1, dS), (Int[1], Int[2])
  # the scalar fast path pairs (N, the layout's stride leaf) per mode.
  # no compose call is emitted, static leaves keep Int markers,
  # even under a runtime shape leaf
  block:
    let dN = 4
    let c = compose(make_layout((dN, 8), (1, 4)), (2, 2))
    check c.shape, (2, 2), (Int[2], Int[2])
    check c.stride, (1, 4), (Int[1], Int[4])
  # a tiler entry carries the implicit (N):(1) stride.
  # the resulting stride leaf is the layout's leaf itself,
  # never a running prefix
  block:
    let c = compose(make_layout((4, 8), (2, 3)), (2, 2))
    check c.shape, (2, 2), (Int[2], Int[2])
    check c.stride, (2, 3), (Int[2], Int[3])
  # a tiler longer than the layout is a compile-time error
  static:
    doAssert not compiles(compose(make_layout(4, 1), (2, 3)))
  # a 1-element tiler keeps its 1-tuple structure
  block:
    let c = compose(make_layout((4, 8), (1, 4)), (2,))
    check c.shape, (2,), (Int[2],)
    check c.stride, (1,), (Int[1],)
  # a sub-tuple tiler element on a scalar dimension is a rank error
  static:
    doAssert not compiles(compose(make_layout((4, 8), (1, 4)), (2, (2, 2))))
  # a symbol-bound layout reads its modes off the destructured parts,
  # the tiler path emits bracket reads, static fields stay Int-typed
  block:
    let sb = make_layout((4, 8), (1, 4))
    let c1 = compose(sb, (16, 2))
    check c1.shape, (16, 2), (Int[16], Int[2])
    check c1.stride, (1, 4), (Int[1], Int[4])
  block:
    let sb = make_layout((4, 8), (1, 4))
    let c2 = compose(sb, (_, make_layout(2, 2)))
    check c2.shape, (4, 2), (Int[4], Int[2])
    check c2.stride, (1, 8), (Int[1], Int[8])
  echo "  tiler: 17 cases OK"

# ── symbol-bound arguments ───────────────────────────────────
proc runComposeDynamicTests =
  ## Layouts bound to symbols must match the literal form.
  block:
    let b = make_layout(4, 1)
    let c = compose(make_layout(12, 1), b)
    check c.shape, 4, Int[4]
    check c.stride, 1, Int[1]
  block:
    let a = make_layout(16, 2)
    let b = make_layout(4, 2)
    let c = compose(a, b)
    check c.shape, 4, Int[4]
    check c.stride, 4, Int[4]
  block:
    let c = compose(make_layout((12, 3), (1, 24)), make_layout(4, 1))
    check c.shape, 4, Int[4]
    check c.stride, 1, Int[1]
  block:
    let c = compose(make_layout((128, 24, 5), (1, 128, 3072)), make_layout(64, 2))
    check c.shape, 64, Int[64]
    check c.stride, 2, Int[2]
  block:
    let c = compose(make_layout((128, 24, 5), (1, 128, 3072)), make_layout(480, 32))
    check c.shape, 480, Int[480]
    check c.stride, 32, Int[32]
  block:
    let b = make_layout(4, 1)
    assertCompositionProperty(make_layout(12, 1), b)
  block:
    let a = make_layout(16, 2)
    let b = make_layout(4, 2)
    assertCompositionProperty(a, b)
  echo "  symbol-bound: 5 cases OK"

# ── zero-stride RHS ──────────────────────────────────────────
proc runComposeZeroStrideTests =
  # a static stride-0 RHS dimension maps every coordinate to offset 0.
  # The composed dimension is the RHS dimension itself, the LHS
  # is never touched
  block:
    let lhs = make_layout((2, 3), (3, 1))
    let r = compose(lhs, make_layout(1, 0))
    check r.shape, 1, Int[1]
    check r.stride, 0, Int[0]
  # the (1):(3) filler inside a logical_divide of make_layout((2, 3), (3, 1)),
  # the pipeline behind max_alignment
  block:
    let lhs = make_layout((2, 3), (3, 1))
    let permuted = logical_divide(lhs, right_inverse(lhs))
    check size(make_layout(permuted.shape[0], permuted.stride[0])), 6, Int[6]
    check permuted.stride[1], Int[3](), Int[3]
  echo "  zero-stride: 2 cases OK"

# ── symbol-bound zero dispatch ───────────────────────────────
proc runComposeSymbolZeroTests =
  ## Statics bound in a symbol's field types drive the dispatch:
  ## stride-0 and shape-1 RHS dispatch off the field types,
  ## value expressions are field reads opaque to getStaticInt
  block:
    # a symbol-bound stride-0 RHS maps every coordinate to offset 0
    let b = make_layout(4, 0)
    let c = compose(make_layout(4, 1), b)
    check c.shape, 4, Int[4]
    check c.stride, 0, Int[0]
    assertCompositionProperty(make_layout(4, 1), b)
  block:
    let b = make_layout(1, 0)
    let c = compose(make_layout(4, 1), b)
    check c.shape, 1, Int[1]
    check c.stride, 0, Int[0]
    assertCompositionProperty(make_layout(4, 1), b)
  block:
    # a symbol-bound shape-1 RHS evaluates the LHS at one point
    let b = make_layout(1, 2)
    let c = compose(make_layout(4, 2), b)
    check c.shape, 1, Int[1]
    check c.stride, 4, Int[4]
    assertCompositionProperty(make_layout(4, 2), b)
  block:
    let a = make_layout(4, 1)
    let b = make_layout(4, 0)
    let c = compose(a, b)
    check c.shape, 4, Int[4]
    check c.stride, 0, Int[0]
    assertCompositionProperty(a, b)
  block:
    # a symbol-bound stride-0 LHS mode, in-domain RHS
    let a = make_layout((4, 3, 1), (3, 1, 0))
    let b = make_layout(4, 1)
    let c = compose(a, b)
    check c.shape, 4, Int[4]
    check c.stride, 3, Int[3]
    assertCompositionProperty(a, b)
  block:
    let a = make_layout((2, 3), (3, 1))
    let b = make_layout(1, 0)
    let c = compose(a, b)
    check c.shape, 1, Int[1]
    check c.stride, 0, Int[0]
    assertCompositionProperty(a, b)
  block:
    let b = make_layout(4, 0)
    let c = compose(make_layout(12, 1), b)
    check c.shape, 4, Int[4]
    check c.stride, 0, Int[0]
    assertCompositionProperty(make_layout(12, 1), b)
  echo "  symbol zero dispatch: 7 cases OK"

# ── symbol-bound tiler ───────────────────────────────────────
proc runComposeSymbolTilerTests =
  # a tiler bound to a symbol must match the literal tuple
  block:
    let t = (16, 2)
    let lit = compose(make_layout((32, 8), (1, 32)), (16, 2))
    let viaSym = compose(make_layout((32, 8), (1, 32)), t)
    doAssert viaSym === lit
  block:
    let t = (make_layout(16, 1), 2)
    let lit = compose(make_layout((32, 8), (1, 32)), (make_layout(16, 1), 2))
    let viaSym = compose(make_layout((32, 8), (1, 32)), t)
    doAssert viaSym === lit
  echo "  symbol tiler: 2 cases OK"

# ── coordinate strides ────────────────────────────────────────
proc runComposeCoordStrideTests =
  # LHS Coords: a basis-stride LHS under int tilers
  block:
    let c = compose(make_layout(12, E(0)), make_layout((4, 3), (1, 4)))
    doAssert c === ((4, 3), (E(0), E(4, 0)))
  block:
    let c = compose(make_layout(12, E(2, 1)), make_layout((4, 3), (1, 4)))
    doAssert c === ((4, 3), (E(2, 1), E(8, 1)))
  block:
    let c = compose(make_layout(12, E(0)), make_layout((4, 3), (3, 1)))
    doAssert c === ((4, 3), (E(3, 0), E(0)))
  block:
    let c = compose(make_layout(12, E(2, 1)), make_layout((4, 3), (3, 1)))
    doAssert c === ((4, 3), (E(6, 1), E(2, 1)))
  block:
    let a = make_layout((4, 3), (E(0), E(1)))
    let c = compose(a, make_layout((4, 3), (1, 4)))
    doAssert c === ((4, 3), (E(0), E(1)))
  block:
    let a = make_layout((4, 3), (E(0), E(1)))
    let c = compose(a, make_layout(12, 1))
    doAssert c === ((4, 3), (E(0), E(1)))
  block:
    let c = compose(make_layout((4, 3), (E(0), E(1))), make_layout(6, 2))
    doAssert c === ((2, 3), (E(2, 0), E(1)))
  block:
    let c = compose(make_layout((4, 3), (E(0), E(1))), make_layout((6, 2), (2, 1)))
    doAssert c === (((2, 3), 2), ((E(2, 0), E(1)), E(0)))
  block:
    let c = compose(make_layout((4, 3), (E(1), E(0))), make_layout(6, 2))
    doAssert c === ((2, 3), (E(2, 1), E(0)))
  block:
    let c = compose(make_layout((4, 3), (E(1), E(0))), make_layout((6, 2), (2, 1)))
    doAssert c === (((2, 3), 2), ((E(2, 1), E(0)), E(1)))
  block:
    let c = compose(make_layout((4, 3), (E(6, 1), E(2, 1))), make_layout(6, 2))
    doAssert c === ((2, 3), (E(12, 1), E(2, 1)))
  block:
    let c = compose(make_layout((4, 3), (E(6, 1), E(2, 1))), make_layout((6, 2), (2, 1)))
    doAssert c === (((2, 3), 2), ((E(12, 1), E(2, 1)), E(6, 1)))
  # LHS Coords: multi-term strides under a mode-whole tiler
  block:
    let a = make_layout((4, 4), (E((1, 1)), E((3, 1))))
    let c = compose(a, make_layout((4, 2), (2, 1)))
    doAssert c === (((2, 2), 2), ((E((2, 2)), E((3, 1))), E((1, 1))))
  # LHS Coords: rank-3 basis strides
  block:
    let a = make_layout((4, 6, 8), (E(0), E(1), E(2)))
    let c = compose(a, make_layout((2, 2, 2), (1, 2, 4)))
    doAssert c === ((2, 2, 2), (E(0), E(2, 0), E(1)))
  # RHS Coords: a basis-stride tiler over an int LHS
  block:
    let c = compose(make_layout((4, 4), (4, 1)), make_layout((4, 4), (E(0), E(1))))
    doAssert c === ((4, 4), (4, 1))
  block:
    let c = compose(make_layout((4, 4), (4, 1)), make_layout((4, 4), (E(1), E(0))))
    doAssert c === ((4, 4), (1, 4))
  block:
    let c = compose(make_layout((4, 5), (5, 1)), make_layout(30, E(0)))
    doAssert c === (30, 5)
  block:
    let c = compose(make_layout((4, 5), (5, 1)), make_layout(12, E(1)))
    doAssert c === (12, 1)
  block:
    let c = compose(make_layout((4, 6, 8), (1, 4, 24)), make_layout((2, 2, 2), (E(0), E(1), E(2))))
    doAssert c === ((2, 2, 2), (1, 4, 24))
  block:
    let c = compose(make_layout((4, 6, 8), (1, 4, 24)), make_layout((2, 2, 2), (E(2), E(0), E(1))))
    doAssert c === ((2, 2, 2), (24, 1, 4))
  block:
    let a = make_layout((4, 6, 8), (E(0), E(1), E(2)))
    let c = compose(a, make_layout((2, 2, 2), (E(0), E(1), E(2))))
    doAssert c === ((2, 2, 2), (E(0), E(1), E(2)))
  block:
    let a = make_layout((3, 5, 7, 11), (E(0), E(1), E(2), E(3)))
    let c = compose(a, make_layout(3, E(4, 2)))
    doAssert c === (3, E(4, 2))
  block:
    let a = make_layout((3, 5, 7, 11), (E(0), E(1), E(2), E(3)))
    let c = compose(a, make_layout(3, E((0, 0, 4, 2))))
    doAssert c === (3, E((0, 0, 4, 2)))
  block:
    let a = make_layout((3, 5, 7, 11), (E(0), E(1), E(2), E(3)))
    let c = compose(a, make_layout(3, E((1, 0, 0, 1))))
    doAssert c === (3, E((1, 0, 0, 1)))
  # Diag: an int LHS under a multi-term tiler
  block:
    let c = compose(make_layout((4, 4), (3, 42)), make_layout(4, E((1, 1))))
    doAssert c === (4, 45)
  block:
    let c = compose(make_layout((4, 8), (3, 42)), make_layout(4, E((1, 2))))
    doAssert c === (4, 87)
  echo "  coordinate strides (pycute suite): 27 cases OK"

runComposeExactValueTests()
runComposeFixtureTests()
runComposeSimpleTests()
runComposeMultiDimensionTests()
runComposeRemainderTests()
runComposeNestedTests()
runComposeNegStrideTests()
runComposeTilerTests()
runComposeDynamicTests()
runComposeZeroStrideTests()
runComposeSymbolZeroTests()
runComposeSymbolTilerTests()
runComposeCoordStrideTests()

echo "ALL TESTS PASSED"
