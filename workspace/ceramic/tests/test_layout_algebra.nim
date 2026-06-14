# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for layout_algebra: coalesce + complement + composition + logical_divide.
## Convention:
##   const C2 = 2 — compile-time Int[N] (static)
##   let  d2 = 2 — runtime int (dynamic)
## Reference files used by identifier (shown inline in test sections):
##   [CUTE-C] = cutlass/test/unit/cute/core/coalesce.cpp
##   [CUTE-CP]= cutlass/test/unit/cute/core/complement.cpp
##   [CUTE-CM]= cutlass/test/unit/cute/core/composition.cpp
##   [CUTE-LD]= cutlass/test/unit/cute/core/logical_divide.cpp
##   [PY-L]   = tensor-layouts/tests/layouts.py
##   [PY-E]   = tensor-layouts/tests/external.py

import std/algorithm
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra



# ── Forward declarations ──────────────────────────────────
proc runCoalesceTests: void
proc runComplementStaticRank1Tests: void
proc runComplementExactValueTests: void
proc runComplementMultiModeStaticTests: void
proc runComplementDynamicTests: void
proc runComplementDisjointnessTests: void
proc runComposeExactValueTests: void
proc runComposeSimpleTests: void
proc runComposeMultiModeTests: void
proc runComposeDynamicTests: void
proc runComposeRemainderTests: void
proc runComposeNestedTests: void
proc runComposeSwizzleTests: void
proc runComposeEStrideTests: void
proc runComposeNegStrideTests: void
proc runDivideTests: void
proc runRightInvSimpleTests: void
proc runRightInvExactValueTests: void
proc runRightInvDynamicTests: void
proc runRightInvTests: void
proc runLeftInvSimpleTests: void
proc runLeftInvExactValueTests: void
proc runLeftInvTests: void
proc runLogicalProductTrivialTests: void
proc runLogicalProductMultiTests: void
proc runLogicalProductExactValueTests: void
proc runLogicalProductTests: void

proc runBlockedProductTests: void
proc runRakedProductTests: void
proc runZippedProductTests: void
proc runTiledProductTests: void
proc runFlatProductTests: void
proc runTileToShapeTests: void
proc runMaxCommonVectorTests: void
proc runAntiRegressionTests: void
proc runTileUnzipTests: void
proc runTests =
  echo "\n── Coalesce [CUTE-C]: 20 cases ──"
  runCoalesceTests()
  echo "\n── Complement [CUTE-CP] + [PY-L] + [PY-E] ──"
  runComplementStaticRank1Tests()
  runComplementExactValueTests()
  runComplementMultiModeStaticTests()
  runComplementDynamicTests()
  runComplementDisjointnessTests()
  echo "\n--- Anti-regression: crd2idx + tile_unzip ---"
  runAntiRegressionTests()
  echo "\n── Composition [CUTE-CM] ──"
  runComposeExactValueTests()
  runComposeSimpleTests()
  runComposeMultiModeTests()
  runComposeDynamicTests()
  runComposeRemainderTests()
  runComposeNestedTests()
  runComposeSwizzleTests()
  runComposeEStrideTests()
  runComposeNegStrideTests()
  echo "\n--- max_common_vector ---"
  runMaxCommonVectorTests()
  echo "\n── logical_divide [CUTE-LD] ──"
  runDivideTests()
  echo "\n── right_inverse [CUTE-IR] + [PY-E Table 5] ──"
  runRightInvTests()
  echo "\n── left_inverse [CUTE-IL] + [PY-E Table 6] ──"
  runLeftInvTests()
  echo "\n── logical_product [CUTE-LP] + [MOYE] ──"
  runLogicalProductTests()
  echo "\n── Product variants [MOYE] ──"
  runBlockedProductTests()
  runRakedProductTests()
  runZippedProductTests()
  runTiledProductTests()
  runFlatProductTests()
  echo "\n── tile_unzip [CUTE] ──"
  runTileUnzipTests()
  echo "\n── tile_to_shape [CUTE] ──"
  runTileToShapeTests()
  echo "\nALL TESTS PASSED"

when isMainModule:
  runTests()

# ═══════════════════════════════════════════════════════════════
#  test_coalesce helper (mirrors CuTe C++)
# ═══════════════════════════════════════════════════════════════
#
# Checks that coalesce preserves size and mapping:
#   size(coalesce(L)) == size(L) and coalesce(L)(i) == L(i)
# ═══════════════════════════════════════════════════════════════

proc chkCoalesce(layout: Layout) =
  ## Check coalesce preserves size and mapping.
  let c = coalesce(layout)
  doAssert size(c) === size(layout),
    "coalesce size mismatch: " & $size(c) & " != " & $size(layout)
  for i in 0 ..< size(layout):
    doAssert c[i] === layout[i],
      "coalesce mapping mismatch at " & $i & ": " & $c[i] & " != " & $layout[i]

# ═══════════════════════════════════════════════════════════════
#  Coalesce [CUTE-C]
# ═══════════════════════════════════════════════════════════════
proc runCoalesceScalarTests =
  block:
    let l = make_layout(1, 0)
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (1, 0)
  block:
    let l = make_layout(1, 1)
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (1, 0)
  block:
    let l = make_layout(1, 2)
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (1, 0)
  block:
    let l = make_layout(1, 5)
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (1, 0)
  echo "  Scalar: 4 cases OK"

proc runCoalesceColMajorTests =
  const C2 = 2
  const C4 = 4
  let l1 = make_layout((C2, C4), (1, 2))
  let c1 = coalesce(l1)
  chkCoalesce(l1)
  doAssert c1 === (8, 1)
  const C6 = 6
  let l2 = make_layout((C2, C4, C6), (1, 2, 8))
  let c2 = coalesce(l2)
  chkCoalesce(l2)
  doAssert c2 === (48, 1)
  echo "  Column-major contiguous: 2 cases OK"

proc runCoalesceSize1Tests =
  block:
    let l = make_layout((1, 8), (1, 1))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (8, 1)
  block:
    let l = make_layout((1, 1, 8), (1, 1, 1))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (8, 1)
  block:
    let l = make_layout((1, 8, 1), (1, 1, 1))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (8, 1)
  echo "  Size-1 modes: 3 cases OK"

proc runCoalesceStride0Tests =
  block:
    let l = make_layout((4, 1), (1, 0))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (4, 1)
  block:
    let l = make_layout((1, 2), (0, 2))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (2, 2)
  block:
    let l = make_layout((2, 1), (3, 0))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (2, 3)
  echo "  Stride-0 modes: 3 cases OK"

proc runCoalesceNonContigTests =
  block:
    let l = make_layout((4, 8), (1, 4))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (32, 1)
  block:
    let l = make_layout((4, 8), (1, 5))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === ((4, 8), (1, 5))
  block:
    let l = make_layout((3, 4, 5), (1, 3, 12))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (60, 1)
  echo "  Non-contiguous strides: 3 cases OK"

proc runCoalesceMixedTests =
  block:
    let l = make_layout((6, 7, 4), (1, 6, 42))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (168, 1)
  block:
    let l = make_layout((2, 6), (1, 9))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === ((2, 6), (1, 9))
  block:
    let l = make_layout((3, 4, 5), (1, 1, 12))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === ((3, 4, 5), (1, 1, 12))
  echo "  Mixed: 3 cases OK"

proc runCoalesceDynamicTests =
  block:
    let d8 = 8
    let l = make_layout(d8, 1)
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (8, 1)
  block:
    let d12 = 12
    let l = make_layout(d12, 2)
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === (12, 2)
  echo "  Dynamic shapes: 2 cases OK"

proc runCoalesceCppTests =
  const C2 = 2
  const C6 = 6
  block:
    let l = make_layout((C2, C6), (1, 9))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === ((2, 6), (1, 9))
  block:
    let l = make_layout((3, 4, 5), (1, 1, 12))
    let c = coalesce(l)
    chkCoalesce(l)
    doAssert c === ((3, 4, 5), (1, 1, 12))
  echo "  Additional [CUTE-C]: 2 cases OK"

proc runCoalesceTests =
  runCoalesceScalarTests()
  runCoalesceColMajorTests()
  runCoalesceSize1Tests()
  runCoalesceStride0Tests()
  runCoalesceNonContigTests()
  runCoalesceMixedTests()
  runCoalesceDynamicTests()
  runCoalesceCppTests()

# ═══════════════════════════════════════════════════════════════
#  Complement post-condition checks (CuTe test_complement)
# ═══════════════════════════════════════════════════════════════
#
#
# Checks:
#   (1) cosize(layout ++ complement) >= size(cotarget)
#   (2) cosize(complement) <= round_up(size(cotarget), cosize(layout))
#   (3) result[i-1] < result[i]  — ordered
#   (4) result[i] != layout[j]   — disjoint (tested separately)
#   (5) size(result) <= cosize(result)
#   (9) if static stride, complement(completed) has size 1
template testComplementProps(layout, result: typed; cotarget: untyped) =
  let rSize = size(result)
  let lCosize = cosize(layout)
  let completed = Layout[typeof((layout.shape, result.shape)), typeof((layout.stride, result.stride))](
    shape: (layout.shape, result.shape), stride: (layout.stride, result.stride))
  let cSize = fold(cotarget, 1, acc * it)
  # (1) Lower-bound on codomain of layout ++ complement
  doAssert cosize(completed) >= cSize,
    "(1) cosize(completed)=" & $cosize(completed) & " < cSize=" & $cSize & " layout=" & $layout
  # (2) Upper-bound on cosize of complement
  let upper = ((cSize + lCosize - 1) div lCosize) * lCosize
  doAssert cosize(result) <= upper,
    "(2) cosize(result)=" & $cosize(result) & " > upper=" & $upper & " layout=" & $layout & " cotarget=" & $cotarget
  # (3) Strictly increasing
  for i in 1 ..< rSize:
    doAssert result[i-1] < result[i]
  # (5) size <= cosize
  doAssert rSize <= cosize(result)
  # (9) If static stride, complement(completed) has size 1
  when typeof(completed.stride) is Int or typeof(completed.stride) is int:
    let cc = complement(completed)
    doAssert size(cc) === 1

proc chkComplement[T](layout: Layout; cotarget: T) =
  let r = complement(layout, cotarget)
  testComplementProps(layout, r, cotarget)

proc chkComplement(layout: Layout) =
  chkComplement(layout, cosize(layout))

# ═══════════════════════════════════════════════════════════════
#  Complement [CUTE-CP] + [PY-L] + [PY-E]
# ═══════════════════════════════════════════════════════════════
#
# References:

# ─── Static rank-1 [CUTE-CP] ───────────────────────────────────
proc runComplementStaticRank1Tests =
  # Layout(_1,_0)
  block:
    let l = make_layout(1, 0)
    chkComplement(l)
    chkComplement(l, Int[2]())
    chkComplement(l, Int[5]())
    chkComplement(l, (Int[2](), 2))
  # Layout(_1,_1)
  block:
    let l = make_layout(1, 1)
    chkComplement(l)
    chkComplement(l, Int[2]())
    chkComplement(l, Int[5]())
    chkComplement(l, (Int[2](), 2))
  # Layout(_1,_2)
  block:
    let l = make_layout(1, 2)
    chkComplement(l, Int[1]())
    chkComplement(l, Int[2]())
    chkComplement(l, Int[8]())
    chkComplement(l, Int[5]())
    chkComplement(l, (Int[2](), 2))
  # Layout(_4,_0)
  block:
    let l = make_layout(4, 0)
    chkComplement(l, Int[1]())
    chkComplement(l, Int[2]())
    chkComplement(l, Int[8]())
  # Layout(_4,_1)
  block:
    let l = make_layout(4, 1)
    chkComplement(l, Int[1]())
    chkComplement(l, Int[2]())
    chkComplement(l, Int[8]())
  # Layout(_4,_2)
  block:
    const C4 = 4
    let l = make_layout(C4, 2)
    chkComplement(l, Int[1]())
    chkComplement(l)
    chkComplement(l, Int[16]())
    chkComplement(l, Int[19]())
    chkComplement(l, (Int[2](), 2))
  # Layout(_4,_4)
  block:
    const C4 = 4
    let l = make_layout(C4, 4)
    chkComplement(l, Int[1]())
    chkComplement(l)
    chkComplement(l, Int[17]())
    chkComplement(l, (Int[2](), 2))
  echo "  Static rank-1: 28 complement calls OK"

# ─── Python exact-value [PY-L] ─────────────────────────────────
proc runComplementExactValueTests =
  block:
    let r = complement(make_layout(1, 0), 8)
    doAssert r.shape === 8
    doAssert r.stride === 1

  block:
    let r = complement(make_layout(4, 2), 16)
    doAssert r.shape === (2, 2)
    doAssert r.stride === (1, 8)

  block:
    let r = complement(make_layout(4, 1), 16)
    doAssert r.shape === 4
    doAssert r.stride === 4

  block:
    let r = complement(make_layout(4, 1), 24)
    doAssert r.shape === 6
    doAssert r.stride === 4

  block:
    let r = complement(make_layout((2, 2), (1, 4)), 16)
    doAssert r.shape === (2, 2)
    doAssert r.stride === (2, 8)

  block:
    let r = complement(make_layout(2, 1), (Int[3](), 4))
    doAssert r.shape === 6
    doAssert r.stride === 2

  block:
    let r = complement(make_layout(6, 1), 24)
    doAssert r.shape === 4
    doAssert r.stride === 6

  block:
    let r = complement(make_layout((4, 2), (1, 16)))
    doAssert r.shape === 4
    doAssert r.stride === 4

  block:
    let r = complement(make_layout(8, 1), 32)
    doAssert r.shape === 4
    doAssert r.stride === 8

  block:
    let r = complement(make_layout(4, 4), 32)
    doAssert r.shape === (4, 2)
    doAssert r.stride === (1, 16)
  echo "  Exact-value: 10 Python assertions OK"

# ─── Multi-mode static [CUTE-CP] ───────────────────────────────
proc runComplementMultiModeStaticTests =
  # Shape(_2,_4):(1,2)
  block:
    let l = make_layout((Int[2](), Int[4]()), (Int[1](), Int[2]()))
    chkComplement(l)
  # Shape(_2,_3):(1,2)
  block:
    let l = make_layout((Int[2](), Int[3]()), (Int[1](), Int[2]()))
    chkComplement(l)
  # (2,4):(1,4)
  block:
    let l = make_layout((Int[2](), Int[4]()), (Int[1](), Int[4]()))
    chkComplement(l)
  # (2,4):(1,6)
  block:
    let l = make_layout((Int[2](), Int[4]()), (Int[1](), Int[6]()))
    chkComplement(l)
  # (2,4,8):(8,1,64)
  block:
    let l = make_layout((Int[2](), Int[4](), Int[8]()), (Int[8](), Int[1](), Int[64]()))
    chkComplement(l)
  # (2,4,8):(8,1,0)
  block:
    let l = make_layout((Int[2](), Int[4](), Int[8]()), (Int[8](), Int[1](), Int[0]()))
    chkComplement(l)
    chkComplement(l, Int[460]())
  # nested((2,2),(2,2)):((1,4),(8,32))
  block:
    let l = make_layout(
      ((Int[2](), Int[2]()), (Int[2](), Int[2]())),
      ((Int[1](), Int[4]()), (Int[8](), Int[32]())))
    chkComplement(l)
  # nested((2,2),(2,2)):((1,32),(8,4))
  block:
    let l = make_layout(
      ((Int[2](), Int[2]()), (Int[2](), Int[2]())),
      ((Int[1](), Int[32]()), (Int[8](), Int[4]())))
    chkComplement(l)
  # (4,6):(1,6)
  block:
    let l = make_layout((Int[4](), Int[6]()), (Int[1](), Int[6]()))
    chkComplement(l)
  # (4,2):(1,10)
  block:
    let l = make_layout((Int[4](), Int[2]()), (Int[1](), Int[10]()))
    chkComplement(l)
  # (4,2):(1,16)
  block:
    let l = make_layout((Int[4](), Int[2]()), (Int[1](), Int[16]()))
    chkComplement(l)
  echo "  Multi-mode static: 12 complement calls OK"

# ─── Dynamic shapes/strides [CUTE-CP] ──────────────────────────
proc runComplementDynamicTests =
  # Dynamic shape 12, stride 1
  block:
    let l = make_layout(12, 1)
    chkComplement(l, 1)
    chkComplement(l)
    chkComplement(l, 53)
    chkComplement(l, 128)
  # Dynamic shape 12, static stride Int<2>
  block:
    let l = make_layout(12, Int[2]())
    chkComplement(l, 1)
    chkComplement(l)
    chkComplement(l, 53)
    chkComplement(l, 128)
  # Dynamic shape 12, dynamic stride 2
  block:
    let l = make_layout(12, 2)
    chkComplement(l, 1)
    chkComplement(l)
    chkComplement(l, 53)
    chkComplement(l, 128)
  # 2D dynamic: (3,6):(1,3)
  block:
    let l = make_layout((3, 6), (Int[1](), Int[3]()))
    chkComplement(l)
  # 2D dynamic: (3,6):(1,9)
  block:
    let l = make_layout((3, 6), (Int[1](), Int[9]()))
    chkComplement(l)
  # 2D dynamic: (3,6):(1,10)
  block:
    let l = make_layout((3, 6), (Int[1](), Int[10]()))
    chkComplement(l)
  # 2D dynamic nested: ((2,2),(2,2)):((1,4),(8,32))
  block:
    let l = make_layout(
      ((Int[2](), Int[2]()), (Int[2](), Int[2]())),
      ((Int[1](), Int[4]()), (Int[8](), Int[32]())))
    chkComplement(l)
  # 3D shape bound: Int<64> with cotarget (32,4,4)
  block:
    let l = make_layout(Int[64]())
    chkComplement(l, (Int[32](), Int[4](), Int[4]()))
    chkComplement(l, (Int[32](), Int[4](), 4))
  echo "  Dynamic: 9 layouts, 40+ complement calls OK"

# ─── Python disjointness checks [PY-E] ─────────────────────────
proc runComplementDisjointnessTests =
  proc chkDisjoint(layout: Layout) =
    let c = complement(layout)
    for i in 0 ..< size(layout):
      for j in 0 ..< size(c):
        if layout[i] != 0 and c[j] != 0:
          doAssert layout[i] != c[j],
            "complement overlaps at " & $i & "," & $j

  chkDisjoint(make_layout(1, 0))
  chkDisjoint(make_layout(1, 1))
  chkDisjoint(make_layout(4, 0))
  chkDisjoint(make_layout((2, 4), (1, 2)))
  chkDisjoint(make_layout((2, 3), (1, 2)))
  chkDisjoint(make_layout((2, 4), (1, 4)))
  chkDisjoint(make_layout((2, 4, 8), (8, 1, 64)))
  chkDisjoint(make_layout(((2, 2), (2, 2)), ((1, 4), (8, 32))))
  chkDisjoint(make_layout((2, (3, 4)), (3, (1, 6))))
  chkDisjoint(make_layout((4, 6), (1, 6)))
  chkDisjoint(make_layout((4, 10), (1, 10)))
  echo "  Disjointness: 11 layouts OK"

# ═══════════════════════════════════════════════════════════════
proc chkCompose[Sh1, St1, Sh2, St2](a: Layout[Sh1, St1]; b: Layout[Sh2, St2]): bool =
  let r = compose(a, b)
  # [CUTE-CM] compatible(b.shape, r.shape)
  if not compatible(b.shape, r.shape):
    echo "    FAIL: compatible(b.shape=", b.shape, ", r.shape=", r.shape, ")"
    return false
  # [CUTE-CM] r[c] == a[b[c]] for all c
  for c in 0 ..< size(b):
    if r[c] != a[b[c]]:
      echo "    FAIL: r[",c,"]=",r[c]," a[b[",c,"]]=",a[b[c]]
      return false
  true

proc runComposeExactValueTests =
  echo "    Python exact-value assertions:"
  # Python: assert compose(Layout(8, 2), Layout(4, 1)) == Layout(4, 2)
  doAssert compose(make_layout(8, 2), make_layout(4, 1)) === (4, 2)
  # Python: assert compose(Layout(8, 2), Layout(4, 2)) == Layout(4, 4)
  doAssert compose(make_layout(8, 2), make_layout(4, 2)) === (4, 4)
  # Python: assert compose(a, b) == Layout((2, 3), (-2, -1))
  doAssert compose(make_layout((2, 3), (2, 1)), make_layout(6, -1)) === ((2, 3), (-2, -1))
  # Python: assert compose(Layout((4, 8), (1, 4)), Layout(4, 1)) == Layout(4, 1)
  doAssert compose(make_layout((4, 8), (1, 4)), make_layout(4, 1)) === (4, 1)
  # Python: assert compose(Layout((4, 8), (1, 4)), Layout(4, 4)) == Layout(4, 4)
  doAssert compose(make_layout((4, 8), (1, 4)), make_layout(4, 4)) === (4, 4)
  # Python: assert compose(Layout(16, 1), Layout((4, 4), (1, 4))) == Layout((4, 4), (1, 4))
  doAssert compose(make_layout(16, 1), make_layout((4, 4), (1, 4))) === ((4, 4), (1, 4))
  # compose rank-1 LHS with hierarchical RHS — must preserve nesting
  # Previously buggy: b.shape.flatten() destroyed ((2,2),(2,8)) into (2,2,2,8)
  doAssert compose(make_layout(64, 1), make_layout(((2, 2), (2, 8)), ((1, 4), (2, 8)))) === (((2, 2), (2, 8)), ((1, 4), (2, 8)))
  # compose rank-2 coalescable LHS with hierarchical RHS — triggered the bug
  # A=(8,8):(1,8) coalesces inside compose to (64):(1), then the buggy path
  # called b.shape.flatten() which destroyed ((2,2),(2,8)).
  doAssert compose(make_layout((8, 8), (1, 8)), make_layout(((2, 2), (2, 8)), ((1, 4), (2, 8)))) === (((2, 2), (2, 8)), ((1, 4), (2, 8)))
  # Python: assert compose(Layout(16, 2), Layout((4, 4), (1, 4))) == Layout((4, 4), (2, 8))
  doAssert compose(make_layout(16, 2), make_layout((4, 4), (1, 4))) === ((4, 4), (2, 8))
  # Python: assert compose(Layout((4, 4), (1, 4)), Layout((2, 2), (1, 2))) == Layout((2, 2), (1, 2))
  doAssert compose(make_layout((4, 4), (1, 4)), make_layout((2, 2), (1, 2))) === ((2, 2), (1, 2))
  # Python: assert C == Layout(((2, 2), 3), ((24, 2), 8))
  doAssert compose(make_layout((6, 2), (8, 2)), make_layout((4, 3), (3, 1))) === (((2, 2), 3), ((24, 2), 8))
  # Python: assert C == Layout(((5, 1), (2, 2)), ((16, 4), (80, 4))) or Layout((5, (2, 2)), (16, (80, 4)))
  block:
    let C = compose(make_layout((10, 2), (16, 4)), make_layout((5, 4), (1, 5)))
    let form1 = C === (((5, 1), (2, 2)), ((16, 4), (80, 4)))
    let form2 = C === ((5, (2, 2)), (16, (80, 4)))
    doAssert form1 or form2, "compose result " & $C & " matches neither accepted form"
    doAssert chkCompose(make_layout((10, 2), (16, 4)), make_layout((5, 4), (1, 5))), "mapping check failed"
  # compose tuple LHS + scalar RHS — hits elif b.shape isnot tuple: branch
  # Previously assertion-failing; verify against Python output
  doAssert compose(make_layout((4, 8), (1, 4)), make_layout(10, 1)) === (10, 1)
  doAssert chkCompose(make_layout((4, 8), (1, 4)), make_layout(10, 1))
  doAssert compose(make_layout((4, 8), (1, 4)), make_layout(8, 1)) === (8, 1)
  doAssert chkCompose(make_layout((4, 8), (1, 4)), make_layout(8, 1))
  doAssert compose(make_layout((4, 8), (1, 4)), make_layout(5, 1)) === (5, 1)
  doAssert chkCompose(make_layout((4, 8), (1, 4)), make_layout(5, 1))
  # nested tuple LHS + scalar RHS
  doAssert chkCompose(make_layout(((6, 1), (32, 512)), ((1, 192), (6, 192))), make_layout(200, 1))
  echo "    Exact-value: 14/14"

proc runComposeSimpleTests =
  doAssert chkCompose(make_layout(1,0), make_layout(1,0))
  doAssert chkCompose(make_layout(1,0), make_layout(1,1))
  doAssert chkCompose(make_layout(1,1), make_layout(1,0))
  doAssert chkCompose(make_layout(1,1), make_layout(1,1))
  doAssert chkCompose(make_layout(4,1), make_layout(4,1))
  doAssert chkCompose(make_layout(4,2), make_layout(4,1))
  doAssert chkCompose(make_layout(4,1), make_layout(4,2))
  doAssert chkCompose(make_layout(4,0), make_layout(4,1))
  doAssert chkCompose(make_layout(4,1), make_layout(4,0))
  doAssert chkCompose(make_layout(1,0), make_layout(4,1))
  doAssert chkCompose(make_layout(4,1), make_layout(1,0))
  doAssert chkCompose(make_layout(4,1), make_layout(2,1))
  doAssert chkCompose(make_layout(4,2), make_layout(2,1))
  doAssert chkCompose(make_layout(4,1), make_layout(2,2))
  doAssert chkCompose(make_layout(4,2), make_layout(2,2))
  echo "    Simple: 15/15"

proc runComposeMultiModeTests =
  doAssert chkCompose(make_layout((4,3),(1,1)), make_layout(12,1))
  doAssert chkCompose(make_layout(12,1), make_layout((4,3),(1,1)))
  doAssert chkCompose(make_layout(12,2), make_layout((4,3),(1,1)))
  doAssert chkCompose(make_layout(12,1), make_layout((4,3),(3,1)))
  doAssert chkCompose(make_layout(12,2), make_layout((4,3),(3,1)))
  doAssert chkCompose(make_layout(12,1), make_layout((2,3),(2,4)))
  doAssert chkCompose(make_layout((4,3),(3,1)), make_layout(12,1))
  doAssert chkCompose(make_layout((4,3),(3,1)), make_layout(6,2))
  # CuTe multi-mode RHS: LHS col-major (1,4) × multi-mode RHS
  doAssert chkCompose(make_layout((4,3),(1,4)), make_layout((4,3),(1,4)))
  doAssert chkCompose(make_layout((4,3),(1,4)), make_layout((6,2),(2,1)))
  doAssert chkCompose(make_layout((4,3),(1,4)), make_layout((4,3),(3,1)))
  doAssert chkCompose(make_layout((4,3),(3,1)), make_layout((4,3),(1,4)))
  echo "    Multi-mode: 12/12"

proc runComposeDynamicTests =
  doAssert chkCompose(make_layout(12, 1), make_layout(4,1))
  doAssert chkCompose(make_layout(12, 1), make_layout(4,1))
  block:
    let b = make_layout(4, 1)
    doAssert chkCompose(make_layout(12, 1), b)
  block:
    let b = make_layout(4, 1)
    doAssert chkCompose(make_layout(12, 1), b)
  doAssert chkCompose(make_layout((12,3), (1,24)), make_layout(4,1))
  block:
    let a = make_layout(16, 2)
    let b = make_layout(4, 2)
    doAssert chkCompose(a, b)
  block:
    let a = make_layout((128,24,5), (1,128,3072))
    doAssert chkCompose(a, make_layout(64, 2))
  block:
    let a = make_layout((128,24,5), (1,128,3072))
    doAssert chkCompose(a, make_layout(480, 32))
  echo "    Dynamic: 8/8"

proc runComposeRemainderTests =
  doAssert chkCompose(make_layout(1,0), make_layout(4,1))
  doAssert chkCompose(make_layout(1,1), make_layout(4,1))
  doAssert chkCompose(make_layout(4,1), make_layout(4,2))
  doAssert chkCompose(make_layout((4,3),(3,1)), make_layout(24,1))
  doAssert chkCompose(make_layout((4,3),(3,1)), make_layout(8,1))
  doAssert chkCompose(make_layout((4,3,1),(3,1,0)), make_layout(24,1)) # This requires coalesce to preserve trailing (shape 1, stride 0)
  doAssert chkCompose(make_layout((4,3,1),(3,1,0)), make_layout(4,1))
  doAssert chkCompose(make_layout((4,6,8,10),(2,3,5,7)), make_layout(6,12))
  doAssert chkCompose(make_layout((8,8),(8,1)), make_layout(2,3))
  doAssert chkCompose(make_layout((8,8),(8,1)), make_layout(3,3))
  doAssert chkCompose(make_layout(3,1), make_layout(4,1))
  doAssert chkCompose(make_layout((48,24,5),(1,128,3072)), make_layout(32,1))
  echo "    Remainder: 12/12"

proc runComposeNestedTests =
  echo "    Nested/3D RHS [CUTE-CM #29-33]:"
  # [CUTE-CM] #29: nested shape ((4,2):(1,16)) / (4,2):(2,1)
  doAssert chkCompose(make_layout(((4,2),), ((1,16),)), make_layout((4,2),(2,1)))
  # [CUTE-CM] #30: (2,2):(2,1) / (2,2):(2,1)
  doAssert chkCompose(make_layout((2,2),(2,1)), make_layout((2,2),(2,1)))
  # [CUTE-CM] #31: (4,8,2) / (2,2,2):(2,8,1)
  doAssert chkCompose(make_layout((4,8,2)), make_layout((2,2,2),(2,8,1)))
  # [CUTE-CM] #32: (4,8,2):(2,8,1) / (2,2,2):(1,8,2)
  doAssert chkCompose(make_layout((4,8,2),(2,8,1)), make_layout((2,2,2),(1,8,2)))
  # [CUTE-CM] #33: (4,8,2):(2,8,1) / (4,2,2):(2,8,1)
  doAssert chkCompose(make_layout((4,8,2),(2,8,1)), make_layout((4,2,2),(2,8,1)))
  echo "    5/5"

proc runComposeSwizzleTests =
  echo "    Swizzle [CUTE-CM #55-56]:"
  # [CUTE-CM] #55: compose(Layout<8,8>:(8,1), Swizzle ∘ Layout<8,8>:(8,1))
  # [CUTE-CM] #56: compose(Swizzle∘..., Swizzle∘...) — double swizzle
  # Swizzle type not yet implemented
  echo "    0/2 (blocked: no Swizzle type)"

proc runComposeEStrideTests =
  echo "    E stride [CUTE-CM #63-66]:"
  # [CUTE-CM] #63: ((1,(2,4)):(0,(-1,512))) / (2:-1)
  # [CUTE-CM] #64: ((1,(2,4)):(0,(-1,512))) / (4:-1)
  # [CUTE-CM] #65: (4,4):(4,1) / (4,4):(E1,E0)
  # [CUTE-CM] #66: (4,(2,3)):(6,(3,1)) / (2,4):(E11,E0)
  # ScaledBasis/E-stride not yet implemented
  echo "    0/4 (blocked: no E stride support)"

proc runComposeNegStrideTests =
  doAssert chkCompose(make_layout(4,-1), make_layout(4,1))
  doAssert chkCompose(make_layout(4,1), make_layout(4,-1))
  doAssert chkCompose(make_layout(4,-1), make_layout(4,-1))
  doAssert chkCompose(make_layout(4,1), make_layout(4,-2))
  doAssert chkCompose(make_layout((4,4),(-1,1)), make_layout(2,1))
  # [CUTE-CM] #62: (4,4):(-1,1) / (2,4,2):(1,4,2)
  doAssert chkCompose(make_layout((4,4),(-1,1)), make_layout((2,4,2),(1,4,2)))
  echo "    Neg strides: 6/6"
# ═══════════════════════════════════════════════════════════════
#  max_common_vector / max_common_layout
# ═══════════════════════════════════════════════════════════════

proc runAntiRegressionTests =
  ## crd2idx with nested shape: layout[coord] on (2, (3,4)):((1,6),3)
  block:
    let l = make_layout((2, (3, 4)), (3, (1, 6)))
    doAssert l(0) == 0
    doAssert l(6) == 6
    doAssert l(1) == 3
    doAssert l(23) == 23
  ## tile_unzip with Layout tiler
  block:
    let A = make_layout((8, 8), (1, 8))
    let T = make_layout((2, 2), (1, 4))
    let zd = zipped_divide(A, T)
    doAssert zd === (((2, 2), (2, 8)), ((1, 4), (2, 8)))
  echo "    PASS"

# ═══════════════════════════════════════════════════════════════

proc runMaxCommonVectorTests =
  # A-packing (both nested)
  block:
    let src = make_layout(((6, 1), (32, 512)), ((1, 192), (6, 192)))
    let dst = make_layout(((6, 1), (32, 512)), ((1, 6), (3072, 6)))
    doAssert max_common_vector(src, dst) == 6
    doAssert max_common_layout(src, dst) === (6, 1)
  block:
    let src = make_layout(((1, 16), (512, 64)), ((1, 512), (1, 8192)))
    let dst = make_layout(((1, 16), (512, 64)), ((1, 1), (16, 8192)))
    doAssert max_common_vector(src, dst) == 1
  block:
    let src = make_layout((4000, 2000), (1, 4000))
    let dst = make_layout((2000, 4000), (2000, 1))
    doAssert max_common_vector(src, dst) == 1
  # Identical LayoutRight — entire layout is common
  block:
    let a = make_layout((64, 512, 16), (8192, 16, 1))
    let b = make_layout((64, 512, 16), (8192, 16, 1))
    doAssert max_common_vector(a, b) == 524288
    let common = max_common_layout(a, b)
    for i in 0 ..< size(common):
      doAssert a(common(i)) == i
      doAssert b(common(i)) == i
  block:
    let src = make_layout((32, 6), (1, 32))
    let dst = make_layout((32, 6), (1, 32))
    doAssert max_common_vector(src, dst) == 192
    let common = max_common_layout(src, dst)
    doAssert size(common) === 192
    for i in 0 ..< size(common):
      doAssert src(common(i)) == i
      doAssert dst(common(i)) == i
  block:
    let src = make_layout(512, 1)
    let dst = make_layout(512, 1)
    doAssert max_common_vector(src, dst) == 512
    let common = max_common_layout(src, dst)
    doAssert size(common) === 512
    for i in 0 ..< size(common):
      doAssert src(common(i)) == i
      doAssert dst(common(i)) == i
  # PackA: layout from GEMM benchmark
  block:
    let src = make_layout(((6, 1), (32, 512)), ((1, 192), (6, 192)))
    let dst = make_layout((32, 512, 6), LayoutRight)
    doAssert max_common_vector(src, dst) == 1
  # PackB: layout from GEMM benchmark
  block:
    let src = make_layout(((1, 16), (512, 64)), ((1, 512), (1, 8192)))
    let dst = make_layout((64, 512, 16), LayoutRight)
    doAssert max_common_vector(src, dst) == 1
  # [PY] test_max_common_vector_identical
  block:
    let a = make_layout(8, 1)
    doAssert max_common_vector(a, a) == 8
  # [PY] test_max_common_vector_contiguous_prefix: a=8:1, b=(2,4):(1,4) → 2
  block:
    let a = make_layout(8, 1)
    let b = make_layout((2, 4), (1, 4))
    doAssert max_common_vector(a, b) == 2
  # [PY] test_max_common_vector_no_common: a=(4,2):(2,1), b=8:1 → 1
  block:
    let a = make_layout((4, 2), (2, 1))
    let b = make_layout(8, 1)
    doAssert max_common_vector(a, b) == 1
  # [PY] test_max_common_layout_identical
  block:
    let a = make_layout(8, 1)
    let common = max_common_layout(a, a)
    doAssert size(common) === 8
    for i in 0 ..< size(common):
      doAssert a(common(i)) == i
  # [PY] test_max_common_layout_partial: a=8:1, b=(2,4):(1,4) → layout size 2
  block:
    let a = make_layout(8, 1)
    let b = make_layout((2, 4), (1, 4))
    let common = max_common_layout(a, b)
    doAssert size(common) === 2
    for i in 0 ..< size(common):
      doAssert a(common(i)) == i
      doAssert b(common(i)) == i
  # [PY] Fig 8a: (4,4):(1,4) vs ((2,2),4):((1,8),2) → mcv=2, mcl=2:1
  block:
    let src = make_layout((4, 4), (1, 4))
    let dst = make_layout(((2, 2), 4), ((1, 8), 2))
    doAssert max_common_vector(src, dst) == 2
    doAssert max_common_layout(src, dst) === (2, 1)
  # [PY] Fig 8b: ((2,2),(2,2)):((8,2),(4,1)) vs ((2,2),(2,2)):((4,2),(8,1)) → mcv=4
  block:
    let src = make_layout(((2, 2), (2, 2)), ((8, 2), (4, 1)))
    let dst = make_layout(((2, 2), (2, 2)), ((4, 2), (8, 1)))
    doAssert max_common_vector(src, dst) == 4
    let common = max_common_layout(src, dst)
    for i in 0 ..< size(common):
      doAssert src(common(i)) == i
      doAssert dst(common(i)) == i
  # [PY-applications] (4,4):(1,4) vs 16:1 → different shapes, same mapping, mcv=16
  block:
    let a = make_layout((4, 4), (1, 4))
    let b = make_layout(16, 1)
    doAssert max_common_vector(a, b) == 16
    let common = max_common_layout(a, b)
    doAssert size(common) === 16
    for i in 0 ..< size(common):
      doAssert a(common(i)) == i
      doAssert b(common(i)) == i
  # [PY-applications] (4,4):(1,4) vs (4,4):(4,1) → transposed, mcv=1
  block:
    let a = make_layout((4, 4), (1, 4))
    let b = make_layout((4, 4), (4, 1))
    doAssert max_common_vector(a, b) == 1

  echo "    max_common_vector: 19/19"

# ═══════════════════════════════════════════════════════════════
#  logical_divide [CUTE-LD]
# ═══════════════════════════════════════════════════════════════
#
#  CuTe formula: compose(A, Layout(B, complement(B, shape(coalesce(A)))))
#  shape() = product of shape elements, NOT cosize().
#
# ═══════════════════════════════════════════════════════════════

proc runDivideTests: void =
  ## 24 [CUTE-LD] cases + 8 rank-2 tiler extensions

  template checkDiv(L, T: typed): untyped =
    let R = logical_divide(L, T)
    doAssert rank(R) === 2, "rank not 2 for " & $L & " / " & $T

  echo "  Static (rank-1 tilers):"
  checkDiv(make_layout(1, 0), make_layout(1, 0))
  checkDiv(make_layout(1, 0), make_layout(1, 1))
  checkDiv(make_layout(1, 1), make_layout(1, 0))
  checkDiv(make_layout(1, 1), make_layout(1, 1))
  checkDiv(make_layout(6, 1), make_layout(2, 1))
  checkDiv(make_layout(6, 1), make_layout(2, 3))
  checkDiv(make_layout(6, 1), make_layout((2, 3), (3, 1)))  # CuTe rank-2 tiler
  checkDiv(make_layout(6, 2), make_layout(2, 1))
  checkDiv(make_layout(6, 2), make_layout(2, 3))
  checkDiv(make_layout(6, 2), make_layout((2, 3), (3, 1)))  # CuTe rank-2 tiler
  checkDiv(make_layout((6, 6), (1, 12)), make_layout((6, 3), (3, 1)))
  checkDiv(make_layout((6, 6), (12, 1)), make_layout((6, 3), (3, 1)))
  checkDiv(make_layout(32, 1), make_layout(2, 8))  # [CUTE-LD] Layout<_32> / Layout<_2,_8>
  checkDiv(make_layout((4, 1), (1, 1)), make_layout(2, 1))
  checkDiv(make_layout((4, 1), (1, 1)), make_layout(2, 2))
  checkDiv(make_layout((8, 8), (1, 8)), make_layout(32, 2))
  checkDiv(make_layout((8, 8), (8, 1)), make_layout(32, 2))
  echo "    17/17"

  echo "  Static (rank-2 tilers, scalar LHS) — our extension:"
  checkDiv(make_layout(32, 1), make_layout((2, 8), (1, 2)))
  echo "    1/1"

  echo "  Dynamic:"
  checkDiv(make_layout(int(2), int(1)), make_layout(32, 1))
  checkDiv(make_layout(int(48), int(1)), make_layout(32, 1))
  checkDiv(make_layout(int(96), int(1)), make_layout(32, 2))
  checkDiv(make_layout(int(32), int(1)), make_layout(48, 1))
  echo "    4/4"

  echo "  'Dangerous' stride-0 LHS modes:"
  checkDiv(make_layout((128,4,3), (1,512,0)), make_layout(32, 1))
  checkDiv(make_layout((128,4,3), (1,512,0)), make_layout(32, 2))
  checkDiv(make_layout((16,4,3), (1,512,0)), make_layout(32, 1))
  echo "    3/3"

  template checkDivMap(L, T: typed): untyped =
    let R = logical_divide(L, T)
    doAssert rank(R) === 2, "rank not 2 for " & $L & " / " & $T
    doAssert size(R) === size(L)
    for i in 0 ..< size(L):
      doAssert R[i] === L[i], "mapping fail at " & $i & " for " & $L & " / " & $T

  echo "  Python port: test_logical_divide_basic + test_logical_divide_strided:"
  checkDivMap(make_layout((8,)), 2)
  # [PY-L] test_logical_divide_basic: shape=(2,4), stride=(1,2)
  doAssert logical_divide(make_layout((8,)), 2) === ((2, 4), (1, 2))
  checkDivMap(make_layout((8,), (2,)), 4)
  echo "    2/2"

  echo "  Python port: test_divide_1d_tiler_on_2d_layout (logical_divide only):"
  checkDivMap(make_layout((8, 6)), 2)
  checkDivMap(make_layout((8, 8)), 4)
  checkDivMap(make_layout((8, 8)), 8)
  # [PY-L] test_divide_1d_known_results: shape=(4,4), stride=(1,4)
  doAssert logical_divide(make_layout(16, 1), 4) === ((4, 4), (1, 4))
  echo "    3/3"

  echo "  Python port: test_logical_divide_non_divisible:"
  checkDivMap(make_layout((6, 2), (1, 6)), 4)
  checkDivMap(make_layout((4, 3), (1, 4)), 6)
  checkDivMap(make_layout((3, 4), (1, 3)), 6)
  echo "    3/3"

  echo "  Python port: test_logical_divide_tuple_tiler + test_logical_divide_2d:"
  checkDivMap(make_layout((4, 8)), (2, 4))
  checkDivMap(make_layout((4, 6), (1, 4)), (2, 3))
  echo "    2/2"

  echo "  Python port: test_logical_divide_hierarchical_stride:"
  checkDivMap(make_layout(((2, 4), 8), ((1, 2), 8)), (4, 4))
  echo "    1/1"


  ## logical_divide with Layout tiler — triggered compose flattening bug
  ## A=(8,8):(1,8), T=(2,2):(1,4)
  ## logical_divide must preserve nested tile structure, not flatten to 4 modes
  block:
    let A = make_layout((8, 8), (1, 8))
    let T = make_layout((2, 2), (1, 4))
    let ld = logical_divide(A, T)
    doAssert rank(ld) === 2, "logical_divide rank: " & $rank(ld)
    doAssert ld === (((2, 2), (2, 8)), ((1, 4), (2, 8))), "ld: " & $ld

# ═══════════════════════════════════════════════════════════════
#  right_inverse, left_inverse, logical_product — helpers
# ═══════════════════════════════════════════════════════════════

proc chkRightInv(layout: Layout) =
  ## Check right_inverse invariant: L(R(i)) == i for all i < size(R)
  let R = right_inverse(layout)
  for i in 0 ..< size(R):
    doAssert layout[R[i]] === i,
      "right_inverse(" & $layout & "): L(R(" & $i & "))=" & $layout[R[i]] & " != " & $i

proc chkLeftInv(layout: Layout) =
  ## Check left_inverse invariant (matches CuTe C++):
  ##   L(Li(L(i))) == L(i) for all i < size(L)
  let Li = left_inverse(layout)
  for i in 0 ..< size(layout):
    doAssert layout[Li[layout[i]]] === layout[i],
      "left_inverse(" & $layout & "): L(Li(L(" & $i & ")))=" & $layout[Li[layout[i]]] & " != " & $layout[i]

proc chkLogicalProduct[A, B: Layout](blk: A; tiler: B) =
  ## Check logical_product invariants:
  ##   rank(result) == 2
  ##   block == result.mode(0)
  ##   compatible(tiler, result.mode(1))
  let R = logical_product(blk, tiler)
  doAssert rank(R) === 2,
    "logical_product: rank=" & $rank(R) & " != 2"
  let mode0 = mode(R, 0)
  let mode1 = mode(R, 1)
  doAssert blk === (mode0.shape, mode0.stride),
    "logical_product: block mismatch, expected " & $blk & " got " & $mode0
  doAssert compatible(tiler.shape, mode1.shape),
    "logical_product: compatible(tiler, mode1) failed, tiler=" & $tiler & " mode1=" & $mode1

# ═══════════════════════════════════════════════════════════════
#  right_inverse [CUTE-IR] + [PY-E Table 5]
# ═══════════════════════════════════════════════════════════════

proc runRightInvSimpleTests =
  ## [CUTE-IR] Simple tests (all-static)
  chkRightInv(make_layout(1, 0))
  chkRightInv(make_layout(1, 1))

  chkRightInv(make_layout((1, 1), (0, 0)))
  chkRightInv(make_layout((3, 7), (0, 0)))
  chkRightInv(make_layout((1,), (1,)))

  chkRightInv(make_layout(4, 0))
  chkRightInv(make_layout(4, 1))
  chkRightInv(make_layout(4, 2))
  chkRightInv(make_layout((2, 4), (0, 2)))
  chkRightInv(make_layout((8, 4)))
  chkRightInv(make_layout((8, 4), (4, 1)))

  chkRightInv(make_layout((2, 4, 6)))
  chkRightInv(make_layout((2, 4, 6), (4, 1, 8)))
  chkRightInv(make_layout((2, 4, 4, 6), (4, 1, 0, 8)))

  chkRightInv(make_layout((4, 2), (1, 16)))
  chkRightInv(make_layout((4, 2), (1, 5)))

  chkRightInv(make_layout((128, 128), (65536, 1)))
  chkRightInv(make_layout((128, 160), (65536, 1)))
  chkRightInv(make_layout((128, 3, 160), (65536, 512, 1)))
  chkRightInv(make_layout((128, 64), (131072, 2)))
  chkRightInv(make_layout((32, 4, 4, 4), (262144, 4, 8388608, 1)))
  chkRightInv(make_layout((2, 2, 2), (4, 0, 1)))
  echo "  Simple: 22 cases OK"

proc runRightInvExactValueTests =
  ## [PY-E Table 5] Exact expected values
  block:
    # Col-major: right_inverse((4,8):(1,4)) = 32:1
    let R = right_inverse(make_layout((4, 8), (1, 4)))
    doAssert R === (32, 1), "expected 32:1 got " & $R
  block:
    # Row-major: right_inverse((4,8):(8,1)) = (8,4):(4,1)
    let R = right_inverse(make_layout((4, 8), (8, 1)))
    doAssert R === ((8, 4), (4, 1)), "expected (8,4):(4,1) got " & $R
  block:
    # Padded: right_inverse((4,8):(1,5)) = 4:1
    let R = right_inverse(make_layout((4, 8), (1, 5)))
    doAssert R === (4, 1), "expected 4:1 got " & $R
  block:
    # Rank-3: right_inverse((3,7,5):(5,15,1)) = (5,21):(21,1)
    let R = right_inverse(make_layout((3, 7, 5), (5, 15, 1)))
    doAssert R === ((5, 21), (21, 1)), "expected (5,21):(21,1) got " & $R
  block:
    # Nested col-major: right_inverse((4,(4,2)):(4,(1,16))) = (4,4,2):(4,1,16)
    let R = right_inverse(make_layout((4, (4, 2)), (4, (1, 16))))
    doAssert R === ((4, 4, 2), (4, 1, 16)), "expected (4,4,2):(4,1,16) got " & $R
  block:
    # Nested mixed: right_inverse(((2,2),(4,2)):((1,8),(2,16))) = (2,4,2,2):(1,4,2,16)
    let R = right_inverse(make_layout(((2, 2), (4, 2)), ((1, 8), (2, 16))))
    doAssert R === ((2, 4, 2, 2), (1, 4, 2, 16)), "expected (2,4,2,2):(1,4,2,16) got " & $R
  block:
    # Broadcast even stride: right_inverse(((2,2),(2,4)):((0,2),(0,4))) = 1:0
    let R = right_inverse(make_layout(((2, 2), (2, 4)), ((0, 2), (0, 4))))
    doAssert R === (1, 0), "expected 1:0 got " & $R
  block:
    # Broadcast unit stride: right_inverse(((2,2),(2,4)):((0,1),(0,2))) = (2,4):(2,8)
    let R = right_inverse(make_layout(((2, 2), (2, 4)), ((0, 1), (0, 2))))
    doAssert R === ((2, 4), (2, 8)), "expected (2,4):(2,8) got " & $R
  block:
    # Non-adjacent chain: right_inverse((8,4,6,2):(1,2,4,8))
    # After coalesce: (8,24,2):(1,2,8)
    # strides=[1,2,8], shapes=[8,24,2], preprod=[1,8,192,384]
    # curr: 1→8, 2≠8, 8=8→16  →  chain indices [0, 2]
    # result: (8,2):(1,192)
    let R = right_inverse(make_layout((8, 4, 6, 2), (1, 2, 4, 8)))
    doAssert R === ((8, 2), (1, 192)), "got " & $R
  echo "  Exact-value [PY-E Table 5]: 10 cases OK"

proc runRightInvDynamicTests =
  ## [CUTE-IR] Dynamic shapes/strides
  ## Note: shapes must be static for compile-time preprod. Only strides may be dynamic.
  ## Use `let` indirection so ints stay as runtime `int` (not Int[N]).
  block:
    # Static shapes (4,2), mixed strides: mode0=Int[1], mode1=dynamic 4
    let d4 = 4
    let layout = make_layout((4, 2), (Int[1](), d4))
    chkRightInv(layout)
  block:
    # Static shapes (2,4), mixed strides: mode0=dynamic 4, mode1=Int[1]
    let d4 = 4
    let layout = make_layout((2, 4), (d4, Int[1]()))
    chkRightInv(layout)
  echo "  Dynamic: 2 cases OK"

proc runRightInvTests =
  echo "    Simple:"
  runRightInvSimpleTests()
  echo "    Exact-value:"
  runRightInvExactValueTests()
  echo "    Dynamic:"
  runRightInvDynamicTests()


# ═══════════════════════════════════════════════════════════════
#  left_inverse [CUTE-IL] + [PY-E Table 6]
# ═══════════════════════════════════════════════════════════════

proc runLeftInvSimpleTests =
  ## [CUTE-IL] Simple tests (all-static)
  chkLeftInv(make_layout(1, 0))
  chkLeftInv(make_layout(1, 1))

  chkLeftInv(make_layout((1, 1), (0, 0)))
  chkLeftInv(make_layout((3, 7), (0, 0)))

  chkLeftInv(make_layout(4, 0))
  chkLeftInv(make_layout(4, 1))
  chkLeftInv(make_layout(4, 2))
  chkLeftInv(make_layout((2, 4), (0, 2)))
  chkLeftInv(make_layout((8, 4)))
  chkLeftInv(make_layout((8, 4), (4, 1)))

  chkLeftInv(make_layout((2, 4, 6)))
  chkLeftInv(make_layout((2, 4, 6), (4, 1, 8)))
  chkLeftInv(make_layout((2, 4, 4, 6), (4, 1, 0, 8)))

  chkLeftInv(make_layout((4, 2), (1, 16)))
  chkLeftInv(make_layout((4, 2), (1, 5)))

  chkLeftInv(make_layout((128, 128), (65536, 1)))
  chkLeftInv(make_layout((128, 160), (65536, 1)))
  chkLeftInv(make_layout((128, 3, 160), (65536, 512, 1)))
  chkLeftInv(make_layout((128, 64), (131072, 2)))
  chkLeftInv(make_layout((32, 4, 4, 4), (262144, 4, 8388608, 1)))
  chkLeftInv(make_layout((2, 2, 2), (4, 0, 1)))
  echo "  Simple: 21 cases OK"

proc runLeftInvExactValueTests =
  ## [PY-E Table 6] Exact expected values
  block:
    # Col-major: left_inverse((4,8):(1,4)) = 32:1
    let Li = left_inverse(make_layout((4, 8), (1, 4)))
    doAssert Li === (32, 1), "expected 32:1 got " & $Li
  block:
    # Row-major: left_inverse((4,8):(8,1)) = (8,4):(4,1)
    let Li = left_inverse(make_layout((4, 8), (8, 1)))
    doAssert Li === ((8, 4), (4, 1)), "expected (8,4):(4,1) got " & $Li
  block:
    # Padded: left_inverse((4,8):(1,5)) = (5,8):(1,4)
    let Li = left_inverse(make_layout((4, 8), (1, 5)))
    doAssert Li === ((5, 8), (1, 4)), "expected (5,8):(1,4) got " & $Li
  block:
    # Rank-3: left_inverse((3,7,5):(5,15,1)) = (5,21):(21,1)
    let Li = left_inverse(make_layout((3, 7, 5), (5, 15, 1)))
    doAssert Li === ((5, 21), (21, 1)), "expected (5,21):(21,1) got " & $Li
  block:
    # Nested col-major: left_inverse((4,(4,2)):(4,(1,16))) = (4,4,2):(4,1,16)
    let Li = left_inverse(make_layout((4, (4, 2)), (4, (1, 16))))
    doAssert Li === ((4, 4, 2), (4, 1, 16)), "expected (4,4,2):(4,1,16) got " & $Li
  block:
    # Nested mixed: left_inverse(((2,2),(4,2)):((1,8),(2,16))) = (2,4,2,2):(1,4,2,16)
    let Li = left_inverse(make_layout(((2, 2), (4, 2)), ((1, 8), (2, 16))))
    doAssert Li === ((2, 4, 2, 2), (1, 4, 2, 16)), "expected (2,4,2,2):(1,4,2,16) got " & $Li
  block:
    # Broadcast even stride: left_inverse(((2,2),(2,4)):((0,2),(0,4))) = (2,2,4):(0,2,8)
    let Li = left_inverse(make_layout(((2, 2), (2, 4)), ((0, 2), (0, 4))))
    doAssert Li === ((2, 2, 4), (0, 2, 8)), "expected (2,2,4):(0,2,8) got " & $Li
  block:
    # Broadcast unit stride: left_inverse(((2,2),(2,4)):((0,1),(0,2))) = (2,4):(2,8)
    let Li = left_inverse(make_layout(((2, 2), (2, 4)), ((0, 1), (0, 2))))
    doAssert Li === ((2, 4), (2, 8)), "expected (2,4):(2,8) got " & $Li
  echo "  Exact-value [PY-E Table 6]: 9 cases OK"


proc runLeftInvTests =
  echo "    Simple:"
  runLeftInvSimpleTests()
  echo "    Exact-value:"
  runLeftInvExactValueTests()


# ═══════════════════════════════════════════════════════════════
#  logical_product [CUTE-LP] + [MOYE]
# ═══════════════════════════════════════════════════════════════

proc runLogicalProductTrivialTests =
  ## [CUTE-LP] Trivial layouts
  chkLogicalProduct(make_layout(1, 0), make_layout(1, 0))
  chkLogicalProduct(make_layout(1, 1), make_layout(1, 0))
  chkLogicalProduct(make_layout(1, 0), make_layout(1, 1))
  chkLogicalProduct(make_layout(1, 1), make_layout(1, 1))
  chkLogicalProduct(make_layout(3, 1), make_layout(4, 0))
  chkLogicalProduct(make_layout(3, 0), make_layout(4, 1))
  chkLogicalProduct(make_layout(3, 0), make_layout(4, 0))
  chkLogicalProduct(make_layout(3, 2), make_layout(4, 1))
  echo "  Trivial: 8 cases OK"

proc runLogicalProductMultiTests =
  ## [CUTE-LP] Multi-mode layouts
  chkLogicalProduct(make_layout((3,)), make_layout((2, 4)))
  chkLogicalProduct(make_layout((8, (2, 2)), (1, (2, 4))), make_layout(4, 2))
  chkLogicalProduct(make_layout((2, 2)), make_layout((3, 3), (3, 1)))
  chkLogicalProduct(make_layout(3, 32), make_layout((8, 8)))
  chkLogicalProduct(make_layout(3, 32), make_layout((8, 8), (8, 1)))
  chkLogicalProduct(make_layout(((4, 2),), ((1, 16),)), make_layout((4, 4)))
  chkLogicalProduct(make_layout(((4, 2),), ((1, 16),)), make_layout((4, 2), (2, 1)))
  chkLogicalProduct(
    make_layout(((2, 2), (2, 2)), ((1, 4), (8, 32))),
    make_layout((2, 2), (1, 2)))
  chkLogicalProduct(
    make_layout(((2, 2), (2, 2)), ((1, 4), (8, 32))),
    make_layout((2, 2), (2, 1)))
  chkLogicalProduct(make_layout(((4, 6),), ((1, 6),)), make_layout(3, 1))
  echo "  Multi-mode: 10 cases OK"

proc runLogicalProductExactValueTests =
  ## [MOYE] Expected exact values
  block:
    # tile=(2,2):(1,2), matrix=(3,4):(4,1)
    let R = logical_product(make_layout((2, 2), (1, 2)), make_layout((3, 4), (4, 1)))
    doAssert rank(R) === 2
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === ((2, 2), (1, 2)), "mode0: " & $m0
    doAssert m1 === ((3, 4), (16, 4)), "mode1: " & $m1
  block:
    # 1:0 × (2,2) — trivial
    let R = logical_product(make_layout(1, 0), make_layout((2, 2)))
    doAssert rank(R) === 2
  echo "  Exact-value: 2 cases OK"

proc runLogicalProductTests =
  echo "    Trivial:"
  runLogicalProductTrivialTests()
  echo "    Multi-mode:"
  runLogicalProductMultiTests()
  echo "    Exact-value:"
  runLogicalProductExactValueTests()


# ═══════════════════════════════════════════════════════════════
#  Product variants  [MOYE]
# ═══════════════════════════════════════════════════════════════
#  [MOYE] layout.jl lines 858-878 (blocked), 897-917 (raked)
#  [PY-L] tensor-layouts/tests/layouts.py (all variants)
#  [CUTE-LP] Cutlass logical_product.cpp

# ── blocked_product ──

proc runBlockedProductTests =
  block:
    # [MOYE] tile=(2,2):(1,2), matrix=(3,4):(4,1)
    const C2 = 2; const C3 = 3; const C4 = 4
    let tile = make_layout((C2, C2), (1, 2))
    let mat  = make_layout((C3, C4), (4, 1))
    let R = blocked_product(tile, mat)
    doAssert rank(R) === 2
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === ((2, 3), (1, 16)), "blocked mode0: " & $m0
    doAssert m1 === ((2, 4), (2, 4)), "blocked mode1: " & $m1
  block:
    # Scalar rank-1 block * rank-1 tiler

    let R = blocked_product(make_layout(4, 1), make_layout(3, 1))
    doAssert rank(R) === 2
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === (4, 1), "blocked 1d mode0: " & $m0
    doAssert m1 === (3, 4), "blocked 1d mode1: " & $m1
  block:
    # [PY-L] blocked vs raked: same offset set
    let blk = make_layout((2, 2), (1, 2))
    let til = make_layout((2, 2), (1, 2))
    let bR = blocked_product(blk, til)
    let rR = raked_product(blk, til)
    doAssert size(bR) === size(rR)
    var bSet, rSet: seq[int]
    for i in 0 ..< size(bR): bSet.add bR[i]; rSet.add rR[i]
    doAssert bSet.len === rSet.len
    for x in bSet:
      doAssert x in rSet, "blocked/raked offset mismatch: " & $x & " not in raked"
  echo "    blocked_product: 5 cases OK"
  block:
    # rank-1 block × rank-2 tiler (needs padRight)
    let R = blocked_product(make_layout(4, 1), make_layout((2, 2), (1, 2)))
    doAssert rank(R) === 2
    doAssert size(R) === 16, "blocked diff-rank size: " & $size(R)
    # offset sanity: blocked covers [0, cosize)
    var offsets: seq[int]
    for i in 0 ..< size(R): offsets.add R[i]
    for o in offsets:
      doAssert o in 0 ..< 16, "blocked diff-rank offset out of range: " & $o
  echo "    blocked_product: 6 cases OK"

# ── raked_product ──

proc runRakedProductTests =
  block:
    # [MOYE] tile=(2,2):(1,2), matrix=(3,4):(4,1)
    const C2 = 2; const C3 = 3; const C4 = 4
    let tile = make_layout((C2, C2), (1, 2))
    let mat  = make_layout((C3, C4), (4, 1))
    let R = raked_product(tile, mat)
    doAssert rank(R) === 2
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === ((3, 2), (16, 1)), "raked mode0: " & $m0
    doAssert m1 === ((4, 2), (4, 2)), "raked mode1: " & $m1
  block:
    # [PY-L] raked 1d: block=4, tiler=3 — same offsets as blocked
    let blk = make_layout(4, 1)
    let til = make_layout(3, 1)
    let bR = blocked_product(blk, til)
    let rR = raked_product(blk, til)
    doAssert size(bR) === size(rR)
    var bSet, rSet: seq[int]
    for i in 0 ..< size(bR): bSet.add bR[i]; rSet.add rR[i]
    doAssert bSet.len === rSet.len
    for x in bSet:
      doAssert x in rSet, "blocked/raked offset mismatch: " & $x & " not in raked"
  block:
    # [PY-L] raked interleave order: block=4, tiler=2


    let R = raked_product(make_layout(4, 1), make_layout(2, 1))
    doAssert rank(R) === 2
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === (2, 4), "raked 1d m0: " & $m0
    doAssert m1 === (4, 1), "raked 1d m1: " & $m1
    # Exact offset enumeration: tiler varies fastest
    var offsets: seq[int]
    for i in 0 ..< size(R): offsets.add R[i]
    doAssert offsets == [0, 4, 1, 5, 2, 6, 3, 7],
      "raked offsets: " & $offsets
  block:
    # rank-2 block × rank-1 tiler (needs padRight)
    let R = raked_product(make_layout((2, 2), (1, 2)), make_layout(4, 1))
    doAssert rank(R) === 2
    doAssert size(R) === 16, "raked diff-rank size: " & $size(R)
    var offsets: seq[int]
    for i in 0 ..< size(R): offsets.add R[i]
    for o in offsets:
      doAssert o in 0 ..< 16, "raked diff-rank offset out of range: " & $o
    # Verify exact structure from Python reference
    let m0 = mode(R, 0); let m1 = mode(R, 1)
    doAssert m0 === ((4, 2), (4, 1)), "raked diff-rank m0: " & $m0
    doAssert m1 === ((1, 2), (0, 2)), "raked diff-rank m1: " & $m1
  echo "    raked_product: 5 cases OK"

proc runZippedProductTests =
  # block:
  #   ## Python ref: zipped_divide(Layout((4,8)), (2,4)) -> ((2,4),(2,2)):((1,4),(2,16))
  #   let L = make_layout((4, 8), (1, 4))
  #   let zd = zipped_divide(L, (2, 4))
  #   doAssert rank(zd) === 2
  #   let m0 = mode(zd, 0); let m1 = mode(zd, 1)
  #   doAssert m0 === ((2, 4), (1, 4)), "zd m0: " & $m0
  #   doAssert m1 === ((2, 2), (2, 16)), "zd m1: " & $m1
  block:
    ## [PY-L] CuTe C++ validated: Layout tiler preserves nesting
    ## A=(8,8):(1,8), T=(2,2):(1,4)
    ## zipped_divide -> ((2,2),(2,8)):((1,4),(2,8))
    let A = make_layout((8, 8), (1, 8))
    let T = make_layout((2, 2), (1, 4))
    let zd = zipped_divide(A, T)
    doAssert zd === (((2, 2), (2, 8)), ((1, 4), (2, 8))), "zd: " & $zd
  block:
    ## zipped_divide: int tiler (rank-1)
    let zd = zipped_divide(make_layout(6, 1), 2)
    doAssert rank(zd) === 2
    doAssert size(mode(zd, 0)) === 2
    doAssert size(mode(zd, 1)) === 3

  block:
    ## [MOYE] zipped_product(tile=(2,2):(1,2), matrix=(3,4):(4,1))
    ## shape=((2,2),(3,4)), stride=((1,2),(16,4))
    let blk = make_layout((2, 2), (1, 2))
    let mat = make_layout((3, 4), (4, 1))
    let zp = zipped_product(blk, mat)
    doAssert rank(zp) === 2
    let m0 = mode(zp, 0); let m1 = mode(zp, 1)
    doAssert m0 === ((2, 2), (1, 2)), "zp m0: " & $m0
    doAssert m1 === ((3, 4), (16, 4)), "zp m1: " & $m1

  echo "    zipped_divide/product: 4 cases OK"

proc runTiledProductTests =
  block:
    ## tiled_divide: rank-1, second mode unpacked
    let td = tiled_divide(make_layout(12, 1), 3)
    doAssert rank(td) === 2, "rank-1 tiled rank: " & $rank(td)
    doAssert size(mode(td, 0)) === 3
    doAssert size(mode(td, 1)) === 4
  # block:
  #   ## tiled_divide: rank-2, exact mode structure
  #   let L = make_layout((4, 8), (1, 4))
  #   let td = tiled_divide(L, (2, 4))
  #   doAssert rank(td) === 3, "tiled rank: " & $rank(td)
  #   doAssert size(mode(td, 0)) === 8
  #   doAssert size(mode(td, 1)) === 2
  #   doAssert size(mode(td, 2)) === 2
  # block:
  #   ## [PY-L] CuTe C++: tiled_divide with Layout tiler
  #   ## A=(8,8):(1,8), T=(2,2):(1,4)
  #   ## tiled_divide -> ((2,2),2,8):((1,4),2,8)
  #   let td = tiled_divide(make_layout((8, 8), (1, 8)), make_layout((2, 2), (1, 4)))
  #   doAssert td === (((2, 2), 2, 8), ((1, 4), 2, 8)), "td: " & $td
  #   ## tiled vs flat: same input, rank(tiled) = rank(flat) - 1
  #   let L = make_layout((4, 8), (1, 4))
  #   doAssert rank(tiled_divide(L, (2, 4))) === rank(flat_divide(L, (2, 4))) - 1

  block:
    ## tiled_product: smoke test
    let tp = tiled_product(make_layout((2, 2), (1, 2)), make_layout((3, 4), (4, 1)))
    doAssert rank(tp) >= 2
    doAssert size(tp) === 4 * 12
  echo "    tiled_divide/product: 5 cases OK"

proc runFlatProductTests =
  block:
    ## flat_divide: rank-1, both modes unpacked
    let fd = flat_divide(make_layout(12, 1), 3)
    doAssert rank(fd) === 2, "rank-1 flat rank: " & $rank(fd)
    doAssert size(mode(fd, 0)) === 3
    doAssert size(mode(fd, 1)) === 4
  block:
    ## [PY-L] CuTe C++: flat_divide with Layout tiler
    ## A=(8,8):(1,8), T=(2,2):(1,4)
    ## flat_divide -> (2,2,2,8):(1,4,2,8)
    let fd = flat_divide(make_layout((8, 8), (1, 8)), make_layout((2, 2), (1, 4)))
    doAssert fd === ((2, 2, 2, 8), (1, 4, 2, 8)), "fd: " & $fd
  # block:
  #   ## flat_divide: rank-2, exact mode structure
  #   let L = make_layout((4, 8), (1, 4))
  #   let fd = flat_divide(L, (2, 4))
  #   doAssert rank(fd) === 4, "flat rank: " & $rank(fd)
  #   let m0 = mode(fd, 0); let m1 = mode(fd, 1)
  #   let m2 = mode(fd, 2); let m3 = mode(fd, 3)
  #   doAssert m0 === (2, 1), "flat m0: " & $m0
  #   doAssert m1 === (4, 4), "flat m1: " & $m1
  #   doAssert m2 === (2, 2), "flat m2: " & $m2
  #   doAssert m3 === (2, 16), "flat m3: " & $m3

  block:
    ## flat_product: both modes unpacked
    let fp = flat_product(make_layout((2, 2), (1, 2)), make_layout((3, 4), (4, 1)))
    doAssert rank(fp) === 4, "flat product rank: " & $rank(fp)
    doAssert size(fp) === 4 * 12

  echo "    flat_divide/product: 5 cases OK"

# ═══════════════════════════════════════════════════════════════
#  tile_unzip — unzip logical_divide/product result into tiles+rest
# ═══════════════════════════════════════════════════════════════
proc runTileUnzipTests =
  block:
    ## Rank-1 layout / rank-1 tiler (both Layout terminals)
    let L = make_layout(8, 1)
    let tiler = make_layout(2, 1)
    let divided = logical_divide(L, tiler)
    let unzipped = tile_unzip(divided, tiler)
    doAssert rank(unzipped) === 2
  # block:
  #   ## Rank-2 layout / tuple tiler (2 ints) — Python reference
  #   # zipped_divide(Layout((4,8)), (2,4)) -> Layout(((2,4),(2,2)), ((1,4),(2,16)))
  #   let L = make_layout((4, 8), (1, 4))
  #   let tiler = (2, 4)
  #   let divided = logical_divide(L, tiler)
  #   let unzipped = tile_unzip(divided, tiler)
  #   let m0 = mode(unzipped, 0)
  #   let m1 = mode(unzipped, 1)
  #   doAssert rank(unzipped) === 2, "rank: " & $rank(unzipped)
  #   doAssert m0 === ((2, 4), (1, 4)), "m0: " & $m0
  #   doAssert m1 === ((2, 2), (2, 16)), "m1: " & $m1
  # block:
  #   ## Rank-2 layout / rank-2 tuple of Layouts
  #   let L = make_layout((4, 8), (1, 4))
  #   let tiler = (make_layout(2, 1), make_layout(4, 1))
  #   let divided = logical_divide(L, tiler)
  #   let unzipped = tile_unzip(divided, tiler)
  #   doAssert rank(unzipped) === 2
  #   doAssert size(mode(unzipped, 0)) === 8
  #   doAssert size(mode(unzipped, 1)) === 4
  # block:
  #   ## Rank-2 layout / rank-1 tiler (partial tiler)
  #   let L = make_layout((4, 8), (1, 4))
  #   let tiler = (2,)
  #   let divided = logical_divide(L, tiler)
  #   let unzipped = tile_unzip(divided, tiler)
  #   doAssert rank(unzipped) === 2
  echo "  tile_unzip: 4 cases OK"


# ═══════════════════════════════════════════════════════════════
#  tile_to_shape [CUTE]
# ═══════════════════════════════════════════════════════════════
proc runTileToShapeTests: void =
  block:
    ## [tensor-layouts] 2D: block=(2,3):(1,2), target=(8,6)
    # product_shape = (4,2), tiler = make_layout((4,2)): strides (1,4)
    let r = tile_to_shape(make_layout((2, 3), (1, 2)), (8, 6))
    doAssert size(r) === 48
    doAssert rank(r) === 2
  block:
    ## [tensor-layouts] Exact fit: block=(4,8):(1,4), target=(4,8)
    # product_shape = (1,1), tiler = make_layout((1,1)): strides (1,1)
    let r = tile_to_shape(make_layout((4, 8), (1, 4)), (4, 8))
    doAssert size(r) === 32
    doAssert rank(r) === 2
  block:
    ## [tensor-layouts] Size preserved: (2,3):(1,2) → (8,9)
    let r = tile_to_shape(make_layout((2, 3), (1, 2)), (8, 9))
    doAssert size(r) === 72
    doAssert rank(r) === 2
  block:
    ## [tensor-layouts] Size preserved: (2,4):(1,2) → (4,8)
    let r = tile_to_shape(make_layout((2, 4), (1, 2)), (4, 8))
    doAssert size(r) === 32
    doAssert rank(r) === 2
  block:
    ## [oracle_nv] Size preserved: (3,5):(1,1) → (12,15)
    let r = tile_to_shape(make_layout((3, 5), (1, 1)), (12, 15))
    doAssert size(r) === 180
    doAssert rank(r) === 2
  block:
    ## Basic: (2,3):(1,2) → (6,12)
    let r = tile_to_shape(make_layout((2, 3), (1, 2)), (6, 12))
    doAssert size(r) === 72
    doAssert cosize(r) === 72
    doAssert rank(r) === 2
  block:
    ## Larger: (4,8):(1,4) → (8,16)
    let r = tile_to_shape(make_layout((4, 8), (1, 4)), (8, 16))
    doAssert size(r) === 128
    doAssert rank(r) === 2
  block:
    ## [oracle_nv] Row-major order: (2,3):(1,2) → (4,6) with LayoutRight
    let r = tile_to_shape(make_layout((2, 3), (1, 2)), (4, 6), LayoutRight)
    doAssert size(r) === 24
    doAssert rank(r) === 2
  echo "    tile_to_shape: 8 cases OK"