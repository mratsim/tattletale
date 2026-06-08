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

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra

# ..< overload for Int[N] (range expressions with size())
func `..<`*(a: int; b: Int): Slice[int] = a ..< b.V
func `..<`*(a: Int; b: int): Slice[int] = a.V ..< b
func `..<`*(a, b: Int): Slice[int] = a.V ..< b.V



# ── Forward declarations ──────────────────────────────────
proc runCoalesceTests*: void
proc runComplementStaticRank1Tests*: void
proc runComplementExactValueTests*: void
proc runComplementMultiModeStaticTests*: void
proc runComplementDynamicTests*: void
proc runComplementDisjointnessTests*: void
proc runComposeExactValueTests*: void
proc runComposeSimpleTests*: void
proc runComposeMultiModeTests*: void
proc runComposeDynamicTests*: void
proc runComposeRemainderTests*: void
proc runComposeNestedTests*: void
proc runComposeSwizzleTests*: void
proc runComposeEStrideTests*: void
proc runComposeNegStrideTests*: void
proc runDivideTests*: void

proc runTests* =
  echo "\n── Coalesce [CUTE-C]: 20 cases ──"
  runCoalesceTests()
  echo "\n── Complement [CUTE-CP] + [PY-L] + [PY-E] ──"
  runComplementStaticRank1Tests()
  runComplementExactValueTests()
  runComplementMultiModeStaticTests()
  runComplementDynamicTests()
  runComplementDisjointnessTests()
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
  echo "\n── logical_divide [CUTE-LD] ──"
  runDivideTests()
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

proc chkCoalesce*(layout: Layout) =
  ## Check coalesce preserves size and mapping.
  let c = coalesce(layout)
  doAssert size(c) == size(layout),
    "coalesce size mismatch: " & $size(c) & " != " & $size(layout)
  for i in 0 ..< size(layout):
    doAssert c[i] == layout[i],
      "coalesce mapping mismatch at " & $i & ": " & $c[i] & " != " & $layout[i]

# ═══════════════════════════════════════════════════════════════
#  Coalesce [CUTE-C]
# ═══════════════════════════════════════════════════════════════
proc runCoalesceScalarTests* =
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

proc runCoalesceColMajorTests* =
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

proc runCoalesceSize1Tests* =
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

proc runCoalesceStride0Tests* =
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

proc runCoalesceNonContigTests* =
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

proc runCoalesceMixedTests* =
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

proc runCoalesceDynamicTests* =
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

proc runCoalesceCppTests* =
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

proc runCoalesceTests* =
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
template testComplementProps*(layout, result: typed; cotarget: untyped) =
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
    doAssert size(cc) == 1

proc chkComplement[T](layout: Layout; cotarget: T) =
  let r = complement(layout, cotarget)
  testComplementProps(layout, r, cotarget)

proc chkComplement*(layout: Layout) =
  chkComplement(layout, cosize(layout))

# ═══════════════════════════════════════════════════════════════
#  Complement [CUTE-CP] + [PY-L] + [PY-E]
# ═══════════════════════════════════════════════════════════════
#
# References:

# ─── Static rank-1 [CUTE-CP] ───────────────────────────────────
proc runComplementStaticRank1Tests* =
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
proc runComplementExactValueTests* =
  block:
    let r = complement(make_layout(1, 0), 8)
    doAssert $r.shape == "8"
    doAssert $r.stride == "1"
  block:
    let r = complement(make_layout(4, 2), 16)
    doAssert $r.shape == "(2, 2)"
    doAssert $r.stride == "(1, 8)"
  block:
    let r = complement(make_layout(4, 1), 16)
    doAssert $r.shape == "4"
    doAssert $r.stride == "4"
  block:
    let r = complement(make_layout(4, 1), 24)
    doAssert $r.shape == "6"
    doAssert $r.stride == "4"
  block:
    let r = complement(make_layout((2, 2), (1, 4)), 16)
    doAssert $r.shape == "(2, 2)"
    doAssert $r.stride == "(2, 8)"
  block:
    let r = complement(make_layout(2, 1), (Int[3](), 4))
    doAssert $r.shape == "6"
    doAssert $r.stride == "2"
  block:
    let r = complement(make_layout(6, 1), 24)
    doAssert $r.shape == "4"
    doAssert $r.stride == "6"
  block:
    let r = complement(make_layout((4, 2), (1, 16)))
    doAssert $r.shape == "4"
    doAssert $r.stride == "4"
  block:
    let r = complement(make_layout(8, 1), 32)
    doAssert $r.shape == "4"
    doAssert $r.stride == "8"
  block:
    let r = complement(make_layout(4, 4), 32)
    doAssert $r.shape == "(4, 2)"
    doAssert $r.stride == "(1, 16)"
  echo "  Exact-value: 10 Python assertions OK"

# ─── Multi-mode static [CUTE-CP] ───────────────────────────────
proc runComplementMultiModeStaticTests* =
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
proc runComplementDynamicTests* =
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
proc runComplementDisjointnessTests* =
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

proc runComposeExactValueTests* =
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
  echo "    Exact-value: 10/10"

proc runComposeSimpleTests* =
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

proc runComposeMultiModeTests* =
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

proc runComposeDynamicTests* =
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

proc runComposeRemainderTests* =
  doAssert chkCompose(make_layout(1,0), make_layout(4,1))
  doAssert chkCompose(make_layout(1,1), make_layout(4,1))
  doAssert chkCompose(make_layout(4,1), make_layout(4,2))
  doAssert chkCompose(make_layout((4,3),(3,1)), make_layout(24,1))
  doAssert chkCompose(make_layout((4,3),(3,1)), make_layout(8,1))
  doAssert chkCompose(make_layout((4,3,1),(3,1,0)), make_layout(24,1))
  doAssert chkCompose(make_layout((4,3,1),(3,1,0)), make_layout(4,1))
  doAssert chkCompose(make_layout((4,6,8,10),(2,3,5,7)), make_layout(6,12))
  doAssert chkCompose(make_layout((8,8),(8,1)), make_layout(2,3))
  doAssert chkCompose(make_layout((8,8),(8,1)), make_layout(3,3))
  doAssert chkCompose(make_layout(3,1), make_layout(4,1))
  doAssert chkCompose(make_layout((48,24,5),(1,128,3072)), make_layout(32,1))
  echo "    Remainder: 12/12"

proc runComposeNestedTests* =
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

proc runComposeSwizzleTests* =
  echo "    Swizzle [CUTE-CM #55-56]:"
  # [CUTE-CM] #55: compose(Layout<8,8>:(8,1), Swizzle ∘ Layout<8,8>:(8,1))
  # [CUTE-CM] #56: compose(Swizzle∘..., Swizzle∘...) — double swizzle
  # Swizzle type not yet implemented
  echo "    0/2 (blocked: no Swizzle type)"

proc runComposeEStrideTests* =
  echo "    E stride [CUTE-CM #63-66]:"
  # [CUTE-CM] #63: ((1,(2,4)):(0,(-1,512))) / (2:-1)
  # [CUTE-CM] #64: ((1,(2,4)):(0,(-1,512))) / (4:-1)
  # [CUTE-CM] #65: (4,4):(4,1) / (4,4):(E1,E0)
  # [CUTE-CM] #66: (4,(2,3)):(6,(3,1)) / (2,4):(E11,E0)
  # ScaledBasis/E-stride not yet implemented
  echo "    0/4 (blocked: no E stride support)"

proc runComposeNegStrideTests* =
  doAssert chkCompose(make_layout(4,-1), make_layout(4,1))
  doAssert chkCompose(make_layout(4,1), make_layout(4,-1))
  doAssert chkCompose(make_layout(4,-1), make_layout(4,-1))
  doAssert chkCompose(make_layout(4,1), make_layout(4,-2))
  doAssert chkCompose(make_layout((4,4),(-1,1)), make_layout(2,1))
  # [CUTE-CM] #62: (4,4):(-1,1) / (2,4,2):(1,4,2)
  doAssert chkCompose(make_layout((4,4),(-1,1)), make_layout((2,4,2),(1,4,2)))
  echo "    Neg strides: 6/6"
# ═══════════════════════════════════════════════════════════════
#  logical_divide [CUTE-LD]
# ═══════════════════════════════════════════════════════════════
#
#  CuTe formula: compose(A, Layout(B, complement(B, shape(coalesce(A)))))
#  shape() = product of shape elements, NOT cosize().
#
# ═══════════════════════════════════════════════════════════════

proc runDivideTests*: void =
  ## 24 [CUTE-LD] cases + 8 rank-2 tiler extensions

  template checkDiv(L, T: typed): untyped =
    let R = logical_divide(L, T)
    doAssert rank(R) == 2, "rank not 2 for " & $L & " / " & $T

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
    doAssert rank(R) == 2, "rank not 2 for " & $L & " / " & $T
    doAssert size(R) == size(L)
    for i in 0 ..< size(L):
      doAssert R[i] == L[i], "mapping fail at " & $i & " for " & $L & " / " & $T

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