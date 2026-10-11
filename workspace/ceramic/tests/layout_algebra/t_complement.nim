# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Run: nim c -r workspace/ceramic/tests/layout_algebra/t_complement.nim
#
# Complement tests: integer post-conditions, exact values,
# coordinate-stride complements. Vectors cross-checked against pycute.
# Sources: t_layout_algebra.nim, t_layout_algebra_anti_regressions.nim.

{.experimental: "callOperator".}

import std/algorithm
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/ism_coord_strides

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

# ═══════════════════════════════════════════════════════════════
#  Complement post-condition checks (CuTe test_complement)
# ═══════════════════════════════════════════════════════════════
#
#
# Checks:
#   (1) cosize(layout ++ complement) >= size(cotarget)
#   (2) cosize(complement) <= round_up(size(cotarget), cosize(layout))
#   (3) result[i-1] < result[i]  ordered
#   (4) result[i] != layout[j]   disjoint (tested separately)
#   (5) size(result) <= cosize(result)
#   (9) if static stride, complement(completed) has size 1
template testComplementProps(layout, result: typed, cotarget: untyped) =
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
    doAssert result(i-1) < result(i)
  # (5) size <= cosize
  doAssert rSize <= cosize(result)
  # (9) If static stride, complement(completed) has size 1
  when typeof(completed.stride) is Int or typeof(completed.stride) is int:
    let cc = complement(completed)
    doAssert size(cc) === 1

proc chkComplement[T](layout: Layout, cotarget: T) =
  let r = complement(layout, cotarget)
  testComplementProps(layout, r, cotarget)

proc chkComplement(layout: Layout) =
  chkComplement(layout, cosize(layout))

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

  # Full coverage, the complement is the lone (1):(coverage) dimension
  block:
    let r = complement(make_layout(2, 1), 2)
    doAssert r.shape === 1
    doAssert r.stride === 2

  # Strides covering the range without gaps
  block:
    let r = complement(make_layout((2, 2), (1, 4)), 8)
    doAssert r.shape === 2
    doAssert r.stride === 2

  # Broadcast layout, a scalar stride-0 over a nested shape collapses to (bound):(1)
  block:
    let r = complement(make_layout((8, 4), 0), 32)
    doAssert r.shape === 32
    doAssert r.stride === 1
    let rDefault = complement(make_layout((8, 4), 0))
    doAssert rDefault.shape === 1
    doAssert rDefault.stride === 1
  echo "  Exact-value: 14 Python assertions OK"

# ─── Symbol-routed static bound ──────────────────────────────────────
proc runComplementSymbolBoundTests =
  # The Int or int overload receives the bound as a symbol, the static
  # branch must read V from the type
  block:
    let r = complement(make_layout(4, 2), Int[8]())
    doAssert r.shape === 2
    doAssert r.stride === 1
  echo "  Symbol-routed static bound: 1 pin OK"

# ─── Multi-dimension static [CUTE-CP] ───────────────────────────────
proc runComplementMultiDimensionStaticTests =
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
  echo "  Multi-dimension static: 12 complement calls OK"

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
        if layout(i) !== 0 and c(j) !== 0:
          doAssert layout(i) !== c(j),
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
#  Anti-regression fixtures (from t_layout_algebra_anti_regressions)
# ═══════════════════════════════════════════════════════════════

proc runComplementFixtureTests =
  # Guarded cases: complement with a compile-time bound must produce
  # the coalesced result: the rem dimension (ceil_div(460, 512) = 1) is
  # statically 1, so coalesce's trailing size-1 discard removes it.
  block:
    let r = complement(make_layout((2, 2), (1, 4)), 16)
    check r.shape, (2, 2), (Int[2], Int[2])
    check r.stride, (2, 8), (Int[2], Int[8])
  block:
    let r = complement(make_layout((2, 4, 8), (8, 1, 64)), 460)
    check r.shape, (2, 4), (Int[2], Int[4])
    check r.stride, (4, 16), (Int[4], Int[16])
  block:
    let r = complement(make_layout((2, (3, 4)), (3, (1, 6))))
    check r.shape, 1, Int[1]
    check r.stride, 24, Int[24]

  echo "    complement fixture: 3 guarded cases OK"

proc runComplementDynamicShapeTests =
  # Guarded cases: a runtime shape and the identical constant-spelled
  # layout must give the same result.
  block:
    let n = 2
    let r = complement(make_layout((2, n, 4), (1, 3, 12)), 200)
    check r.shape, (2, 5), (int, Int[5])
    check r.stride, (6, 48), (int, Int[48])
    let rs = complement(make_layout((2, 2, 4), (1, 3, 12)), 200)
    doAssert r.shape === rs.shape
    doAssert r.stride === rs.stride
  block:
    let m = 4
    let r = complement(make_layout((m, 4, 8), (8, 1, 64)), 460)
    check r.shape, (2, 2), (Int[2], int)
    check r.stride, (4, 32), (Int[4], int)
    let rs = complement(make_layout((4, 4, 8), (8, 1, 64)), 460)
    doAssert r.shape === rs.shape
    doAssert r.stride === rs.stride

  echo "    complement dynamic-shape fixture: 2 guarded cases OK"

proc runComplementInlineSkipTests =
  ## Complement's own walk skips stride-0 and size-1 dimensions.
  ## A stride-0 dimension maps every coordinate to offset 0.
  ## A size-1 dimension covers a single offset.
  ## Neither dimension joins the gap-fill chain.
  block:
    ## dynamic shape, static stride-0 dimension: (n, m):(1, 0) behaves
    ## as (n):(1), the complement is (ceil_div(64, n)):(n).
    ## The stride-0 dimension's shape never enters the arithmetic
    let n = 6
    let m = 4
    let r = complement(make_layout((n, m), (1, 0)), 64)
    let rs = complement(make_layout((6, 4), (1, 0)), 64)
    check rs.shape, 11, Int[11]
    check rs.stride, 6, Int[6]
    doAssert r.shape === rs.shape
    doAssert r.stride === rs.stride
  block:
    ## scalar stride-0 broadcast over a dynamic shape profile: every
    ## coordinate maps to offset 0, the complement is the full bound,
    ## and the default bound (cosize = 1) collapses it to the (1):(1) mark
    let n = 6
    let m = 4
    let r = complement(make_layout((n, m), 0), 32)
    doAssert r.shape === 32
    doAssert r.stride === 1
    let rd = complement(make_layout((n, m), 0))
    doAssert rd.shape === 1
    doAssert rd.stride === 1
  block:
    ## static size-1 dimension under a dynamic shape: (1, n):(8, 3)
    ## behaves as (n):(3), the complement is (3, ceil_div(32, 3n)):(1, 3n)
    let n = 6
    let r = complement(make_layout((1, n), (8, 3)), 32)
    let rs = complement(make_layout((1, 6), (8, 3)), 32)
    check rs.shape, (3, 2), (Int[3], Int[2])
    check rs.stride, (1, 18), (Int[1], Int[18])
    doAssert r.shape === rs.shape
    doAssert r.stride === rs.stride
  echo "    PASS"

proc runComplementFullCoverageTests =
  block:
    let r = complement(make_layout(4, 1), 4)
    check r.shape, 1, Int[1]
    check r.stride, 4, Int[4]
  block:
    let r = complement(make_layout((4, 2)), 8)
    check r.shape, 1, Int[1]
    check r.stride, 8, Int[8]
  block:
    ## Remainder of (2,2):(1,4) over 8, the free offsets {2,3,6,7}
    let r = complement(make_layout((2, 2), (1, 4)), 8)
    check r.shape, 2, Int[2]
    check r.stride, 2, Int[2]

  echo "    complement full-coverage fixture: 3 guarded cases OK"

# ═══════════════════════════════════════════════════════════════
#  Coordinate strides [pycute]
# ═══════════════════════════════════════════════════════════════
#
#  complement over coordinate-strided layouts, the result's rest
#  strides are E-atoms (pycute: complement(tv) == (1,1):(32@0,16@1)).
#  The walk: per codomain axis, gapShape = scale // covered,
#  covered = scale * extent, scales in colex order.

proc runComplementCoordStrideTests =
  let tv = make_layout((((2, 2, 2, 2, 2), 2), (4, 2)),
                       (((ism_coord_strides.E(2, 1), ism_coord_strides.E(1, 0),
                          ism_coord_strides.E(2, 0), ism_coord_strides.E(4, 1),
                          ism_coord_strides.E(4, 0)), ism_coord_strides.E(1, 1)),
                         (ism_coord_strides.E(8, 0), ism_coord_strides.E(8, 1))))
  block:
    ## the tvcrd complement, the tile-corner lift as rest strides
    let c = complement(tv)
    check c.shape, (1, 1), (Int[1], Int[1])
    check c.stride[0], ism_coord_strides.E(32, 0), ism_coord_strides.CoordStride[(32,)]
    check c.stride[1], ism_coord_strides.E(16, 1), ism_coord_strides.CoordStride[(0, 16)]
  block:
    ## extended to the (96, 192) world, per-axis ceil growth
    let c = complement(tv, (96, 192))
    check c.shape, (3, 12), (Int[3], Int[12])
    check c.stride[0], ism_coord_strides.E(32, 0), ism_coord_strides.CoordStride[(32,)]
    check c.stride[1], ism_coord_strides.E(16, 1), ism_coord_strides.CoordStride[(0, 16)]
  block:
    ## zipped_divide of the integer world: the rest modes are integers
    let w = make_layout((96, 192), (192, 1))
    let zd = zipped_divide(w, tv)
    doAssert zd === make_layout(
      ((((2, 2, 2, 2, 2), 2), (4, 2)), (3, 12)),
      ((((2, 192, 384, 4, 768), 1), (1536, 8)), (6144, 16))), "zd: " & $zd
  block:
    ## coordinate divide: identity world map over the tvcrd layout,
    ## the corner lift lands in the rest strides
    let cworld = make_layout((96, 192), (ism_coord_strides.E(0), ism_coord_strides.E(1)))
    let rc = zipped_divide(cworld, tv)
    check rc.stride[1][0], ism_coord_strides.E(32, 0), ism_coord_strides.CoordStride[(32,)]
    check rc.stride[1][1], ism_coord_strides.E(16, 1), ism_coord_strides.CoordStride[(0, 16)]
  block:
    ## dynamic cotarget: a runtime world mode extends its terminal
    ## shape by runtime ceil_div, the static mode folds, the paired
    ## static run over (96, 192) gives rest shape (3, 12)
    let M = 96
    let cworldDyn = make_layout((M, 192), (ism_coord_strides.E(0), ism_coord_strides.E(1)))
    let rcDyn = zipped_divide(cworldDyn, tv)
    check rcDyn.stride[1][0], ism_coord_strides.E(32, 0), ism_coord_strides.CoordStride[(32,)]
    check rcDyn.stride[1][1], ism_coord_strides.E(16, 1), ism_coord_strides.CoordStride[(0, 16)]
    check rcDyn.shape[1][1], 12, Int[12]
    check rcDyn.shape[1][0], ceil_div(M, 32), int
  echo "  Coordinate strides: 4 cases OK"

# ═══════════════════════════════════════════════════════════════

proc runTests =
  echo "\n── Complement [CUTE-CP] + [PY-L] + [PY-E] ──"
  runComplementStaticRank1Tests()
  runComplementExactValueTests()
  runComplementSymbolBoundTests()
  runComplementMultiDimensionStaticTests()
  runComplementDynamicTests()
  runComplementDisjointnessTests()
  echo "\n── Anti-regression fixtures ──"
  runComplementFixtureTests()
  runComplementDynamicShapeTests()
  runComplementInlineSkipTests()
  runComplementFullCoverageTests()
  echo "\n── Coordinate strides [pycute] ──"
  runComplementCoordStrideTests()
  echo "\nALL TESTS PASSED"

when isMainModule:
  runTests()
