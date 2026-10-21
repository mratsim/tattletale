# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for the layout divide family: logical_divide, tiled_divide, flat_divide, zipped_divide.
##
## Test that the divide family splits layouts into (tile, rest), preserving the element mapping and the zipped regrouping.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip workspace/ceramic/tests/layout_algebra/t_divide.nim

{.experimental: "callOperator".}

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra

template check(got: untyped, expected: typed, expectedType: typedesc): untyped =
  block:
    let tmp = got
    type TmpType = typeof(tmp)
    when TmpType is expectedType:
      doAssert tmp === expected
    else:
      {.error: "[ttt] Please check constant-folding: type is " &
          $TmpType & ", expected " & $expectedType.}

# ── logical_divide [CUTE-LD] ─────────────────────────────────
proc runLogicalDivideTests =
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

  echo "  'Dangerous' stride-0 LHS dimensions:"
  checkDiv(make_layout((128,4,3), (1,512,0)), make_layout(32, 1))
  checkDiv(make_layout((128,4,3), (1,512,0)), make_layout(32, 2))
  checkDiv(make_layout((16,4,3), (1,512,0)), make_layout(32, 1))
  echo "    3/3"

  template checkDivMap(L, T: typed): untyped =
    let R = logical_divide(L, T)
    doAssert rank(R) === 2, "rank not 2 for " & $L & " / " & $T
    doAssert size(R) === size(L)
    for i in 0 ..< size(L):
      doAssert R(i) === L(i), "mapping fail at " & $i & " for " & $L & " / " & $T

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

  echo "  Rank-1 direct divide (value-identical to the general formula):"
  doAssert logical_divide(make_layout(10, 2), 4) === ((4, 3), (2, 8))
  doAssert logical_divide(make_layout(10, 0), 4) === ((4, 3), (0, 0))
  doAssert logical_divide(make_layout(10, 2), Int[4]()) === ((4, 3), (2, 8))
  let dynS = 16
  let dynT = 4
  doAssert logical_divide(make_layout(dynS, 1), 4) === ((4, 4), (1, 4))
  doAssert logical_divide(make_layout(16, 1), dynT) === ((4, 4), (1, 4))
  doAssert logical_divide(make_layout(dynS, 1), dynT) === ((4, 4), (1, 4))
  echo "    6/6"

  echo "  Direct divide through the per-dimension tuple-tiler path:"
  doAssert logical_divide(make_layout((10, 8), (2, 1)), (4, 4)) ===
    (((4, 3), (4, 2)), ((2, 8), (1, 4)))
  # a (1, 1) tiler divides each dimension by 1, the per-dimension
  # result is (1, shape):(stride, stride), which locks the degenerate
  # (1, 1) syntax of the per-dimension divide path
  doAssert logical_divide(make_layout((2, 4, 8), (1, 2, 3)), (1, 1)) ===
    (((1, 2), (1, 4), 8), ((1, 1), (2, 2), 3))
  echo "    2/2"

  echo "  Python port: test_logical_divide_tuple_tiler + test_logical_divide_2d:"
  checkDivMap(make_layout((4, 8)), (2, 4))
  checkDivMap(make_layout((4, 6), (1, 4)), (2, 3))
  echo "    2/2"

  echo "  Python port: test_logical_divide_hierarchical_stride:"
  checkDivMap(make_layout(((2, 4), 8), ((1, 2), 8)), (4, 4))
  echo "    1/1"

  # contract-doc Examples on the Layout-tiler overload, locked:
  # logical_divide(make_layout(16, 3), 4) -> (4, 4):(3, 12)
  doAssert logical_divide(make_layout(16, 3), 4) === ((4, 4), (3, 12))
  # CuTe C++ 1-D worked example:
  # A=(4,2,3):(2,1,8), B=4:2 -> ((2,2),(2,3)):((4,1),(2,8))
  let ldExample = logical_divide(make_layout((4, 2, 3), (2, 1, 8)), make_layout(4, 2))
  doAssert ldExample === (((2, 2), (2, 3)), ((4, 1), (2, 8)))

  ## logical_divide with a Layout tiler, A=(8,8):(1,8), T=(2,2):(1,4):
  ## the result keeps the tile modes nested, (tile modes, rest)
  block:
    let A = make_layout((8, 8), (1, 8))
    let T = make_layout((2, 2), (1, 4))
    let ld = logical_divide(A, T)
    doAssert rank(ld) === 2, "logical_divide rank: " & $rank(ld)
    doAssert ld === (((2, 2), (2, 8)), ((1, 4), (2, 8))), "ld: " & $ld

# ═══════════════════════════════════════════════════════════════
#  Section 21. tiled_divide / flat_divide
# ═══════════════════════════════════════════════════════════════
#
#  Both tile a layout into the (tile, rest) pair:
#  - tiled_divide keeps dimension 0 as the grouped tile, dimension 1
#    unpacked one level
#  - flat_divide unpacks both dimensions
#  - no runtime binding at the call site, the constructor's base
#    tuples pass through

proc runDivideReassemblyTests =
  block:
    ## tuple tiler, exact dimension structure, deep path
    let L = make_layout((4, 8), (1, 4))
    check(tiled_divide(L, (2, 4)), (((2, 4), 2, 2), ((1, 4), 2, 16)), Layout)
    check(flat_divide(L, (2, 4)), ((2, 4, 2, 2), (1, 4, 2, 16)), Layout)
  block:
    ## rank-1 layout, scalar tiler, whole-layout fast path
    check(tiled_divide(make_layout(12, 1), 3), ((3, 4), (1, 3)), Layout)
    check(flat_divide(make_layout(12, 1), 3), ((3, 4), (1, 3)), Layout)
  block:
    ## [PY-L] CuTe C++, A=(8,8):(1,8), T=(2,2):(1,4) as a Layout tiler
    let T = make_layout((2, 2), (1, 4))
    check(tiled_divide(make_layout((8, 8), (1, 8)), T), (((2, 2), 2, 8), ((1, 4), 2, 8)), Layout)
    check(flat_divide(make_layout((8, 8), (1, 8)), T), ((2, 2, 2, 8), (1, 4, 2, 8)), Layout)
  block:
    ## Meta tensor-layouts doc, tiled_divide(Layout((8,8)), (2,2))
    check(tiled_divide(make_layout((8, 8), (1, 8)), (2, 2)), (((2, 2), 4, 4), ((1, 8), 2, 16)), Layout)
  block:
    ## empty tiler: nothing divides, an empty tile part
    ## precedes the dimensions that pass through whole:
    ## - identical to the zipped gather result
    let e = tiled_divide(make_layout((10, 8), (2, 1)), ())
    doAssert e === make_layout(((), 10, 8), ((), 2, 1)), "empty tiler: " & $e
    doAssert typeof(e.shape[1]) is Int[10], "empty tiler type: " & $typeof(e.shape[1])
    doAssert typeof(e.stride[2]) is Int[1], "empty tiler type: " & $typeof(e.stride[2])
  block:
    ## runtime lvalue layout and runtime tiler symbol match the literal forms
    let L = make_layout((4, 8), (1, 4))
    let T = (2, 4)
    check(tiled_divide(L, T), (((2, 4), 2, 2), ((1, 4), 2, 16)), Layout)
    check(flat_divide(L, T), ((2, 4, 2, 2), (1, 4, 2, 16)), Layout)

  echo "    divide macros: 6 guarded cases OK"

# ── divide tuple-tiler ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
#
#  One tiler element per layout dimension, the per-dimension divide
#  is emitted twice, once per shape and once per stride projection,
#  both copies pure expressions, no evalOnceAs and no lets introduced
#  by the macro itself. The elements come off a tuple dims stream,
#  the layout dimensions off a shape/stride stream zip, symbols
#  and calls work exactly like tuple constructors.

proc runDivideTupleTilerTests =
  block:
    ## static tiler, the fully folded emission
    let r = logical_divide(make_layout((10, 8), (2, 1)), (4, 4))
    check r.shape, ((4, 3), (4, 2)), ((Int[4], Int[3]), (Int[4], Int[2]))
    check r.stride, ((2, 8), (1, 4)), ((Int[2], Int[8]), (Int[1], Int[4]))
  block:
    ## a tiler carried by a symbol: the elements are bracket reads,
    ## the runtime elements stay runtime, the static parts still fold,
    ## and the result equals the fully static tiler result
    let t = (4, 4)
    let r = logical_divide(make_layout((10, 8), (2, 1)), t)
    let rs = logical_divide(make_layout((10, 8), (2, 1)), (4, 4))
    doAssert r === rs
  block:
    ## an empty tiler divides nothing, every dimension passes through,
    ## the result is the layout itself
    let r = logical_divide(make_layout((10, 8), (2, 1)), ())
    check r.shape, (10, 8), (Int[10], Int[8])
    check r.stride, (2, 1), (Int[2], Int[1])
    doAssert r === make_layout((10, 8), (2, 1))
  block:
    ## a runtime layout through the same path, the runtime leaves stay
    ## runtime and the static folds still fold
    let dM = 4
    let dN = 5
    let r = logical_divide(make_layout((dM, dN), (1, 16)), (4, 4))
    check r.shape, ((4, 1), (4, 2)), ((Int[4], int), (Int[4], int))
    check r.stride, ((1, 4), (16, 64)), ((Int[1], Int[4]), (Int[16], Int[64]))
  block:
    ## a tiler longer than the layout is a compile-time error
    static:
      # TODO: using compiles/not(compiles) violates repo policy where it is used
      doAssert not compiles(logical_divide(make_layout((2, 4), (1, 2)), (4, 4, 2)))
  echo "    divide tuple-tiler rewrite: 5 cases OK"

# ── zipped_divide ────────────────────────────────────────────
proc runZippedDivideRank1Tests =
  block:
    ## Rank-1 layout / rank-1 tiler (both Layout terminals)
    let L = make_layout(8, 1)
    let tiler = make_layout(2, 1)
    let zipped = zipped_divide(L, tiler)
    doAssert rank(zipped) === 2
  echo "  zipped_divide rank-1: 1 case OK"

proc runZippedDivideStructureTests =
  block:  # zipped_divide layout structure
    doAssert zipped_divide(make_layout((6, 6), (1, 6)), (2, 2)) ===
      (((2, 2), (3, 3)), ((1, 6), (2, 12)))

  block:  # zipped_divide non-square tiler
    doAssert zipped_divide(make_layout((12, 6), (1, 12)), (3, 2)) ===
      (((3, 2), (4, 3)), ((1, 12), (3, 24)))

  echo "  zipped_divide structure: 2 cases OK"

# ═══════════════════════════════════════════════════════════════
#  Section 15. hier_unzip zipped gather over the stream walk
# ═══════════════════════════════════════════════════════════════
#
#  One tiler element per layout dimension, the splitter call per terminal
#  element binds once, the four shape/stride projections read the binding.
#  A single-element gather collapses to a scalar,
#  so a 1-element tiler and a 1-element sub-tiler carry no tuple wrapper
#  and a leftover dimension splices verbatim into the rest part.

proc runZippedDivideGatherTests =
  block:
    ## int vs 1-element tuple tiler agree on a rank-1 layout
    let L = make_layout(6, 1)
    doAssert zipped_divide(L, 2) === zipped_divide(L, (2,)),
      "scalar tiler guard: " & $zipped_divide(L, (2,))
  block:
    ## 1-element tiler: the tile part collapses to a scalar, the leftover
    ## dimension splices verbatim into the rest part
    let zd = zipped_divide(make_layout((4, 8), (1, 4)), (2,))
    doAssert zd === ((2, (2, 8)), (1, (2, 4))), "1-element tiler: " & $zd
  block:
    ## empty tiler: nothing divides, every dimension gathers into the rest
    ## part behind an empty tile part
    let z = zipped_divide(make_layout((10, 8), (2, 1)), ())
    doAssert z === make_layout(((), (10, 8)), ((), (2, 1))), "empty tiler: " & $z
  block:
    ## sub-tuple tiler element recurses, the gathered pair contributes
    ## one element per part, one result level per tiler level
    let r = zipped_divide(make_layout((8, (4, 2)), (16, (1, 2))), (2, (4, 2)))
    doAssert r === (((2, (4, 2)), (4, (1, 1))), ((16, (1, 2)), (32, (4, 4)))),
      "sub-tuple tiler: " & $r
  echo "    zipped gather rewrite: 4 cases OK"


# ── int tiler over layouts the tiler does not evenly divide ──
proc runIntTilerRaggedTests =
  ## Divide results for tilers whose size (128) does not divide
  ## the layout size.

  block:  # 3000 elements, 24 whole tiles of 128, 72 slot overhang
    doAssert logical_divide(make_layout((10, 300), (1, 10)), make_layout((32, 4), (4, 1))) ===
      (((32, 4), 24), ((4, 1), 128))

  block:  # 4096 elements, exactly 32 tiles of 128
    doAssert logical_divide(make_layout((16, 256), (256, 1)), make_layout((32, 4), (4, 1))) ===
      ((((4, 8), 4), 32), (((1024, 1), 256), 8))

  block:  # the grid mode of the 3000-element divide: ceil(3000/128) = 24
    doAssert complement(make_layout((32, 4), (4, 1)), (10, 300)) === (24, 128)

  block:  # the tiler lane stride steps 4 coordinates of mode 0,
    ## mode 0 has dims 10: 4 does not divide 10, the divide is rejected
    static:
      # TODO: using compiles/not(compiles) violates repo policy where it is used
      doAssert not compiles(
        logical_divide(make_layout((10, 300), (300, 1)), make_layout((32, 4), (4, 1))))

  # A runtime layout cannot generally be divided like the same layout written with compile-time constants:
  #
  # - the divide result shape is fixed when the program compiles
  # - whether the layout collapses to a flat list of elements depends on stride values, known at run time
  # - one compiled result shape cannot be correct for every stride value
  #
  # block:  # Test whther a runtime-leaf layout divides exactly like the same
  #         # layout written with compile-time constants
  #   let dM = 10
  #   let dK = 300
  #   let dS = 10
  #   let r = logical_divide(make_layout((dM, dK), (1, dS)), make_layout((32, 4), (4, 1)))
  #   doAssert r === logical_divide(make_layout((10, 300), (1, 10)), make_layout((32, 4), (4, 1))),
  #     "runtime divide: " & $r
  echo "  int-tiler ragged: 4 cases OK"

runLogicalDivideTests()
runDivideReassemblyTests()
runDivideTupleTilerTests()
runZippedDivideRank1Tests()
runZippedDivideStructureTests()
runZippedDivideGatherTests()
runIntTilerRaggedTests()

