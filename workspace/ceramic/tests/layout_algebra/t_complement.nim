# Run: nim c -r --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip workspace/ceramic/tests/layout_algebra/t_complement.nim

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra

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

# ── rank-1, static shapes and strides ─────────────────────────
proc runRank1StaticTests =
  # Layout(_1,_0), the layout covers offset 0 only
  block:
    let r = complement(make_layout(1, 0))
    check r.shape, 1, Int[1]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(1, 0), 2)
    check r.shape, 2, Int[2]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(1, 0), 5)
    check r.shape, 5, Int[5]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(1, 0), 8)
    check r.shape, 8, Int[8]
    check r.stride, 1, Int[1]
  block:
    # a tuple bound enters as its product, 2*2
    let r = complement(make_layout(1, 0), (2, 2))
    check r.shape, 4, Int[4]
    check r.stride, 1, Int[1]
  # Layout(_1,_1)
  block:
    let r = complement(make_layout(1, 1))
    check r.shape, 1, Int[1]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(1, 1), 5)
    check r.shape, 5, Int[5]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(1, 1), (2, 2))
    check r.shape, 4, Int[4]
    check r.stride, 1, Int[1]
  # Layout(_1,_2), stride 2 covers offset 0 only
  block:
    let r = complement(make_layout(1, 2), 1)
    check r.shape, 1, Int[1]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(1, 2), 2)
    check r.shape, 2, Int[2]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(1, 2), 8)
    check r.shape, 8, Int[8]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(1, 2), (2, 2))
    check r.shape, 4, Int[4]
    check r.stride, 1, Int[1]
  # Layout(_4,_0), a stride-0 layout covers offset 0 only
  block:
    let r = complement(make_layout(4, 0), 1)
    check r.shape, 1, Int[1]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(4, 0), 2)
    check r.shape, 2, Int[2]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(4, 0), 8)
    check r.shape, 8, Int[8]
    check r.stride, 1, Int[1]
  # Layout(_4,_1), full coverage to its cosize
  block:
    let r = complement(make_layout(4, 1))
    check r.shape, 1, Int[1]
    check r.stride, 4, Int[4]
  block:
    let r = complement(make_layout(4, 1), 16)
    check r.shape, 4, Int[4]
    check r.stride, 4, Int[4]
  block:
    let r = complement(make_layout(4, 1), 24)
    check r.shape, 6, Int[6]
    check r.stride, 4, Int[4]
  # Layout(_4,_2), the even offsets, the odd offsets complement
  block:
    let r = complement(make_layout(4, 2))
    check r.shape, (2, 1), (Int[2], Int[1])
    check r.stride, (1, 8), (Int[1], Int[8])
  block:
    let r = complement(make_layout(4, 2), 1)
    check r.shape, 2, Int[2]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(4, 2), 16)
    check r.shape, (2, 2), (Int[2], Int[2])
    check r.stride, (1, 8), (Int[1], Int[8])
  block:
    let r = complement(make_layout(4, 2), 19)
    check r.shape, (2, 3), (Int[2], Int[3])
    check r.stride, (1, 8), (Int[1], Int[8])
  block:
    let r = complement(make_layout(4, 2), (2, 2))
    check r.shape, 2, Int[2]
    check r.stride, 1, Int[1]
  # Layout(_4,_4), full coverage
  block:
    let r = complement(make_layout(4, 4))
    check r.shape, (4, 1), (Int[4], Int[1])
    check r.stride, (1, 16), (Int[1], Int[16])
  block:
    let r = complement(make_layout(4, 4), 1)
    check r.shape, 4, Int[4]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout(4, 4), 17)
    check r.shape, (4, 2), (Int[4], Int[2])
    check r.stride, (1, 16), (Int[1], Int[16])
  block:
    let r = complement(make_layout(4, 4), 32)
    check r.shape, (4, 2), (Int[4], Int[2])
    check r.stride, (1, 16), (Int[1], Int[16])
  block:
    let r = complement(make_layout(4, 4), (2, 2))
    check r.shape, 4, Int[4]
    check r.stride, 1, Int[1]
  # other rank-1 fills
  block:
    let r = complement(make_layout(6, 1), 24)
    check r.shape, 4, Int[4]
    check r.stride, 6, Int[6]
  block:
    let r = complement(make_layout(8, 1), 32)
    check r.shape, 4, Int[4]
    check r.stride, 8, Int[8]
  block:
    # full coverage, the complement is the lone (1):(coverage) mode
    let r = complement(make_layout(2, 1), 2)
    check r.shape, 1, Int[1]
    check r.stride, 2, Int[2]
  block:
    let r = complement(make_layout((2, 2), (1, 4)), 8)
    check r.shape, 2, Int[2]
    check r.stride, 2, Int[2]

# ── rank-2 and nested, static ─────────────────────────────────
proc runMultiDimensionStaticTests =
  block:
    # full coverage
    let r = complement(make_layout((2, 4), (1, 2)))
    check r.shape, 1, Int[1]
    check r.stride, 8, Int[8]
  block:
    # full coverage
    let r = complement(make_layout((2, 3), (1, 2)))
    check r.shape, 1, Int[1]
    check r.stride, 6, Int[6]
  block:
    let r = complement(make_layout((2, 4), (1, 4)))
    check r.shape, (2, 1), (Int[2], Int[1])
    check r.stride, (2, 16), (Int[2], Int[16])
  block:
    let r = complement(make_layout((2, 4), (1, 6)))
    check r.shape, (3, 1), (Int[3], Int[1])
    check r.stride, (2, 24), (Int[2], Int[24])
  block:
    let r = complement(make_layout((2, 4, 8), (8, 1, 64)))
    check r.shape, (2, 4, 1), (Int[2], Int[4], Int[1])
    check r.stride, (4, 16, 512), (Int[4], Int[16], Int[512])
  block:
    let r = complement(make_layout((2, 4, 8), (8, 1, 0)))
    check r.shape, (2, 1), (Int[2], Int[1])
    check r.stride, (4, 16), (Int[4], Int[16])
  block:
    let r = complement(make_layout((2, 4, 8), (8, 1, 0)), 460)
    check r.shape, (2, 29), (Int[2], Int[29])
    check r.stride, (4, 16), (Int[4], Int[16])
  block:
    let r = complement(make_layout(((2, 2), (2, 2)), ((1, 4), (8, 32))))
    check r.shape, (2, 2, 1), (Int[2], Int[2], Int[1])
    check r.stride, (2, 16, 64), (Int[2], Int[16], Int[64])
  block:
    let r = complement(make_layout(((2, 2), (2, 2)), ((1, 32), (8, 4))))
    check r.shape, (2, 2, 1), (Int[2], Int[2], Int[1])
    check r.stride, (2, 16, 64), (Int[2], Int[16], Int[64])
  block:
    # full coverage
    let r = complement(make_layout((4, 6), (1, 6)))
    check r.shape, 1, Int[1]
    check r.stride, 36, Int[36]
  block:
    let r = complement(make_layout((4, 2), (1, 10)))
    check r.shape, (2, 1), (Int[2], Int[1])
    check r.stride, (4, 20), (Int[4], Int[20])
  block:
    let r = complement(make_layout((4, 2), (1, 16)))
    check r.shape, (4, 1), (Int[4], Int[1])
    check r.stride, (4, 32), (Int[4], Int[32])
  block:
    # a nested shape with a scalar stride-0 collapses to (bound):(1)
    let r = complement(make_layout((8, 4), 0), 32)
    check r.shape, 32, Int[32]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout((8, 4), 0))
    check r.shape, 1, Int[1]
    check r.stride, 1, Int[1]
  block:
    let r = complement(make_layout((2, (3, 4)), (3, (1, 6))))
    check r.shape, 1, Int[1]
    check r.stride, 24, Int[24]
  block:
    let r = complement(make_layout((4, 10), (1, 10)))
    check r.shape, (2, 1), (Int[2], Int[1])
    check r.stride, (4, 100), (Int[4], Int[100])

proc runTests =
  runRank1StaticTests()
  runMultiDimensionStaticTests()

runTests()
echo "ALL TESTS PASSED"
