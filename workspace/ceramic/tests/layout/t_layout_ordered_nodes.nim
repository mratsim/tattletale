## Layout nodes: make_ordered_layout over IntOrIntTuple operands
##
## Test that make_ordered_layout keeps static shape leaves as folded phantom
## Int[...] strides, lets runtime shape leaves compute runtime strides through
## the same body, and ranks dynamic order entries after static ones.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests/t_layout_ordered_nodes workspace/ceramic/tests/layout/t_layout_ordered_nodes.nim

import std/typetraits
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra

proc runStaticTypeTests =
  # The all-static path keeps phantom Int[...] stride leaves, folded in place.
  # column-major 2D, dimension 0 fastest
  block:
    let l = make_ordered_layout((2, 3), (0, 1))
    doAssert l === ((2, 3), (1, 2))
    doAssert typeof(l.stride[0]) is Int[1]
    doAssert typeof(l.stride[1]) is Int[2]
    echo "    1. static 2D col-major phantom strides: OK"

  block:  # 3D permutation
    let l = make_ordered_layout((2, 3, 4), (0, 2, 1))
    doAssert l === ((2, 3, 4), (1, 8, 2))
    doAssert typeof(l.stride[1]) is Int[8]
    echo "    2. static 3D permutation phantom strides: OK"

  block:  # scalar shape keeps a scalar stride
    let l = make_ordered_layout(2, 0)
    doAssert l === (2, 1)
    doAssert typeof(l.stride) is Int[1]
    echo "    3. scalar shape scalar stride: OK"

proc runRuntimeLeafTests =
  # A runtime shape leaf is a degenerate input: the same body computes its strides at runtime and the static siblings still fold.
  block:  # (rows, 32), order (0,1): stride[0] folds, stride[1] is the rows mul
    let rows = 7
    let l = make_ordered_layout((rows, 32), (0, 1))
    doAssert l.shape[0] == rows
    doAssert l.shape[1] === Int[32]()
    doAssert l.stride[0] === Int[1]()
    doAssert typeof(l.stride[1]) is int
    doAssert l.stride[1] == rows
    echo "    4. runtime rows, static 32: (Int[1], int) strides: OK"

  block:  # a static leaf folds into a stride next to a runtime sibling
    let rows = 7
    let l = make_ordered_layout((rows, 32, 4), (1, 0, 2))
    doAssert l.stride[1] === Int[1]()
    doAssert l.stride[0] === Int[32]()
    doAssert typeof(l.stride[2]) is int
    doAssert l.stride[2] == rows * 32
    echo "    5. runtime rows between static leaves: folded neighbors: OK"

  block:  # scalar runtime shape
    let rows = 7
    let l = make_ordered_layout(rows, 0)
    doAssert l.stride === Int[1]()
    echo "    6. scalar runtime shape: OK"

proc runOrderRegimeTests =
  # Order leaves keep their regime: static entries rank by value, runtime entries rank after every static entry in position order.
  block:  # const order symbol
    const order = (1, 0)
    let l = make_ordered_layout((2, 3), order)
    doAssert l === ((2, 3), (3, 1))
    echo "    7. const order symbol: OK"

  block:  # runtime order entries, all dynamic
    var ordDyn = (1, 0)
    let l = make_ordered_layout((2, 3), ordDyn)
    doAssert l === ((2, 3), (1, 2))
    echo "    8. runtime order entries rank after static: OK"

  block:  # Int-leaf tuple variable shape
    var shape = (Int[2](), Int[3]())
    let l = make_ordered_layout(shape, (0, 1))
    doAssert l === ((2, 3), (1, 2))
    echo "    9. Int-leaf tuple variable shape: OK"

proc runTests =
  echo "--- static result types ---"
  runStaticTypeTests()
  echo "--- runtime shape leaves ---"
  runRuntimeLeafTests()
  echo "--- order regimes ---"
  runOrderRegimeTests()

when isMainModule:
  runTests()
