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
  block:
    let l = make_ordered_layout((2, 3), (0, 1))
    doAssert l === ((2, 3), (1, 2))
    doAssert typeof(l.stride[0]) is Int[1]
    doAssert typeof(l.stride[1]) is Int[2]
    echo "    1. static 2D col-major phantom strides: OK"

  block:
    let l = make_ordered_layout((2, 3, 4), (0, 2, 1))
    doAssert l === ((2, 3, 4), (1, 8, 2))
    doAssert typeof(l.stride[1]) is Int[8]
    echo "    2. static 3D permutation phantom strides: OK"

  block:
    let l = make_ordered_layout(2, 0)
    doAssert l === (2, 1)
    doAssert typeof(l.stride) is Int[1]
    echo "    3. scalar shape scalar stride: OK"

proc runRuntimeLeafTests =
  block:
    let rows = 7
    let l = make_ordered_layout((rows, 32), (0, 1))
    doAssert l.shape[0] == rows
    doAssert l.shape[1] === Int[32]()
    doAssert l.stride[0] === Int[1]()
    doAssert typeof(l.stride[1]) is int
    doAssert l.stride[1] == rows
    echo "    4. runtime rows, static 32: (Int[1], int) strides: OK"

  block:
    let rows = 7
    let l = make_ordered_layout((rows, 32, 4), (1, 0, 2))
    doAssert l.stride[1] === Int[1]()
    doAssert l.stride[0] === Int[32]()
    doAssert typeof(l.stride[2]) is int
    doAssert l.stride[2] == rows * 32
    echo "    5. runtime rows between static leaves: folded neighbors: OK"

  block:
    let rows = 7
    let l = make_ordered_layout(rows, 0)
    doAssert l.stride === Int[1]()
    echo "    6. scalar runtime shape: OK"

proc runOrderRegimeTests =
  block:
    const order = (1, 0)
    let l = make_ordered_layout((2, 3), order)
    doAssert l === ((2, 3), (3, 1))
    echo "    7. const order symbol: OK"

  block:
    var ordDyn = (1, 0)
    let l = make_ordered_layout((2, 3), ordDyn)
    doAssert l === ((2, 3), (1, 2))
    echo "    8. runtime order entries rank after static: OK"

  block:
    var shape = (Int[2](), Int[3]())
    let l = make_ordered_layout(shape, (0, 1))
    doAssert l === ((2, 3), (1, 2))
    echo "    9. Int-leaf tuple variable shape: OK"

  block:
    let l = make_ordered_layout((2, 3), (Int[1](), Int[0]()))
    doAssert l === ((2, 3), (3, 1))
    doAssert typeof(l.stride[0]) is Int[3]
    echo "    10. Int-node order leaves rank as their phantom value: OK"

  block:
    var dyn = 1
    let l = make_ordered_layout((2, 3, 4), (0, dyn, 2))
    doAssert l === ((2, 3, 4), (1, 8, 2))
    doAssert typeof(l.stride[1]) is Int[8]
    echo "    11. mixed static/runtime order keeps literal ranks: OK"

  block:
    var dyn = 2
    let l = make_ordered_layout((2, 3, 4), (Int[1](), dyn, Int[0]()))
    doAssert l === ((2, 3, 4), (4, 8, 1))
    echo "    12. Int-node and runtime order leaves mix: OK"

  block:
    let l = make_ordered_layout(((2, 3), 4), (0, 1, 2))
    doAssert l === (((2, 3), 4), ((1, 2), 2))
    doAssert typeof(l.stride[0][0]) is Int[1]
    echo "    13. nested shape keeps its nesting: OK"

proc runBoundarySymbolTests =
  block:
    const s = (2, 3)
    let l = make_ordered_layout(s, (0, 1))
    doAssert l === ((2, 3), (1, 2))
    doAssert typeof(l.stride[0]) is Int[1]
    echo "    14. const symbol shape folds: OK"

  block:
    let s = (2, 3)
    let l = make_ordered_layout(s, (0, 1))
    doAssert l.stride[1] == 2
    doAssert typeof(l.stride[1]) is int
    echo "    15. let symbol shape stays a runtime leaf: OK"

proc runRefusalTests =
  block:
    doAssert not compiles(make_ordered_layout((2, 3, 4), 1))
    echo "    16. scalar order on a rank-3 shape refuses at compile time: OK"

  block:
    doAssert not compiles(make_ordered_layout((2, 3), (0, 1, 2)))
    echo "    17. overlong order tuple refuses at compile time: OK"

proc runTests =
  echo "--- static result types ---"
  runStaticTypeTests()
  echo "--- runtime shape leaves ---"
  runRuntimeLeafTests()
  echo "--- order regimes ---"
  runOrderRegimeTests()
  echo "--- boundary symbols ---"
  runBoundarySymbolTests()
  echo "--- rank refusals ---"
  runRefusalTests()

when isMainModule:
  runTests()
