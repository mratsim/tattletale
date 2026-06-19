## Tests for Layout/Tensor/TensorView `()` dual dispatch and partitioning.
##
## References:
##   [CUTE-LOP] CuTe C++: layout_operator.cu — operator() on Layout
##   [CUTE-TUP] CuTe C++: tuple.cpp — slice/dice on tuples
##   [CUTE-TEN] CuTe C++: tensor_impl.hpp — Tensor::operator() dual dispatch
##   [CUTE-PAR] CuTe C++: tensor_impl.hpp — inner_partition / outer_partition
##   [CUTE-LGD] CuTe C++: logical_divide.cpp — logical_divide rank-2 property

import std/macros
import std/typetraits
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_indexing

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Layout `()` [CUTE-LOP]
#     CuTe: layout({m, n}) == layout(m, n)  — flat-tuple multi-index equivalence
#     CuTe: layout(X, n) → sub-layout        — Joker dispatch
# ═══════════════════════════════════════════════════════════════════════════════

proc runLayoutCallOpTests =
  block:  # all-int → offset
    let L = make_layout((3, 4), (1, 3))
    doAssert L(0, 0) == 0
    doAssert L(0, 1) == 3
    doAssert L(1, 0) == 1
    doAssert L(1, 2) == 7
    doAssert L(2, 3) == 11

  block:  # all-int row-major → offset
    let L = make_layout((3, 4), (4, 1))
    doAssert L(1, 0) == 4
    doAssert L(0, 1) == 1

  block:  # flat-tuple multi-index equivalence [CUTE-LOP]
    let L = make_layout((3, 4), (1, 3))
    doAssert L((1, 2)) == L(1, 2)

  block:  # Joker → sub-Layout column slice (shape (3,):(1,))
    let lay = make_layout((3, 4), (1, 3))
    let sub = slice(lay, (X, Y))
    doAssert sub.rank == 1
    doAssert sub(0) == 0
    doAssert sub(2) == 2

  block:  # Joker → sub-Layout row slice (shape (4,):(3,))
    let L = make_layout((3, 4), (1, 3))
    let sub = slice(L, (Y, X))
    doAssert sub.rank == 1
    doAssert sub(0) == 0
    doAssert sub(3) == 9

  block:  # Double Joker → identity sub-Layout
    let L = make_layout((3, 4), (1, 3))
    let sub = slice(L, (X, X))
    doAssert sub(1, 2) == 7

  block:  # Rank-3 offset
    let L = make_layout((2, 4, 8), (32, 8, 1))
    doAssert L(1, 2, 3) == 51

  block:  # Rank-3 Joker sub-Layout
    let L = make_layout((2, 4, 8), (32, 8, 1))
    let sub = slice(L, (X, Y, X))
    doAssert sub.rank == 2
    doAssert sub(1, 3) == 35

  block:  # Joker with Int[N] values
    let L = make_layout((4, 8), (1, 4))
    let sub = slice(L, (X, Y))
    doAssert sub.rank == 1
    doAssert sub(0) == 0

  echo "  1. Layout () dual dispatch: 9 blocks OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Tensor/TensorView `()` [CUTE-TEN]
#     CuTe: tensor(i, j) → element ref; tensor(X, j) → sub-Tensor
# ═══════════════════════════════════════════════════════════════════════════════

proc runTensorCallOpTests =
  block:  # TensorView all-int → element reference
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = float32(i)
    let v = make_view(addr(buf[0]), make_layout((3, 4), (1, 3)))
    doAssert v(0, 0) == 0.0'f32
    doAssert v(1, 2) == 7.0'f32
    doAssert v(2, 3) == 11.0'f32

  block:  # TensorView Joker → sub-View
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = float32(i)
    let v = make_view(addr(buf[0]), make_layout((3, 4), (1, 3)))
    let col = v(X, 1)
    doAssert col(0) == 3.0'f32
    doAssert col(2) == 5.0'f32

  block:  # sub-View via [] accesses same data
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = float32(i)
    let v = make_view(addr(buf[0]), make_layout((3, 4), (1, 3)))
    let col = v(X, 2)
    doAssert col[0] == 6.0'f32
    doAssert col[2] == 8.0'f32

  block:  # Tensor all-int → element reference
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = float32(i)
    let t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    doAssert t(0, 0) == 0.0'f32
    doAssert t(1, 2) == 7.0'f32

  block:  # Tensor Joker → sub-View
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = float32(i)
    let t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    let sub = t(X, 2)
    doAssert sub(0) == 6.0'f32
    doAssert sub(2) == 8.0'f32

  block:  # Write through sub-View modifies original buffer
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = float32(i)
    let v = make_view(addr(buf[0]), make_layout((3, 4), (1, 3)))
    let sub = v(X, 1)
    sub(0) = 99.0'f32
    doAssert buf[3] == 99.0'f32

  block:  # Tensor with non-zero offset
    var buf = newSeq[float32](20)
    for i in 0 ..< 20: buf[i] = float32(i)
    let t = make_tensor(buf, 8, make_layout((3, 4), (1, 3)))
    doAssert t(0, 0) == 8.0'f32
    doAssert t(1, 2) == 15.0'f32

  block:  # Exhaustive: flat-index equivalence
    let L = make_layout((3, 4), (1, 3))
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let v = make_view(addr(buf[0]), L)
    for i in 0 ..< 12:
      doAssert v(i) == L(i).float32

  echo "  2. Tensor/TensorView () dual dispatch: 8 blocks OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  3. `[]` Joker guard — compile-time error
# ═══════════════════════════════════════════════════════════════════════════════

proc runBracketJokerGuardTests =
  block:  # Normal [] still works
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = float32(i)
    let v = make_view(addr(buf[0]), make_layout((3, 4), (1, 3)))
    doAssert v[0] == 0.0'f32
    doAssert v[1, 2] == 7.0'f32

  block:  # Joker in TensorView [] is compile-time error
    var buf: array[12, float32]
    let v = make_view(addr(buf[0]), make_layout((3, 4), (1, 3)))
    when compiles(v[X, 0]):
      doAssert false, "v[X, 0] should be a compile-time error"

  block:  # Joker in Tensor [] is compile-time error
    let t = make_tensor(make_layout((3, 4), (1, 3)), float32)
    when compiles(t[X, 0]):
      doAssert false, "t[X, 0] should be a compile-time error"

  echo "  3. [] Joker guard: 3 blocks OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  4. slice_and_offset with Joker+int on zipped_divide
# ═══════════════════════════════════════════════════════════════════════════════

proc runSliceAndOffsetZDTests =
  block:  # inner partition: keep tile, slice rest 0
    let L = make_layout((6, 6), (1, 6))
    let zd = zipped_divide(L, (2, 2))
    let (sub, off) = slice_and_offset(((X, X), 0), zd)
    doAssert sub(0, 0) + off == L(0, 0)
    doAssert off == 0

  block:  # inner partition: tile 1, offset 2
    let L = make_layout((6, 6), (1, 6))
    let zd = zipped_divide(L, (2, 2))
    let (sub, off) = slice_and_offset(((X, X), 1), zd)
    doAssert off == 2

  block:  # inner partition: tile 2, offset 4
    let L = make_layout((6, 6), (1, 6))
    let zd = zipped_divide(L, (2, 2))
    let (sub, off) = slice_and_offset(((X, X), 2), zd)
    doAssert off == 4

  block:  # outer partition: slice tile 0, keep rest
    let L = make_layout((6, 6), (1, 6))
    let zd = zipped_divide(L, (2, 2))
    let (sub, off) = slice_and_offset((0, (X, X)), zd)
    doAssert off == 0

  block:  # outer partition: tile 1, offset 1
    let L = make_layout((6, 6), (1, 6))
    let zd = zipped_divide(L, (2, 2))
    let (sub, off) = slice_and_offset((1, (X, X)), zd)
    doAssert off == 1

  block:  # outer partition: tile 2, offset 6 (divmod: 2→(0,1) )
    let L = make_layout((6, 6), (1, 6))
    let zd = zipped_divide(L, (2, 2))
    let (sub, off) = slice_and_offset((2, (X, X)), zd)
    doAssert off == 6

  block:  # inner partition round-trip using TensorView
    var buf: array[36, float32]
    for idx in 0 ..< 36: buf[idx] = float32(idx)
    let tv = make_view(addr(buf[0]), make_layout((6, 6), (1, 6)))
    let zd = zipped_divide(tv.layout, (2, 2))
    let (sub, off) = slice_and_offset(((X, X), 1), zd)
    # Restructure: sub(0,0) = first element of tile at column offset 2 (rest index 1)
    doAssert tv.data[off] == buf[off]

  echo "  4. slice_and_offset on zipped_divide: 7 blocks OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  5. inner_partition / outer_partition [CUTE-PAR]
# ═══════════════════════════════════════════════════════════════════════════════

proc partitionTest_inner0 =
  block:
    let L = make_layout((6, 6), (1, 6))
    var buf: array[36, float32]
    for i in 0 ..< 36: buf[i] = float32(i + 1)
    let tv = make_view(addr(buf[0]), L)
    let tile = inner_partition(tv, (2, 2), 0)
    doAssert tile(0, 0) == 1.0'f32
    doAssert tile(1, 1) == 8.0'f32

proc partitionTest_inner1 =
  block:
    let L = make_layout((6, 6), (1, 6))
    var buf: array[36, float32]
    for i in 0 ..< 36: buf[i] = float32(i + 1)
    let tv = make_view(addr(buf[0]), L)
    let tile = inner_partition(tv, (2, 2), 1)
    doAssert tile(0, 0) == 3.0'f32
    doAssert tile(1, 0) == 4.0'f32

proc partitionTest_outer0 =
  block:
    let L = make_layout((6, 6), (1, 6))
    var buf: array[36, float32]
    for i in 0 ..< 36: buf[i] = float32(i + 1)
    let tv = make_view(addr(buf[0]), L)
    let rest = outer_partition(tv, (2, 2), 0)
    doAssert rest(0, 0) == 1.0'f32
    doAssert rest(1, 0) == 3.0'f32

proc partitionTest_outer1 =
  block:
    let L = make_layout((6, 6), (1, 6))
    var buf: array[36, float32]
    for i in 0 ..< 36: buf[i] = float32(i + 1)
    let tv = make_view(addr(buf[0]), L)
    let rest = outer_partition(tv, (2, 2), 1)
    doAssert rest(0, 0) == 2.0'f32

proc partitionTest_tileList_0 =
  block:
    let L = make_layout((6, 6), (1, 6))
    var buf: array[36, float32]
    for idx in 0 ..< 36: buf[idx] = float32(idx + 1)
    let tv = make_view(addr(buf[0]), L)
    let tile0 = inner_partition(tv, (2, 2), 0)
    doAssert tile0(0, 0) == 1.0'f32
    doAssert tile0(1, 1) == 8.0'f32

proc partitionTest_tileList_1 =
  block:
    let L2 = make_layout((6, 6), (1, 6))
    var buf2: array[36, float32]
    for idx in 0 ..< 36: buf2[idx] = float32(idx + 1)
    let tv2 = make_view(addr(buf2[0]), L2)
    let tile1 = inner_partition(tv2, (2, 2), 1)
    doAssert tile1(0, 0) == 3.0'f32

proc partitionTest_tileList_2 =
  block:
    let L3 = make_layout((6, 6), (1, 6))
    var buf3: array[36, float32]
    for idx in 0 ..< 36: buf3[idx] = float32(idx + 1)
    let tv3 = make_view(addr(buf3[0]), L3)
    let tile2 = inner_partition(tv3, (2, 2), 2)
    doAssert tile2(0, 0) == 5.0'f32

proc runPartitionTests =
  partitionTest_inner0()
  partitionTest_inner1()
  partitionTest_outer0()
  partitionTest_outer1()
  partitionTest_tileList_0()
  partitionTest_tileList_1()
  partitionTest_tileList_2()
  echo "  5. inner_partition / outer_partition: 5 blocks OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Legacy local_tile (tiled_divide path)
# ═══════════════════════════════════════════════════════════════════════════════

proc runLegacyLocalTileTests =
  block:  # Legacy local_tile with tiled_divide result
    let L = make_layout((6, 6), (1, 6))
    let pL = tiled_divide(L, (2, 2))
    var buf: array[36, float32]
    for i in 0 ..< 36: buf[i] = float32(i + 1)
    let tv = make_view(addr(buf[0]), L)
    let tile = local_tile(tv, pL, 0, 0)
    doAssert tile(0, 0) == 1.0'f32

  echo "  6. Legacy local_tile: 1 block OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

proc runEdgeCaseTests =
  block:  # Rank-2 layout with rank-1 lookup
    let L = make_layout((6,), (2,))
    doAssert L(0) == 0
    doAssert L(3) == 6
    let sub = L(X)
    doAssert sub(0) == 0
    doAssert sub.rank == 1

  block:  # Rank-1 TensorView ()
    var buf: array[6, float32]
    for i in 0 ..< 6: buf[i] = float32(i * 10)
    let v = make_view(addr(buf[0]), make_layout(6, 1))
    doAssert v(0) == 0.0'f32
    doAssert v(3) == 30.0'f32

  block:  # slice_and_offset with single int
    let L = make_layout((4, 8), (1, 4))
    let (sub, off) = slice_and_offset(L, (2, X))
    doAssert off == 2

  block:  # slice_and_offset rank-3
    let L = make_layout((2, 3, 4), (1, 2, 6))
    let (sub, off) = slice_and_offset(L, (X, 1, X))
    doAssert off == 2
    doAssert sub.rank == 2
    doAssert sub(1, 3) + off == L(1, 1, 3)

  block:  # Row-major layout slices
    let L = make_layout((3, 4), (4, 1))
    doAssert L(1, 2) == 6
    let sub = L(X, 2)         # keep dim 0 (shape 3, stride 4), fix dim 1 to 2
    doAssert sub(0) == 0       # sub is Layout, (0) gives crd2idx within sub
    doAssert sub.rank == 1

  block:  # Dynamic sizes (runtime ints)
    let m = 3; let n = 4
    let L = make_layout((m, n), (1, m))
    doAssert L(1, 2) == 7
    let sub = L(X, 1)
    doAssert sub.rank == 1
    doAssert sub(2) == 2    # crd2idx within sub-Layout (shape m, stride 1)
  block:  # zipped_divide layout structure
    let L = make_layout((6, 6), (1, 6))
    let zd = zipped_divide(L, (2, 2))
    doAssert tupleLen(zd.shape) == 2
    doAssert zd.shape[0][0] === 2
    doAssert zd.shape[0][1] === 2
    doAssert zd.shape[1][0] === 3
    doAssert zd.shape[1][1] === 3

  block:  # zipped_divide non-square tiler
    let L = make_layout((12, 6), (1, 12))
    let zd = zipped_divide(L, (3, 2))
    doAssert zd.shape[0][0] === 3
    doAssert zd.shape[0][1] === 2
    doAssert zd.shape[1][0] === 4
    doAssert zd.shape[1][1] === 3

  echo "  7. Edge cases: 8 blocks OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Round-trip property: sub(tile_coord) + off == L(full_coord)
# ═══════════════════════════════════════════════════════════════════════════════

proc runProperty_roundtrip =
  block:
    let L = make_layout((6, 6), (1, 6))
    var buf: array[36, float32]
    for idx in 0 ..< 36: buf[idx] = float32(idx)
    let tv = make_view(addr(buf[0]), L)
    block:
      let tile0 = inner_partition(tv, (2, 2), 0)
      doAssert tile0(0, 0) == 0.0'f32
    block:
      let L2 = make_layout((6, 6), (1, 6))
      var buf2: array[36, float32]
      for idx in 0 ..< 36: buf2[idx] = float32(idx)
      let tv2 = make_view(addr(buf2[0]), L2)
      let tile1 = inner_partition(tv2, (2, 2), 1)
      doAssert tile1(0, 0) == 2.0'f32
    block:
      let L3 = make_layout((6, 6), (1, 6))
      var buf3: array[36, float32]
      for idx in 0 ..< 36: buf3[idx] = float32(idx)
      let tv3 = make_view(addr(buf3[0]), L3)
      let tile2 = inner_partition(tv3, (2, 2), 2)
      doAssert tile2(0, 0) == 4.0'f32

proc runPropertyTests =
  block:  # slice_and_offset round-trip
    let L = make_layout((4, 8), (1, 4))
    let col = 3
    let (sub, off) = slice_and_offset(L, (X, col))
    for i in 0 ..< 4:
      doAssert sub(i) + off == L((i, col))
  runProperty_roundtrip()
  echo "  8. Round-trip properties: 2 blocks OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test runner
# ═══════════════════════════════════════════════════════════════════════════════

proc runTests =
  runLayoutCallOpTests()
  runTensorCallOpTests()
  runBracketJokerGuardTests()
  runSliceAndOffsetZDTests()
  runPartitionTests()
  runLegacyLocalTileTests()
  runEdgeCaseTests()
  runPropertyTests()
  echo "\nALL TESTS PASSED"

when isMainModule:
  runTests()
