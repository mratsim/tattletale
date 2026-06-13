## Tests for Tensors: owning Tensor (seq-backed) and TensorView (ptr-backed).
##
## Reference:
##   - CuTe C++: tensor_impl.hpp — operator[] uses layout()(i) for flat indexing
##   - CuTe C++: layout_operator.cu — layout({m, n}) == layout(m, n) flat-tuple ⇔ multi-index
##   - Python: tensor-layouts/tests/tensor.py — test_flat_eval_*, test_*_indexing
{.experimental: "callOperator".}

import std/macros
import workspace/ceramic/src/int_tuples {.all.}
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

# ═════════════════════════════════════════════════════════════════════════════
#  Tensor construction
# ═════════════════════════════════════════════════════════════════════════════

proc runTensorConstructionTests =
  block:  # Tensor from seq + offset + layout (owning copy of seq)
    let buf = newSeq[float32](16)
    var t = make_tensor(buf, 0, make_layout((4, 4), (1, 4)))
    doAssert t.offset == 0

  block:  # rank
    var buf = newSeq[float32](16)
    var t = make_tensor(buf, 0, make_layout((4, 4), (1, 4)))
    doAssert t.rank == 2

  block:  # size
    let buf = newSeq[float32](16)
    let t = make_tensor(buf, 0, make_layout((4, 4), (1, 4)))
    doAssert t.size === 16

  block:  # cosize
    var buf = newSeq[float32](32)
    var t = make_tensor(buf, 0, make_layout((4, 4), (1, 4)))
    doAssert t.cosize === 16

  block:  # Tensor from layout (owning, allocates its own seq)
    var t = make_tensor(make_layout((3, 5), (1, 3)), float32)
    doAssert t.rank == 2
    doAssert t.size === 15
    doAssert t.cosize === 15
    doAssert t.data.len == 15

  block:  # Tensor from shape only (compact col-major)
    var t = make_tensor(make_layout((2, 3), LayoutLeft), float64)
    doAssert t.size === 6

  block:  # Tensor with LayoutRight (row-major)
    var t = make_tensor(make_layout((2, 3), LayoutRight), int32)
    doAssert t.size === 6

  block:  # View with non-zero offset
    var buf = newSeq[float32](32)
    var t = make_tensor(buf, 10, make_layout((2, 3), (1, 2)))
    doAssert t.offset == 10

# ═════════════════════════════════════════════════════════════════════════════
#  TensorView construction
# ═════════════════════════════════════════════════════════════════════════════

proc runTensorViewTests =
  block:  # View from array
    var buf: array[12, float32]
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    doAssert v.rank == 2
    doAssert v.size === 12

  block:  # View from seq
    var buf = newSeq[float32](12)
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    doAssert v.size === 12

  block:  # View from Tensor (conversion)
    var t = make_tensor(make_layout((3, 4), (1, 3)), float32)
    let v = t.view()
    doAssert v.size === 12

# ═════════════════════════════════════════════════════════════════════════════
#  Flat indexing — operator()
#  Matches CuTe C++: Tensor::operator()(int i) → data()[layout()(i)]
#  Matches Python tensor-layouts: test_flat_eval_*
proc runFlatIndexTests =
  # Python/CuTe: t(i) == layout(i) — flat index decomposes via idx2crd,
  # then computes offset via crd2idx (col-major decomposition).
  # With data: v(i) == data[layout(i)].

  block:  # Rank-2 column-major: t(i) == layout(i) for all i
    let L = make_layout((3, 4), (1, 3))
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let v = make_view(addr(buf[0]), L)
    for i in 0 ..< 12:
      doAssert v(i) == L(i).float32

  block:  # Rank-2 row-major: t(i) == layout(i) for all i
    let L = make_layout((3, 4), (4, 1))
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let v = make_view(addr(buf[0]), L)
    for i in 0 ..< 12:
      doAssert v(i) == L(i).float32

  block:  # Rank-3: t(i) == layout(i) for all i
    let L = make_layout((2, 4, 8), (32, 8, 1))
    var buf: array[64, float32]
    for i in 0 ..< 64: buf[i] = i.float32
    let v = make_view(addr(buf[0]), L)
    for i in 0 ..< 64:
      doAssert v(i) == L(i).float32

  block:  # With non-zero offset: t(i) == offset + layout(i)
    let L = make_layout((4, 8), (8, 1))
    var buf = newSeq[float32](44)
    for i in 0 ..< 44: buf[i] = i.float32
    var t = make_tensor(buf, 12, L)
    for i in 0 ..< 32:
      doAssert t(i) == (12 + L(i)).float32

  block:  # Write via t(i) = val, read back through t
    var buf = newSeq[float32](6)
    var t = make_tensor(buf, 0, make_layout((2, 3), (1, 2)))
    for i in 0 ..< 6:
      t(i) = (i * 10).float32
    for i in 0 ..< 6:
      doAssert t(i) == (i * 10).float32

  block:  # Write via v(i) = val modifies backing storage
    var buf: array[6, float32]
    let v = make_view(addr(buf[0]), make_layout((2, 3), (1, 2)))
    for i in 0 ..< 6:
      v(i) = (i * 10).float32
    for i in 0 ..< 6:
      doAssert buf[i] == (i * 10).float32

  block:  # Single-int multi-index matches decomposed multi-index
    let L = make_layout((3, 4), (1, 3))
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let v = make_view(addr(buf[0]), L)
    for i in 0 ..< 12:
      let coord = idx2crd(L, i)
      doAssert v(i) == v[coord], "t(" & $i & ") should equal t(" & $coord & ")"
  block:  # t(flat_idx) always equals multi-index that idx2crd produces
    # Python: test_single_index_flat_eval
    let lay = make_layout((4, 8), (1, 4))
    var buf: array[32, float32]
    for i in 0 ..< 32: buf[i] = i.float32
    let v = make_view(addr(buf[0]), lay)
    # Single element: idx2crd(2, (4,8)) = (2, 0) → offset 2
    doAssert v[2] == 2.0'f32
    for i in 0 ..< 32:
      doAssert v(i) == i.float32

# ═════════════════════════════════════════════════════════════════════════════
#  Multi-index indexing — operator[] with tuple or individual int args
#  Matches CuTe C++: Tensor::operator()(Coord const&) → data()[layout()(coord)]
#  Matches Python tensor-layouts: test_*_indexing
# ═════════════════════════════════════════════════════════════════════════════

proc runMultiIndexTests =
  block:  # Col-major Tensor: (i,j) -> data[i + j*M]  (Python: test_column_major_indexing)
    var buf = newSeq[float32](12)
    var t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    for idx in 0 ..< 12: t(idx) = idx.float32
    doAssert t[(0, 0)] == 0.0'f32
    doAssert t[(2, 0)] == 2.0'f32
    doAssert t[(0, 1)] == 3.0'f32
    doAssert t[(1, 2)] == 7.0'f32
    doAssert t[(2, 3)] == 11.0'f32

  block:  # Exhaustive col-major: for all (i,j), t(i,j) == i + j*M
    let M = 3; let N = 4
    var buf: array[12, float32]
    for i in 0 ..< M:
      for j in 0 ..< N:
        buf[i + j*M] = (i + j*M).float32
    let v = make_view(addr(buf[0]), make_layout((M, N), (1, M)))
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert v[(i, j)] == (i + j*M).float32

  block:  # Exhaustive row-major: for all (i,j), t(i,j) == i*N + j
    let M = 3; let N = 4
    var buf: array[12, float32]
    for i in 0 ..< M:
      for j in 0 ..< N:
        buf[i*N + j] = (i*N + j).float32
    let v = make_view(addr(buf[0]), make_layout((M, N), (N, 1)))
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert v[(i, j)] == (i*N + j).float32

  block:  # 2-arg: t[i, j] tuple-free syntax
    var buf = newSeq[float32](12)
    var t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    for idx in 0 ..< 12: t(idx) = idx.float32
    doAssert t[(0, 0)] == 0.0'f32
    doAssert t[(1, 2)] == 7.0'f32

  block:  # Col-major TensorView
    var buf: array[12, float32]
    for idx in 0 ..< 12: buf[idx] = idx.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    doAssert v[(0, 0)] == 0.0'f32
    doAssert v[(0, 1)] == 3.0'f32
    doAssert v[(1, 2)] == 7.0'f32

  block:  # Row-major TensorView
    var buf: array[12, float32]
    for idx in 0 ..< 12: buf[idx] = idx.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (4, 1)))
    doAssert v[(1, 0)] == 4.0'f32
    doAssert v[(0, 1)] == 1.0'f32

  block:  # Write via TensorView modifies original
    var buf: array[6, float32]
    let p = addr(buf[0])
    let v = make_view(p, make_layout((2, 3), (1, 2)))
    v[(0, 0)] = 1.0'f32
    v[(1, 1)] = 5.0'f32
    v[(0, 2)] = 9.0'f32
    doAssert buf[0] == 1.0'f32
    doAssert buf[3] == 5.0'f32
    doAssert v[(0, 0)] == 1.0'f32
    doAssert v[(1, 1)] == 5.0'f32
    doAssert v[(0, 2)] == 9.0'f32

  block:  # Dynamic col-major (M,N not compile-time)
    let M = 5; let N = 4
    var buf: array[20, float32]
    for idx in 0 ..< (M*N): buf[idx] = idx.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((M, N), (1, M)))
    doAssert v[(0, 0)] == 0.0'f32
    doAssert v[(4, 0)] == 4.0'f32
    doAssert v[(0, 1)] == 5.0'f32
    doAssert v[(3, 2)] == 13.0'f32

  block:  # Write via owning tensor
    var t = make_tensor(make_layout((2, 3), (1, 2)), float32)
    t[(0, 0)] = 10.0'f32
    t[(1, 2)] = 20.0'f32
    doAssert t[(0, 0)] == 10.0'f32
    doAssert t[(1, 2)] == 20.0'f32

  # ───────────────────────────────────────────────────────────────────────
  # Rank-3 indexing (Python: test_rank3_tensor)
  # ───────────────────────────────────────────────────────────────────────

  block:  # 3D col-major: (b, h, w) -> data[b*32 + h*8 + w]
    var buf: array[64, float32]
    for i in 0 ..< 64: buf[i] = i.float32
    let v = make_view(addr(buf[0]), make_layout((2, 4, 8), (32, 8, 1)))
    doAssert v[(0, 0, 0)] == 0.0'f32
    doAssert v[(1, 0, 0)] == 32.0'f32
    doAssert v[(0, 1, 0)] == 8.0'f32
    doAssert v[(0, 0, 1)] == 1.0'f32
    doAssert v[(1, 2, 3)] == (32 + 16 + 3).float32

  block:  # Exhaustive rank-3
    var buf: array[64, float32]
    for i in 0 ..< 64: buf[i] = i.float32
    let v = make_view(addr(buf[0]), make_layout((2, 4, 8), (32, 8, 1)))
    for b in 0 ..< 2:
      for h in 0 ..< 4:
        for w in 0 ..< 8:
          let expected = b*32 + h*8 + w
          doAssert v[(b, h, w)] == expected.float32

  # ───────────────────────────────────────────────────────────────────────
  # Offset tensor: tensor(coord) == offset + layout[coord]
  # (Python: test_with_offset_indexing)
  # ───────────────────────────────────────────────────────────────────────

  block:  # With non-zero offset, all coordinates shifted
    let offset = 1000
    let M = 3; let N = 4
    var buf = newSeq[float32](1012)
    for i in 0 ..< 1012: buf[i] = i.float32
    var t = make_tensor(buf, offset, make_layout((M, N), (1, M)))
    let base = make_tensor(buf, 0, make_layout((M, N), (1, M)))
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert t[(i, j)] == offset.float32 + base[(i, j)]

  block:  # Tensor(i) single-int multi-index
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let v = make_view(addr(buf[0]), make_layout((3, 4), (1, 3)))
    doAssert v[0] == 0.0'f32
    doAssert v[4] == 4.0'f32
    doAssert v[11] == 11.0'f32

# ═════════════════════════════════════════════════════════════════════════════
#  Tensor slicing with Joker (`_`)
# ═════════════════════════════════════════════════════════════════════════════

proc runSliceTests =
  block:  # Slice row: t.slice((row, _)) returns 1D Tensor
    var buf = newSeq[float32](12)
    var t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    for i in 0 ..< 12: t(i) = i.float32
    let row1 = t.slice((1, _))
    doAssert row1.rank == 1
    doAssert row1.size === 4
    doAssert row1[0] == 1.0'f32
    doAssert row1[3] == 10.0'f32

  block:  # Slice column: t.slice((_, col)) returns 1D Tensor
    var buf = newSeq[float32](12)
    var t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    for i in 0 ..< 12: t(i) = i.float32
    let col1 = t.slice((_, 1))
    doAssert col1.rank == 1
    doAssert col1.size === 3
    doAssert col1[0] == 3.0'f32
    doAssert col1[1] == 4.0'f32
    doAssert col1[2] == 5.0'f32

  block:  # TensorView slice: v.slice((row, _))
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    let row = v.slice((1, _))
    doAssert row.rank == 1
    doAssert row.size === 4
    doAssert row[0] == 1.0'f32
    doAssert row[3] == 10.0'f32

  block:  # TensorView slice column
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    let col = v.slice((_, 2))
    doAssert col[0] == 6.0'f32
    doAssert col[2] == 8.0'f32

  block:  # Slice from owning tensor
    var t = make_tensor(make_layout((3, 4), (1, 3)), float32)
    for i in 0 ..< 12: t(i) = i.float32
    let col = t.slice((_, 2))
    doAssert col[0] == 6.0'f32
    doAssert col[2] == 8.0'f32


# ═════════════════════════════════════════════════════════════════════════════
#  displace — offset tensor and return sub-view with new shape
# ═════════════════════════════════════════════════════════════════════════════

proc runDisplaceTests =
  block:  # Basic offset: displace((3,2)) on 10x10 -> data pointer advances by 3+2*10=23
    var buf: array[100, float32]
    for i in 0 ..< 100: buf[i] = i.float32
    let v = make_view(addr(buf[0]), make_layout((10, 10), (1, 10)))
    let sub = displace(v, (3, 2))
    doAssert sub.layout.shape === (7, 8) and sub.layout.stride === (1, 10)
    doAssert sub[0, 0] == 23.0'f32
    doAssert sub[1, 0] == 24.0'f32
    doAssert sub[0, 1] == 33.0'f32

  block:  # No-offset displace: displace((0,0)) -> shape unchanged
    var buf: array[100, float32]
    for i in 0 ..< 100: buf[i] = i.float32
    let v = make_view(addr(buf[0]), make_layout((10, 10), (1, 10)))
    let sub = displace(v, (0, 0))
    doAssert sub.layout.shape === (10, 10) and sub.layout.stride === (1, 10)
    doAssert sub[0, 0] == 0.0'f32
    doAssert sub[9, 9] == 99.0'f32

  block:  # Owned Tensor displace
    var buf = newSeq[float32](100)
    for i in 0 ..< 100: buf[i] = i.float32
    let t = make_tensor(buf, 0, make_layout((10, 10), (1, 10)))
    let sub = displace(t, (3, 2))
    doAssert sub.layout.shape === (7, 8) and sub.layout.stride === (1, 10)
    doAssert sub[0, 0] == 23.0'f32

  echo "  displace: 3 cases OK"

# ═════════════════════════════════════════════════════════════════════════════
#  Tensor of different element types
# ═════════════════════════════════════════════════════════════════════════════

proc runTypeTests =
  block:  # int32 tensor
    var buf: array[8, int32]
    for i in 0 ..< 8: buf[i] = i.int32 * 10
    let p = addr(buf[0])
    let v = make_view(p, make_layout((2, 4), (1, 2)))
    doAssert v[(0, 0)] == 0
    doAssert v[(1, 1)] == 30

  block:  # float64 tensor
    var buf: array[6, float64]
    buf[0] = 3.14
    let p = addr(buf[0])
    let v = make_view(p, make_layout((2, 3), (1, 2)))
    doAssert v[(0, 0)] == 3.14

  block:  # int64 tensor
    var buf: array[4, int64]
    buf[2] = 42
    let p = addr(buf[0])
    let v = make_view(p, make_layout((2, 2), (1, 2)))
    doAssert v[(0, 1)] == 42

# ═════════════════════════════════════════════════════════════════════════════
#  Test runner
# ═════════════════════════════════════════════════════════════════════════════

proc runTests =
  runTensorConstructionTests()
  runTensorViewTests()
  runFlatIndexTests()
  runMultiIndexTests()
  runSliceTests()
  runDisplaceTests()
  runTypeTests()

when isMainModule:
  runTests()
  echo "OK: all tensor tests passed"
