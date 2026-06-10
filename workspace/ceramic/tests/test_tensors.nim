## Tests for Tensors: owning Tensor (seq-backed) and TensorView (ptr-backed).
##
## Reference:
##   - CuTe C++: tensor_impl.hpp
##   - Python: tensor-layouts/tests/tensor.py

import std/macros
import workspace/ceramic/src/int_tuples {.all.}
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

# ═══════════════════════════════════════════════════════════════
#  Tensor construction
# ═══════════════════════════════════════════════════════════════

proc runTensorConstructionTests* =
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
    doAssert t.size == 16

  block:  # cosize
    var buf = newSeq[float32](32)
    var t = make_tensor(buf, 0, make_layout((4, 4), (1, 4)))
    doAssert t.cosize == 16

  block:  # Tensor from layout (owning, allocates its own seq)
    var t = make_tensor(make_layout((3, 5), (1, 3)), float32)
    doAssert t.rank == 2
    doAssert t.size == 15
    doAssert t.cosize == 15
    doAssert t.data.len == 15

  block:  # Tensor from shape only (compact col-major)
    var t = make_tensor(make_layout((2, 3)), float64)
    doAssert t.size == 6

  block:  # Tensor with LayoutRight (row-major)
    var t = make_tensor(make_layout((2, 3), LayoutRight), int32)
    doAssert t.size == 6

  block:  # View with non-zero offset
    var buf = newSeq[float32](32)
    var t = make_tensor(buf, 10, make_layout((2, 3), (1, 2)))
    doAssert t.offset == 10

# ═══════════════════════════════════════════════════════════════
#  TensorView construction
# ═══════════════════════════════════════════════════════════════

proc runTensorViewTests* =
  block:  # View from array
    var buf: array[12, float32]
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    doAssert v.rank == 2
    doAssert v.size == 12

  block:  # View from seq
    var buf = newSeq[float32](12)
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    doAssert v.size == 12

  block:  # View from Tensor (conversion)
    var t = make_tensor(make_layout((3, 4), (1, 3)), float32)
    let v = t.view()
    doAssert v.size == 12

# ═══════════════════════════════════════════════════════════════
#  Flat indexing — operator[]
# ═══════════════════════════════════════════════════════════════

proc runTensorFlatIndexTests* =
  block:  # Fill + flat readback through Tensor
    var buf = newSeq[float32](12)
    for i in 0 ..< 12: buf[i] = i.float32
    var t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    for i in 0 ..< 12:
      doAssert t[i] == i.float32

  block:  # Flat write through tensor, verify through t
    var buf = newSeq[float32](6)
    var t = make_tensor(buf, 0, make_layout((2, 3), (1, 2)))
    for i in 0 ..< 6:
      t[i] = (i * 10).float32
    for i in 0 ..< 6:
      doAssert t[i] == (i * 10).float32

  block:  # Flat index with non-zero offset
    var buf = newSeq[float32](20)
    for i in 0 ..< 20: buf[i] = i.float32
    let t = make_tensor(buf, 10, make_layout((2, 3), (1, 2)))
    doAssert t[0] == 10.0'f32
    doAssert t[5] == 15.0'f32

  block:  # Flat indexing on TensorView
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    for i in 0 ..< 12:
      doAssert v[i] == i.float32

  block:  # Flat write via TensorView modifies original array
    var buf: array[6, float32]
    let p = addr(buf[0])
    let v = make_view(p, make_layout((2, 3), (1, 2)))
    for i in 0 ..< 6:
      v[i] = (i * 10).float32
    for i in 0 ..< 6:
      doAssert buf[i] == (i * 10).float32
    for i in 0 ..< 6:
      doAssert v[i] == (i * 10).float32

# ═══════════════════════════════════════════════════════════════
#  Multi-index indexing — operator()
# ═══════════════════════════════════════════════════════════════

proc runTensorMultiIndexTests* =
  block:  # Col-major Tensor: (i,j) -> data[i + j*M]
    var buf = newSeq[float32](12)
    var t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    for idx in 0 ..< 12:
      t[idx] = idx.float32
    doAssert t((0, 0)) == 0.0'f32
    doAssert t((2, 0)) == 2.0'f32
    doAssert t((0, 1)) == 3.0'f32
    doAssert t((1, 2)) == 7.0'f32
    doAssert t((2, 3)) == 11.0'f32

  block:  # 2-arg: t(i, j)
    var buf = newSeq[float32](12)
    var t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    for idx in 0 ..< 12:
      t[idx] = idx.float32
    doAssert t(0, 0) == 0.0'f32
    doAssert t(1, 2) == 7.0'f32

  block:  # Col-major TensorView
    var buf: array[12, float32]
    for idx in 0 ..< 12: buf[idx] = idx.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    doAssert v((0, 0)) == 0.0'f32
    doAssert v((0, 1)) == 3.0'f32
    doAssert v((1, 2)) == 7.0'f32

  block:  # Row-major TensorView
    var buf: array[12, float32]
    for idx in 0 ..< 12: buf[idx] = idx.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (4, 1)))
    doAssert v((1, 0)) == 4.0'f32
    doAssert v((0, 1)) == 1.0'f32

  block:  # Write via TensorView modifies original
    var buf: array[6, float32]
    let p = addr(buf[0])
    let v = make_view(p, make_layout((2, 3), (1, 2)))
    v((0, 0)) = 1.0'f32
    v((1, 1)) = 5.0'f32
    v((0, 2)) = 9.0'f32
    doAssert buf[0] == 1.0'f32
    doAssert buf[3] == 5.0'f32
    doAssert v((0, 0)) == 1.0'f32
    doAssert v((1, 1)) == 5.0'f32
    doAssert v((0, 2)) == 9.0'f32

  block:  # Dynamic col-major
    let M = 5; let N = 4
    var buf: array[20, float32]
    for idx in 0 ..< (M*N): buf[idx] = idx.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((M, N), (1, M)))
    doAssert v((0, 0)) == 0.0'f32
    doAssert v((4, 0)) == 4.0'f32
    doAssert v((0, 1)) == 5.0'f32
    doAssert v((3, 2)) == 13.0'f32

  block:  # Write via owning tensor
    var t = make_tensor(make_layout((2, 3), (1, 2)), float32)
    t(0, 0) = 10.0'f32
    t(1, 2) = 20.0'f32
    doAssert t(0, 0) == 10.0'f32
    doAssert t(1, 2) == 20.0'f32

# ═══════════════════════════════════════════════════════════════
#  Tensor slicing with Joker (`_`)
# ═══════════════════════════════════════════════════════════════

proc runTensorSliceTests* =
  block:  # Slice row: t.slice((row, _)) returns 1D Tensor
    var buf = newSeq[float32](12)
    var t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    for i in 0 ..< 12: t[i] = i.float32
    let row1 = t.slice((1, _))
    doAssert row1.rank == 1
    doAssert row1.size == 4
    doAssert row1(0) == 1.0'f32
    doAssert row1(3) == 10.0'f32

  block:  # Slice column: t.slice((_, col)) returns 1D Tensor
    var buf = newSeq[float32](12)
    var t = make_tensor(buf, 0, make_layout((3, 4), (1, 3)))
    for i in 0 ..< 12: t[i] = i.float32
    let col1 = t.slice((_, 1))
    doAssert col1.rank == 1
    doAssert col1.size == 3
    doAssert col1(0) == 3.0'f32
    doAssert col1(1) == 4.0'f32
    doAssert col1(2) == 5.0'f32

  block:  # TensorView slice: v.slice((row, _))
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    let row = v.slice((1, _))
    doAssert row.rank == 1
    doAssert row.size == 4
    doAssert row(0) == 1.0'f32
    doAssert row(3) == 10.0'f32

  block:  # TensorView slice column
    var buf: array[12, float32]
    for i in 0 ..< 12: buf[i] = i.float32
    let p = addr(buf[0])
    let v = make_view(p, make_layout((3, 4), (1, 3)))
    let col = v.slice((_, 2))
    doAssert col(0) == 6.0'f32
    doAssert col(2) == 8.0'f32

  block:  # Slice from owning tensor
    var t = make_tensor(make_layout((3, 4), (1, 3)), float32)
    for i in 0 ..< 12: t[i] = i.float32
    let col = t.slice((_, 2))
    doAssert col(0) == 6.0'f32
    doAssert col(2) == 8.0'f32

# ═══════════════════════════════════════════════════════════════
#  Tensor of different element types
# ═══════════════════════════════════════════════════════════════

proc runTensorTypeTests* =
  block:  # int32 tensor
    var buf: array[8, int32]
    for i in 0 ..< 8: buf[i] = i.int32 * 10
    let p = addr(buf[0])
    let v = make_view(p, make_layout((2, 4), (1, 2)))
    doAssert v((0, 0)) == 0
    doAssert v((1, 1)) == 30

  block:  # float64 tensor
    var buf: array[6, float64]
    buf[0] = 3.14
    let p = addr(buf[0])
    let v = make_view(p, make_layout((2, 3), (1, 2)))
    doAssert v((0, 0)) == 3.14

  block:  # int64 tensor
    var buf: array[4, int64]
    buf[2] = 42
    let p = addr(buf[0])
    let v = make_view(p, make_layout((2, 2), (1, 2)))
    doAssert v((0, 1)) == 42

# ═══════════════════════════════════════════════════════════════
#  Test runner
# ═══════════════════════════════════════════════════════════════

proc runTests* =
  runTensorConstructionTests()
  runTensorViewTests()
  runTensorFlatIndexTests()
  runTensorMultiIndexTests()
  runTensorSliceTests()
  runTensorTypeTests()

when isMainModule:
  runTests()
  echo "OK: all tensor tests passed"
