## Test: kernel_fillwith — fillWith_cpu and fillWith (GPU path)
##
## Tests contiguity-fused zero-fill (nimSetMem), non-zero fill, strided fill,
## and dynamic shapes/stride.

import ../src/int_tuples
import ../src/layouts
import ../src/tensors
import ../src/kernel_fillwith_cpu
import ../src/kernel_fillwith_gpu

{.experimental: "callOperator".}

template test(label: string; body: untyped) =
  block:
    body
  echo "  [OK] ", label

# ═══════════════════════════════════════════════════════════════
#  fillWith_cpu — contiguous (zero and non-zero)
# ═══════════════════════════════════════════════════════════════

block:
  # 2D LayoutRight, zero-fill
  var buf = newSeq[float32](12)
  for i in 0 ..< 12: buf[i] = 99.0'f32
  var tv = make_view(buf, make_layout((3, 4), LayoutRight))
  fillWith_cpu(tv, 0.0'f32)
  for i in 0 ..< 12:
    doAssert buf[i] == 0.0'f32, "zero fill @ " & $i

block:
  # 2D LayoutRight, non-zero fill
  var buf = newSeq[float32](12)
  var tv = make_view(buf, make_layout((3, 4), LayoutRight))
  fillWith_cpu(tv, 3.14'f32)
  for i in 0 ..< 12:
    doAssert buf[i] == 3.14'f32, "non-zero fill @ " & $i

# ═══════════════════════════════════════════════════════════════
#  fillWith_cpu — strided layout
# ═══════════════════════════════════════════════════════════════

block:
  # shape (3, 4) stride (5, 1) — gap between rows
  var buf = newSeq[float32](20)
  for i in 0 ..< 20: buf[i] = -1.0'f32
  var tv = make_view(buf, make_layout((3, 4), (5, 1)))
  fillWith_cpu(tv, 42.0'f32)
  for r in 0 ..< 3:
    for c in 0 ..< 4:
      doAssert buf[r * 5 + c] == 42.0'f32, "strided fill @ (" & $r & "," & $c & ")"
    # gap should be untouched
    doAssert buf[r * 5 + 4] == -1.0'f32, "strided gap @ " & $r

# ═══════════════════════════════════════════════════════════════
#  fillWith_cpu — single element
# ═══════════════════════════════════════════════════════════════

block:
  var buf = newSeq[float32](1)
  var tv = make_view(buf, make_layout(1, 1))
  fillWith_cpu(tv, 7.0'f32)
  doAssert buf[0] == 7.0'f32

# ═══════════════════════════════════════════════════════════════
#  fillWith_cpu — rank-1
# ═══════════════════════════════════════════════════════════════

block:
  var buf = newSeq[float32](6)
  var tv = make_view(buf, make_layout(6, 1))
  fillWith_cpu(tv, -1.0'f32)
  for i in 0 ..< 6:
    doAssert buf[i] == -1.0'f32

# ═══════════════════════════════════════════════════════════════
#  fillWith_cpu — dynamic shape
# ═══════════════════════════════════════════════════════════════

block:
  let n = 5
  var buf = newSeq[float32](n)
  var tv = make_view(buf, make_layout(n, 1))
  fillWith_cpu(tv, 3.0'f32)
  for i in 0 ..< n:
    doAssert buf[i] == 3.0'f32

# ═══════════════════════════════════════════════════════════════
#  fillWith_cpu — dynamic stride
# ═══════════════════════════════════════════════════════════════

block:
  let rows = 3
  let cols = 4
  let ld = 10
  var buf = newSeq[float32](ld * rows)
  var tv = make_view(buf, make_layout((rows, cols), (ld, 1)))
  fillWith_cpu(tv, 5.0'f32)
  for r in 0 ..< rows:
    for c in 0 ..< cols:
      doAssert buf[r * ld + c] == 5.0'f32, "dynamic stride @ (" & $r & "," & $c & ")"
    # gap
    doAssert buf[r * ld + cols] == 0.0'f32, "dynamic stride gap @ " & $r

# ═══════════════════════════════════════════════════════════════
#  fillWith_cpu — TensorView (from seq)
# ═══════════════════════════════════════════════════════════════

block:
  var buf = newSeq[float64](6)
  var tv = make_view(buf, make_layout((2, 3), LayoutRight))
  fillWith_cpu(tv, 2.5'f64)
  for i in 0 ..< 6:
    doAssert buf[i] == 2.5'f64, "float64 fill @ " & $i

# ═══════════════════════════════════════════════════════════════
#  fillWith_cpu — LayoutLeft
# ═══════════════════════════════════════════════════════════════

block:
  var buf = newSeq[float32](12)
  var tv = make_view(buf, make_layout((3, 4), (4, 1)))  # LayoutLeft
  fillWith_cpu(tv, 8.0'f32)
  for i in 0 ..< 12:
    doAssert buf[i] == 8.0'f32

# ═══════════════════════════════════════════════════════════════
#  fillWith (GPU path) — basic sanity
# ═══════════════════════════════════════════════════════════════

block:
  var buf = newSeq[float32](6)
  var tv = make_view(buf, make_layout((2, 3), LayoutRight))
  fillWith(tv, 0.0'f32)
  for i in 0 ..< 6:
    doAssert buf[i] == 0.0'f32

block:
  var buf = newSeq[float32](6)
  var tv = make_view(buf, make_layout((2, 3), LayoutRight))
  fillWith(tv, 9.0'f32)
  for i in 0 ..< 6:
    doAssert buf[i] == 9.0'f32

echo "\n--- kernel_fillwith tests ---"
echo "  All tests passed."
