## Test: kernel_copy — copySameShape_cpu, copyPermuted_cpu, copyFrom (GPU)
##
## Tests both CPU and GPU copy paths with static and dynamic layouts.

import ../src/int_tuples
import ../src/layouts
import ../src/tensors
import ../src/kernel_copy_cpu
import ../src/kernel_copy_gpu

{.experimental: "callOperator".}

template test(label: string; body: untyped) =
  block:
    body
  echo "  [OK] ", label

# ═══════════════════════════════════════════════════════════════
#  Helper: hash check for copy correctness
# ═══════════════════════════════════════════════════════════════

proc xorHash(data: openArray[float32]): uint32 =
  for v in data:
    result = result xor cast[uint32](v)

proc allClose(a, b: openArray[float32]; rtol = 1e-4, atol = 1e-4): bool =
  if a.len != b.len: return false
  for i in 0 ..< a.len:
    if abs(a[i] - b[i]) > atol + rtol * max(abs(a[i]), abs(b[i])):
      return false
  return true

# ═══════════════════════════════════════════════════════════════
#  copySameShape_cpu — contiguous layouts
# ═══════════════════════════════════════════════════════════════

block:
  # 2D contiguous (LayoutRight)
  var src = newSeq[float32](12)
  var dst = newSeq[float32](12)
  for i in 0 ..< 12: src[i] = float32(i)
  let srcTV = make_view(src, make_layout((3, 4), (1, 3)))
  var dstTV = make_view(dst, make_layout((3, 4), (1, 3)))
  copySameShape_cpu(dstTV, srcTV)
  doAssert allClose(src, dst), "copySameShape_cpu 2D contiguous"

block:
  # 2D contiguous (LayoutLeft)
  var src = newSeq[float32](12)
  var dst = newSeq[float32](12)
  for i in 0 ..< 12: src[i] = float32(i)
  let srcTV = make_view(src, make_layout((3, 4), (4, 1)))
  var dstTV = make_view(dst, make_layout((3, 4), (4, 1)))
  copySameShape_cpu(dstTV, srcTV)
  doAssert allClose(src, dst), "copySameShape_cpu 2D LayoutLeft"

# ═══════════════════════════════════════════════════════════════
#  copySameShape_cpu — non-contiguous (strided)
# ═══════════════════════════════════════════════════════════════

block:
  # 1D strided (gap between elements)
  var src = newSeq[float32](20)
  var dst = newSeq[float32](20)
  for i in 0 ..< 20: src[i] = float32(i)
  # src shape=(4,), stride=(2,) — every other element
  let srcTV = make_view(src, make_layout(4, 2))
  var dstTV = make_view(dst, make_layout(4, 2))
  copySameShape_cpu(dstTV, srcTV)
  # With stride 2, logical positions are at 0, 2, 4, 6
  doAssert dst[0] == 0.0'f32  # logical coord 0 = src[0]
  doAssert dst[2] == 2.0'f32  # logical coord 1 = src[2]
  doAssert dst[4] == 4.0'f32  # logical coord 2 = src[4]
  doAssert dst[6] == 6.0'f32  # logical coord 3 = src[6]
  doAssert dst[1] == 0.0'f32  # gap untouched
  doAssert dst[3] == 0.0'f32  # gap untouched

# ═══════════════════════════════════════════════════════════════
#  copySameShape_cpu — with blockSize parameter
# ═══════════════════════════════════════════════════════════════

block:
  var src = newSeq[float32](100)
  var dst = newSeq[float32](100)
  for i in 0 ..< 100: src[i] = float32(i)
  let srcTV = make_view(src, make_layout((10, 10), LayoutRight))
  var dstTV = make_view(dst, make_layout((10, 10), LayoutRight))
  copySameShape_cpu(dstTV, srcTV, blockSize = 4)
  doAssert allClose(src, dst), "copySameShape_cpu with blockSize=4"

# ═══════════════════════════════════════════════════════════════
#  copySameShape_cpu — single element
# ═══════════════════════════════════════════════════════════════

block:
  var src = newSeq[float32](1)
  var dst = newSeq[float32](1)
  src[0] = 42.0'f32
  let srcTV = make_view(src, make_layout(1, 1))
  var dstTV = make_view(dst, make_layout(1, 1))
  copySameShape_cpu(dstTV, srcTV)
  doAssert dst[0] == 42.0'f32

# ═══════════════════════════════════════════════════════════════
#  copySameShape_cpu — rank-1
# ═══════════════════════════════════════════════════════════════

block:
  var src = newSeq[float32](8)
  var dst = newSeq[float32](8)
  for i in 0 ..< 8: src[i] = float32(i * 3)
  let srcTV = make_view(src, make_layout(8, 1))
  var dstTV = make_view(dst, make_layout(8, 1))
  copySameShape_cpu(dstTV, srcTV)
  doAssert allClose(src, dst), "copySameShape_cpu rank-1"

# ═══════════════════════════════════════════════════════════════
#  copySameShape_cpu — dynamic shape (runtime int)
# ═══════════════════════════════════════════════════════════════

block:
  let rows = 4
  let cols = 5
  var src = newSeq[float32](rows * cols)
  var dst = newSeq[float32](rows * cols)
  for i in 0 ..< rows * cols: src[i] = float32(i)
  let srcTV = make_view(src, make_layout((rows, cols), LayoutRight))
  var dstTV = make_view(dst, make_layout((rows, cols), LayoutRight))
  copySameShape_cpu(dstTV, srcTV)
  doAssert allClose(src, dst), "copySameShape_cpu dynamic shape"

# ═══════════════════════════════════════════════════════════════
#  copySameShape_cpu — dynamic stride
# ═══════════════════════════════════════════════════════════════

block:
  let rows = 3
  let cols = 4
  let ld = 10  # leading dimension > cols
  var src = newSeq[float32](ld * rows)
  var dst = newSeq[float32](ld * rows)
  for r in 0 ..< rows:
    for c in 0 ..< cols:
      src[r * ld + c] = float32(r * cols + c)
  let srcTV = make_view(src, make_layout((rows, cols), (ld, 1)))
  var dstTV = make_view(dst, make_layout((rows, cols), (ld, 1)))
  copySameShape_cpu(dstTV, srcTV)
  for r in 0 ..< rows:
    for c in 0 ..< cols:
      doAssert dst[r * ld + c] == float32(r * cols + c)

# ═══════════════════════════════════════════════════════════════
#  copySameShape_cpu — size-1 dims (should be skipped)
# ═══════════════════════════════════════════════════════════════

block:
  var src = newSeq[float32](8)
  var dst = newSeq[float32](8)
  for i in 0 ..< 8: src[i] = float32(i)
  # shape (1, 8) — first dim size-1
  let srcTV = make_view(src, make_layout((1, 8), (8, 1)))
  var dstTV = make_view(dst, make_layout((1, 8), (8, 1)))
  copySameShape_cpu(dstTV, srcTV)
  doAssert allClose(src, dst), "copySameShape_cpu size-1 dim"

# ═══════════════════════════════════════════════════════════════
#  copyPermuted_cpu — NCHW→CNHW
# ═══════════════════════════════════════════════════════════════

block:
  let N = 2; let C = 3; let H = 4; let W = 5
  let total = N * C * H * W
  var src = newSeq[float32](total)
  var dst = newSeq[float32](total)
  for i in 0 ..< total: src[i] = float32(i)
  let srcTV = make_view(src, make_layout((N, C, H, W), LayoutRight))
  var dstTV = make_view(dst, make_layout((C, N, H, W), LayoutRight))
  copyPermuted_cpu(dstTV, srcTV, [1, 0, 2, 3])
  # Verify: dst[n,c,h,w] == src[c,n,h,w]
  for n in 0 ..< N:
    for c in 0 ..< C:
      for h in 0 ..< H:
        for w in 0 ..< W:
          let srcOff = n * C * H * W + c * H * W + h * W + w
          let dstOff = c * N * H * W + n * H * W + h * W + w
          doAssert dst[dstOff] == src[srcOff]

# ═══════════════════════════════════════════════════════════════
#  copyPermuted_cpu — with blockSize
# ═══════════════════════════════════════════════════════════════

block:
  let N = 2; let C = 3; let H = 4; let W = 5
  let total = N * C * H * W
  var src = newSeq[float32](total)
  var dst = newSeq[float32](total)
  for i in 0 ..< total: src[i] = float32(i)
  let srcTV = make_view(src, make_layout((N, C, H, W), LayoutRight))
  var dstTV = make_view(dst, make_layout((C, N, H, W), LayoutRight))
  copyPermuted_cpu(dstTV, srcTV, [1, 0, 2, 3], blockSize = 8)
  for n in 0 ..< N:
    for c in 0 ..< C:
      for h in 0 ..< H:
        for w in 0 ..< W:
          let srcOff = n * C * H * W + c * H * W + h * W + w
          let dstOff = c * N * H * W + n * H * W + h * W + w
          doAssert dst[dstOff] == src[srcOff]

# ═══════════════════════════════════════════════════════════════
#  copyFrom (GPU path) — basic sanity
# ═══════════════════════════════════════════════════════════════

block:
  var src = newSeq[float32](6)
  var dst = newSeq[float32](6)
  for i in 0 ..< 6: src[i] = float32(i * 10)
  let srcTV = make_view(src, make_layout((2, 3), LayoutRight))
  var dstTV = make_view(dst, make_layout((2, 3), LayoutRight))
  copyFrom(dstTV, srcTV)
  doAssert allClose(src, dst), "copyFrom (GPU) basic"

# ═══════════════════════════════════════════════════════════════
#  copyFrom (GPU path) — strided
# ═══════════════════════════════════════════════════════════════

block:
  var src = newSeq[float32](20)
  var dst = newSeq[float32](20)
  for i in 0 ..< 20: src[i] = float32(i)
  # shape (3, 4) stride (5, 1) — strided src
  let srcTV = make_view(src, make_layout((3, 4), (5, 1)))
  var dstTV = make_view(dst, make_layout((3, 4), (5, 1)))
  copyFrom(dstTV, srcTV)
  for i in 0 ..< 3:
    for j in 0 ..< 4:
      doAssert dst[i * 5 + j] == float32(i * 5 + j)

echo "\n--- kernel_copy tests ---"
echo "  All tests passed."
