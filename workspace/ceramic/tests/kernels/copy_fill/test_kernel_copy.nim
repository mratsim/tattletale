## Test:
##   kernel_copy, copySameShape_cpu, copyPermuted_cpu, copyFrom (GPU)
##
## Run:
##   nim c -r -d:metal workspace/ceramic/tests/kernels/copy_fill/test_kernel_copy.nim
##
## Tests both CPU and GPU copy paths with static and dynamic layouts.

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernels/k_layout_copy_cpu
import workspace/ceramic/src/kernels/k_layout_copy_gpu
import workspace/ceramic/src/hardware/h_copy_dispatch

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
  # src shape=(4,), stride=(2,), every other element
  let srcTV = make_view(src, make_layout(4, 2))
  var dstTV = make_view(dst, make_layout(4, 2))
  copySameShape_cpu(dstTV, srcTV)
  # With stride 2, logical positions are at 0, 2, 4, 6
  doAssert dst[0] == 0.0'f32  # logical coord 0 = src[0]
  doAssert dst[2] == 2.0'f32  # logical coord 1 = src[2]
  doAssert dst[4] == 4.0'f32  # logical coord 2 = src[4]
  doAssert dst[6] == 6.0'f32  # logical coord 3 = src[6]
  doAssert dst[1] == 0.0'f32  # gap at index 1 untouched
  doAssert dst[3] == 0.0'f32  # gap at index 3 untouched

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
  # shape (1, 8), first dim size-1
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
  # Verify:
  #   dst[n,c,h,w] == src[c,n,h,w]
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
  # shape (3, 4) stride (5, 1), strided src
  let srcTV = make_view(src, make_layout((3, 4), (5, 1)))
  var dstTV = make_view(dst, make_layout((3, 4), (5, 1)))
  copyFrom(dstTV, srcTV)
  for i in 0 ..< 3:
    for j in 0 ..< 4:
      doAssert dst[i * 5 + j] == float32(i * 5 + j)

# ═══════════════════════════════════════════════════════════════
#  copyFrom (GPU path) — coalesced common layout span decomposition
# ═══════════════════════════════════════════════════════════════

template fill(src: var seq[float32]; n: int) =
  for i in 0 ..< n:
    src[i] = float32(i)

template checkSpanCopy(dst, src: var seq[float32]; dstL, srcL: untyped;
                       dstLen, srcLen: int) =
  fill(src, srcLen)
  for i in 0 ..< dstLen:
    dst[i] = -1.0'f32
  var dstTV = make_view(dst, dstL)
  let srcTV = make_view(src, srcL)
  copyFrom(dstTV, srcTV)

proc testCopyFromKvWrite() =
  ## KV write, row-padded dst, row-compact src
  var dst = newSeq[float32](8448)
  var src = newSeq[float32](8192)
  let dstL = make_layout((1, 1, 64, 128), (1, 1, 132, 1))
  let srcL = make_layout((1, 1, 64, 128), (1, 1, 128, 1))
  checkSpanCopy(dst, src, dstL, srcL, 8448, 8192)
  for s in 0 ..< 64:
    for d in 0 ..< 128:
      doAssert dst[s * 132 + d] == float32(s * 128 + d), "KV write"
  doAssert dst[128] == -1.0'f32 and dst[131] == -1.0'f32, "KV padding untouched"
  doAssert dst[8447] == -1.0'f32, "KV trailing padding untouched"

proc testCopyFromKvOddPaddings() =
  ## KV write, odd row paddings (130, 131) exercise 8B chunks and scalar spans
  for rowLen in [130, 131]:
    var dst = newSeq[float32](rowLen * 64)
    var src = newSeq[float32](8192)
    let dstL = make_layout((1, 1, 64, 128), (1, 1, rowLen, 1))
    let srcL = make_layout((1, 1, 64, 128), (1, 1, 128, 1))
    checkSpanCopy(dst, src, dstL, srcL, rowLen * 64, 8192)
    for s in 0 ..< 64:
      for d in 0 ..< 128:
        doAssert dst[s * rowLen + d] == float32(s * 128 + d), "KV write rowLen " & $rowLen
    doAssert dst[63 * rowLen + 127] == float32(8191), "KV write last element"

proc testCopyFromGemmTile() =
  ## GEMM gmem tile (32,16):(1,64) <- (32,16):(1,32)
  var dst = newSeq[float32](992)
  var src = newSeq[float32](512)
  let dstL = make_layout((32, 16), (1, 64))
  let srcL = make_layout((32, 16), (1, 32))
  checkSpanCopy(dst, src, dstL, srcL, 992, 512)
  for y in 0 ..< 16:
    for x in 0 ..< 32:
      doAssert dst[x + 64 * y] == float32(x + 32 * y), "GEMM gmem tile"
  doAssert dst[32] == -1.0'f32 and dst[63] == -1.0'f32, "GEMM tile gaps untouched"

proc testCopyFromCompact() =
  ## fully compact static (32,16):(1,32)
  var dst = newSeq[float32](512)
  var src = newSeq[float32](512)
  let L = make_layout((32, 16), (1, 32))
  checkSpanCopy(dst, src, L, L, 512, 512)
  for i in 0 ..< 512:
    doAssert dst[i] == float32(i), "compact copy"

proc testCopyFromLayoutRight() =
  ## LayoutRight pair (32,16):(16,1)
  var dst = newSeq[float32](512)
  var src = newSeq[float32](512)
  let L = make_layout((32, 16), LayoutRight)
  checkSpanCopy(dst, src, L, L, 512, 512)
  for i in 0 ..< 512:
    doAssert dst[i] == float32(i), "LayoutRight copy"

proc testCopyFromThreeSpanDims() =
  ## three span dimensions, (4,8,8):(1,40,400) <- compact col-major src
  var dst = newSeq[float32](3600)
  var src = newSeq[float32](256)
  let dstL = make_layout((4, 8, 8), (1, 40, 400))
  let srcL = make_layout((4, 8, 8), (1, 4, 32))
  checkSpanCopy(dst, src, dstL, srcL, 3600, 256)
  for j2 in 0 ..< 8:
    for j1 in 0 ..< 8:
      for w in 0 ..< 4:
        doAssert dst[40 * j1 + 400 * j2 + w] ==
          float32(4 * j1 + 32 * j2 + w), "three span dimensions"

proc testCopyFromDynamicSelf() =
  ## dynamic self-copy, runtime span path at N == 1, elementwise at N > 1
  for N in [1, 3]:
    let sh = (Int[32](), Int[8](), N)
    let st = (Int[1](), Int[32](), N)
    var dst = newSeq[float32](32 * 8 * N)
    var src = newSeq[float32](32 * 8 * N)
    fill(src, 32 * 8 * N)
    var dstTV = make_view(dst, make_layout(sh, st))
    let srcTV = make_view(src, make_layout(sh, st))
    copyFrom(dstTV, srcTV)
    for z in 0 ..< N:
      for b in 0 ..< 8:
        for a in 0 ..< 32:
          doAssert dst[a + 32 * b + N * z] == src[a + 32 * b + N * z],
            "dynamic self-copy N=" & $N

proc testCopyFromBroadcastRead() =
  ## broadcast read, src stride 0 writes the same value to every dst element
  var dst = newSeq[float32](8)
  var src = newSeq[float32](1)
  src[0] = 7.5'f32
  var dstTV = make_view(dst, make_layout(8, 1))
  let srcTV = make_view(src, make_layout(8, 0))
  copyFrom(dstTV, srcTV)
  for i in 0 ..< 8:
    doAssert dst[i] == 7.5'f32, "broadcast read"

proc testCopyFromOwnedDst() =
  ## TensorOwned destination, copyFrom into an owned tensor
  var src = newSeq[float32](512)
  fill(src, 512)
  let srcTV = make_view(src, make_layout((32, 16), (1, 32)))
  var own = make_tensor(float32, make_layout((32, 16), (1, 32)))
  copyFrom(own, srcTV)
  for i in 0 ..< 512:
    doAssert own.data[i] == float32(i), "TensorOwned destination"

proc testCopyFromGuardRejectsAmbiguousMultiWrite() =
  ## write-disjointness guard, ambiguous multi-write is a compile-time error
  ## - the dst view is a named var binding, an rvalue make_view never
  ##   compiles against copyFrom's var dst and would make the check vacuous
  var buf = newSeq[float32](64)
  let ambiguousDst = make_layout((4, 8), (1, 0))
  let stridedSrc = make_layout((4, 8), (1, 2))
  var dstTV = make_view(buf, ambiguousDst)
  let srcTV = make_view(buf, stridedSrc)
  doAssert not compiles(copyFrom(dstTV, srcTV)),
    "dst stride-0 leaf with shape > 1 and non-zero src stride must not compile"

proc testCopyFromGuardBroadcastPair() =
  ## guard positives, broadcast pair compiles and copies
  var buf = newSeq[float32](32)
  let bL = make_layout((4, 8), (1, 0))
  var dstTV = make_view(buf, bL)
  let srcTV = make_view(buf, bL)
  copyFrom(dstTV, srcTV)

proc testCopyFromGuardDynamicSrcStride() =
  ## guard positive, dynamic src stride with a dst stride-0 leaf compiles (unprovable)
  var buf = newSeq[float32](64)
  let s = 2
  let dynSrc = make_layout((4, 8), (1, s))
  var dstTV = make_view(buf, make_layout((4, 8), (1, 0)))
  let srcTV = make_view(buf, dynSrc)
  copyFrom(dstTV, srcTV)

testCopyFromKvWrite()
testCopyFromKvOddPaddings()
testCopyFromGemmTile()
testCopyFromCompact()
testCopyFromLayoutRight()
testCopyFromThreeSpanDims()
testCopyFromDynamicSelf()
testCopyFromBroadcastRead()
testCopyFromOwnedDst()
testCopyFromGuardRejectsAmbiguousMultiWrite()
testCopyFromGuardBroadcastPair()
testCopyFromGuardDynamicSrcStride()

proc testCopyLegBlockingAtomExecutes() =
  const units = 2
  var dstBuf: array[8, float32]
  var srcBuf: array[8, float32]
  for i in 0 ..< 8:
    srcBuf[i] = float32(i + 1)
  for i in 0 ..< 8:
    dstBuf[i] = 7.5'f32
  var dstChunks = make_view(addr dstBuf[0], make_layout((1, units), (0, 4)))
  let srcChunks = make_view(addr srcBuf[0], make_layout((1, units), (0, 4)))
  var pred: array[units, bool] = [true, false]
  let predChunks = make_view(addr pred[0], make_layout((1, units), (0, 1)))
  copyFromIfAsync(dstChunks, srcChunks, predChunks)
  float32.commit_group()
  float32.wait_group(0)
  for i in 0 ..< 4:
    doAssert dstBuf[i] == float32(i + 1), "the predicated chunk copies"
  for i in 4 ..< 8:
    doAssert dstBuf[i] == 0.0'f32, "the false chunk zero-fills"

testCopyLegBlockingAtomExecutes()

echo "\n--- kernel_copy tests ---"
echo "  All tests passed."
