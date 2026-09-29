## Test:
##   kernel_copy autovectorization, copyFrom chunk path vs element loop
##
## Run:
##   nim cpp -r tests/test_kernel_copy_autovec.nim
##
## Host-side checks of byte-equivalence between the auto-vectorized
## copyFrom and an independent flat-index element loop (the reference),
## plus chunk-path selection matching the CuTe formula.

import std/macros
import std/math

import ../src/int_tuples
import ../src/layout_algebra
import ../src/tensors
import ../src/kernels/k_layout_copy_gpu

{.experimental: "callOperator".}


proc runTests() =
  block:
    # Fully static row-major float32 (32,16):(1,32), common run 512 elements,
    # vec_bits = gcd(512*32, gcd(512, 512, 128)) = 128 → 4-element chunks.
    const L = make_layout((32, 16), (1, 32))
    static:
      doAssert max_common_vector(L, L) == 512
      doAssert max_alignment(L) == 512
      # vec_bits per the CuTe auto-vectorizing copy formula
      doAssert gcd(gcd(max_common_vector(L, L) * 32, max_alignment(L)), 128) == 128

    var srcBuf: array[512, float32]
    var dstBuf: array[512, float32]
    var expected: array[512, float32]
    for i in 0 ..< 512:
      srcBuf[i] = float32(i) * 1.5
    for i in 0 ..< 512:
      # flat (i mod 32, i div 32) → offset (i mod 32) + 32*(i div 32) = i
      expected[i] = srcBuf[i]

    let srcTV = make_view(srcBuf, L)
    var dstTV = make_view(dstBuf, L)
    dstTV.copyFrom(srcTV)
    for i in 0 ..< 512:
      doAssert dstBuf[i] == expected[i], "chunk copy mismatch at " & $i

  block:
    # Static row-major dst, static col-major src (32,16):(32,1) → (1,32):
    # common run 1 → element loop even though both layouts are static.
    const
      Lrow = make_layout((32, 16), (1, 32))
      Lcol = make_layout((32, 16), (32, 1))
    static:
      doAssert max_common_vector(Lcol, Lrow) == 1

    var srcBuf: array[1008, float32]  # cosize of (32,16):(32,1)
    var dstBuf: array[512, float32]
    for i in 0 ..< 1008:
      srcBuf[i] = float32(i) * 2.0
    let srcTV = make_view(srcBuf, Lcol)
    var dstTV = make_view(dstBuf, Lrow)
    dstTV.copyFrom(srcTV)
    # flat k of dst (row-major) ↔ src (col-major), dst[(k mod 32, k div 32)]
    # = srcBuf[32*(k mod 32) + k div 32]
    for k in 0 ..< 512:
      doAssert dstBuf[k] == srcBuf[32*(k mod 32) + k div 32],
        "permuted copy mismatch at " & $k

  block:
    # Dynamic stride → element loop, unchanged behavior.
    # Dynamic strides, differing between src and dst (padded rows 8 and 10):
    let rowsS = 8
    let rowsD = 10
    let LS = make_layout((4, 4), (1, rowsS))
    let LD = make_layout((4, 4), (1, rowsD))
    static:
      doAssert not isStaticLayout(LS)

    var srcBuf: array[40, int32]  # cosize of (4,4):(1,8)
    var dstBuf: array[40, int32]  # cosize of (4,4):(1,10) is 34
    for i in 0 ..< 40:
      srcBuf[i] = int32(i)
    let srcTV = make_view(srcBuf, LS)
    var dstTV = make_view(dstBuf, LD)
    dstTV.copyFrom(srcTV)
    # flat k → coord (k mod 4, k div 4) → offset (k mod 4) + rows*(k div 4)
    for k in 0 ..< 16:
      doAssert dstBuf[(k mod 4) + rowsD*(k div 4)] == srcBuf[(k mod 4) + rowsS*(k div 4)],
        "dynamic-stride copy mismatch at " & $k

  block:
    # TensorOwned destination, chunk path through the owning buffer.
    const L = make_layout(64, 1)
    var srcBuf: array[64, int32]
    for i in 0 ..< 64:
      srcBuf[i] = int32(1000 + i)
    var dst = make_tensor(int32, L)
    let srcTV = make_view(srcBuf, L)
    dst.copyFrom(srcTV)
    for i in 0 ..< 64:
      doAssert dst.data[i] == int32(1000 + i), "owned copy mismatch at " & $i

when isMainModule:
  runTests()
  echo "test_kernel_copy_autovec OK"
