## Faithful port of CuTe's inner_partition / outer_partition tests.
##
## CuTe reference: _references_kernels/cutlass/test_inner_outer.cpp
##
## Tests the EXISTING ceramic inner_partition / outer_partition templates
## from tensors.nim. No local implementations — only ceramic library calls.

import std/macros
import workspace/ceramic/src/int_tuples {.all.}
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

# ═════════════════════════════════════════════════════════════════════════
#  1. outer_partition — 2D, tuple coord
#     CuTe: data (8,8):(1,8), tiler (4,4), coord (1,0)
#     → shape (2,2):(4,32), elements data[1+m*4, 0+n*4]
# ═════════════════════════════════════════════════════════════════════════

proc runOuter2dTupleCoord(errors: var int) =
  echo "--- 1. outer 2D tuple coord (8x8) tiler (4,4) coord (1,0) ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,8)))
  for i in 0..<64: t(i) = float32(i)
  let v = t.view()
  let p = outer_partition(v, (4,4), (1,0))
  doAssert p.layout.shape === (2,2), "shape=(2,2)"
  doAssert p(0,0) == 1.0'f32, "(0,0)==1"
  doAssert p(1,0) == 5.0'f32, "(1,0)==5"
  doAssert p(0,1) == 33.0'f32, "(0,1)==33"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  2. outer_partition — 2D, scalar coord
#     CuTe: data (8,8), tiler (4,4), coord=5
#     flat idx 5 in (4,4):(1,4) → (1,1) → data[1+m*4, 1+n*4]
#     → shape (2,2):(4,32), elements 9, 13
# ═════════════════════════════════════════════════════════════════════════

proc runOuter2dScalarCoord(errors: var int) =
  echo "--- 2. outer 2D scalar coord (8x8) tiler (4,4) coord=5 ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,8)))
  for i in 0..<64: t(i) = float32(i)
  let v = t.view()
  let p = outer_partition(v, (4,4), 5)
  doAssert p.layout.shape === (2,2), "shape=(2,2)"
  doAssert p(0,0) == 9.0'f32, "(0,0)==9"
  doAssert p(1,0) == 13.0'f32, "(1,0)==13"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  3. inner_partition — 2D, tuple coord
#     CuTe: data (8,8), tiler (4,4), coord (1,0)
#     → shape (4,4):(1,8), elements data[4+m, 0+n]
# ═════════════════════════════════════════════════════════════════════════

proc runInner2dTupleCoord(errors: var int) =
  echo "--- 3. inner 2D tuple coord (8x8) tiler (4,4) coord (1,0) ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,8)))
  for i in 0..<64: t(i) = float32(i)
  let v = t.view()
  let p = inner_partition(v, (4,4), (1,0))
  doAssert p.layout.shape === (4,4), "shape=(4,4)"
  doAssert p(0,0) == 4.0'f32, "(0,0)==4"
  doAssert p(1,0) == 5.0'f32, "(1,0)==5"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  4. outer_partition — 3D, scalar coord
#     CuTe: data (8,4,2):(1,8,32), tiler (4,2), coord=0
#     ceramic: data (8,4,3):(1,8,32), tiler (4,2), coord=0
#     (Int[3] instead of Int[2] avoids C++ backend hash collision)
#     rank<0>=2 rank<1>=3 → tiled(0, _, _, _)
#     → shape (2,2,3):(4,16,32)
# ═════════════════════════════════════════════════════════════════════════

proc runOuter3dScalarCoord(errors: var int) =
  echo "--- 4. outer 3D scalar (8,4,3) tiler (4,2) coord=0 ---"
  var buf = newSeq[float32](96)
  var t = make_tensor(buf, 0, make_layout((8,4,3)))
  for i in 0..<96: t(i) = float32(i)
  let v = t.view()
  let p = outer_partition(v, (4,2), 0)
  # rest group (2,2,3):(4,16,32), all modes at coord 0
  doAssert p.layout.shape === (2,2,3), "shape=(2,2,3)"
  doAssert p(0,0,0) == 0.0'f32, "(0,0,0)==0"
  doAssert p(1,0,0) == 4.0'f32, "(1,0,0)==4"
  doAssert p(0,1,0) == 16.0'f32, "(0,1,0)==16"
  doAssert p(0,0,1) == 32.0'f32, "(0,0,1)==32"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  5. inner_partition — 3D, scalar coord
#     CuTe: data (8,4,2), tiler (4,2), coord=0
#     ceramic: data (8,4,3), tiler (4,2), coord=0
#     (Int[3] instead of Int[2] avoids C++ backend hash collision)
#     → tile part (4,2):(1,8), size=8
# ═════════════════════════════════════════════════════════════════════════

proc runInner3dScalarCoord(errors: var int) =
  echo "--- 5. inner 3D scalar (8,4,3) tiler (4,2) coord=0 ---"
  var buf = newSeq[float32](96)
  var t = make_tensor(buf, 0, make_layout((8,4,3)))
  for i in 0..<96: t(i) = float32(i)
  let v = t.view()
  let p = inner_partition(v, (4,2), 0)
  # tile part (4,2):(1,8)
  doAssert p.size === 8, "size=8"
  doAssert p(0,0) == 0.0'f32, "(0,0)==0"
  doAssert p(1,0) == 1.0'f32, "(1,0)==1"
  doAssert p(0,1) == 8.0'f32, "(0,1)==8"
  echo "  OK"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  6. inner_partition — 2D, _ (X) in coord
#     CuTe: data (8,8), tiler (4,4), coord (1, _)
#     ceramic: data (9,9), tiler (3,3), coord (1, X())
#     (Int[9]/Int[3] avoids C++ backend hash collision w/ tests 1-3)
#     rest mode 0 fixed to 1 (offset=1*3=3), rest mode 1 kept
#     → shape (3,3,3):(1,9,27)
# ═════════════════════════════════════════════════════════════════════════

proc runInner2dUnderscore(errors: var int) =
  echo "--- 6. inner 2D _ coord (9x9) tiler (3,3) coord (1,X) ---"
  var buf = newSeq[float32](81)
  var t = make_tensor(buf, 0, make_layout((9,9)))
  for i in 0..<81: t(i) = float32(i)
  let v = t.view()
  let p = inner_partition(v, (3,3), (1, X()))
  # rest m0 fixed→1 (offset=3), rest m1 kept (3)
  doAssert p.layout.shape === (3,3,3), "shape=(3,3,3)"
  doAssert p(0,0,0) == 3.0'f32, "(0,0,0)==3"
  doAssert p(1,0,0) == 4.0'f32, "(1,0,0)==4"
  doAssert p(0,1,0) == 12.0'f32, "(0,1,0)==12"
  doAssert p(0,0,1) == 30.0'f32, "(0,0,1)==30"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  7. outer_partition — 2D, _ (X) in coord
#     CuTe: data (8,8), tiler (4,4), coord (1, _)
#     ceramic: data (9,9), tiler (3,3), coord (1, X())
#     tile mode 0 fixed to 1, tile mode 1 kept free, rest kept
#     → shape (3,3,3):(9,3,27), offset=1
# ═════════════════════════════════════════════════════════════════════════

proc runOuter2dUnderscore(errors: var int) =
  echo "--- 7. outer 2D _ coord (9x9) tiler (3,3) coord (1,X) ---"
  var buf = newSeq[float32](81)
  var t = make_tensor(buf, 0, make_layout((9,9)))
  for i in 0..<81: t(i) = float32(i)
  let v = t.view()
  let p = outer_partition(v, (3,3), (1, X()))
  # tile m0 fixed→1 (offset=1), tile m1 kept (3), rest kept (3,3)
  doAssert p.layout.shape === (3,3,3), "shape=(3,3,3)"
  doAssert p(0,0,0) == 1.0'f32, "(0,0,0)==1"
  doAssert p(1,0,0) == 10.0'f32, "(1,0,0)==10"
  doAssert p(0,1,0) == 4.0'f32, "(0,1,0)==4"
  doAssert p(0,0,1) == 28.0'f32, "(0,0,1)==28"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  8. inner_partition — 2D, _ (X) in coord (3D result, offset=0)
#     CuTe: data (8,8), tiler (4,4), coord (0, _)
#     ceramic: data (9,9), tiler (3,3), coord (0, X())
#     rest mode 0 fixed to 0 (offset=0), rest mode 1 kept
#     → shape (3,3,3):(1,9,27)
# ═════════════════════════════════════════════════════════════════════════

proc runInner2dUnderscore3d(errors: var int) =
  echo "--- 8. inner 2D _ coord (3D result) (9x9) tiler (3,3) coord (0,X) ---"
  var buf = newSeq[float32](81)
  var t = make_tensor(buf, 0, make_layout((9,9)))
  for i in 0..<81: t(i) = float32(i)
  let v = t.view()
  let p = inner_partition(v, (3,3), (0, X()))
  # rest m0 fixed→0 (offset=0), rest m1 kept (3)
  doAssert p.layout.shape === (3,3,3), "shape=(3,3,3)"
  doAssert p(0,0,0) == 0.0'f32, "(0,0,0)==0"
  doAssert p(0,1,0) == 9.0'f32, "(0,1,0)==9"
  doAssert p(0,0,1) == 27.0'f32, "(0,0,1)==27"
  echo "  OK"


# ═════════════════════════════════════════════════════════════════════════
#  9. local_tile 4-arg with projection — sgemm_1 CTA tile extraction
#     CuTe: test_local_partition.cpp t9_local_tile
# ═════════════════════════════════════════════════════════════════════════

proc runLocalTileCtaExtraction(errors: var int) =
  echo "--- 9. local_tile 4-arg CTA tile extraction ---"
  let M = 512; let N = 512; let K = 64
  let bM = 128; let bN = 128; let bK = 8
  let tiler = (bM, bN, bK)
  let coord0 = (Int[0](), Int[0](), _)

  # mA: (M,K) strides (1, M) = (1, 512), proj = (Y, X, Y)
  block:
    var buf = newSeq[float32](M * K)
    var mA = make_tensor(buf, 0, make_layout((M, K)))
    for i in 0..<M*K: mA(i) = float32(i)
    let v = mA.view()
    let gA = local_tile(v, tiler, coord0, (Y, X, Y))
    # dice(proj, tiler)  → (128, 8)
    # dice(proj, coord0) → (0, _)
    # inner_partition(mA, (128,8), (0,_)):
    #   zipped_divide: mode 0 split by 128→(128,4), mode 1 split by 8→(8,8)
    #   slice(0,_): keep tile, fix rest mode 0 to 0
    #   shape: (128, 8, 8)
    doAssert gA.layout.shape === (bM, bK, K div bK), "shape=(" & $bM & "," & $bK & "," & $(K div bK) & ")"
    doAssert gA(0,0,0) == 0.0'f32, "(0,0,0)==0"
    doAssert gA(0,1,0) == 512.0'f32, "(0,1,0)==512"
    echo "  mA (Y,X,Y) OK"

  # mB: (N,K) = (512,64), proj = (X, Y, Y)
  block:
    var buf = newSeq[float32](N * K)
    var mB = make_tensor(buf, 0, make_layout((N, K)))
    for i in 0..<N*K: mB(i) = float32(i)
    let v = mB.view()
    let gB = local_tile(v, tiler, coord0, (X, Y, Y))
    doAssert gB.layout.shape === (bN, bK, K div bK), "shape=(" & $bN & "," & $bK & "," & $(K div bK) & ")"
    doAssert gB(0,0,0) == 0.0'f32, "(0,0,0)==0"
    echo "  mB (X,Y,Y) OK"

  # mC: (M,N) = (512,512), proj = (Y, Y, X)
  block:
    var buf = newSeq[float32](M * N)
    var mC = make_tensor(buf, 0, make_layout((M, N)))
    for i in 0..<M*N: mC(i) = float32(i)
    let v = mC.view()
    let gC = local_tile(v, tiler, coord0, (Y, Y, X))
    # dice: keep modes 0,1 → (128, 128)
    # local_tile(mC, (128,128), (0,0)):
    #   zipped_divide: split mode 0 by 128→(128,4), mode 1 by 128→(128,4)
    #   slice(0,0): keep tile, fix both rest modes
    #   shape: (128, 128)
    doAssert gC.layout.shape === (bM, bN), "shape=(" & $bM & "," & $bN & ")"
    doAssert gC(0,0) == 0.0'f32, "(0,0)==0"
    doAssert gC(1,0) == 1.0'f32, "(1,0)==1"
    doAssert gC(0,1) == 512.0'f32, "(0,1)==512"
    echo "  mC (Y,Y,X) OK"

  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  local_partition — 3-arg & 4-arg
#  CuTe: test_local_partition.cpp t1-t8
# ═════════════════════════════════════════════════════════════════════════

proc runLocalPartition3Arg2d(errors: var int) =
  echo "--- 10. local_partition 3-arg 2D (8x8) thrLayout (4x4) ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,8)))
  for i in 0..<64: t(i) = float32(i)
  let L = make_layout((4,4))
  let v = t.view()
  block:
    let idx = 0; let p = local_partition(v, L, idx)
    doAssert p(0,0) == 0.0'f32
    doAssert p(1,0) == 4.0'f32
    doAssert p(0,1) == 32.0'f32
    doAssert p(1,1) == 36.0'f32
  block:
    let idx = 1; let p = local_partition(v, L, idx)
    doAssert p(0,0) == 1.0'f32
    doAssert p(1,0) == 5.0'f32
  block:
    let idx = 4; let p = local_partition(v, L, idx)
    doAssert p(0,0) == 8.0'f32
    doAssert p(1,0) == 12.0'f32
  block:
    let idx = 15; let p = local_partition(v, L, idx)
    doAssert p(0,0) == 27.0'f32
    doAssert p(1,1) == 63.0'f32
  echo "  OK"

proc runLocalPartition3Arg1d(errors: var int) =
  echo "--- 11. local_partition 3-arg 1D (32,) thrLayout (16,) ---"
  var buf = newSeq[float32](32)
  var t = make_tensor(buf, 0, make_layout((32,)))
  for i in 0..<32: t(i) = float32(i)
  let L = make_layout((16,))
  let v = t.view()
  block:
    let p = local_partition(v, L, 0)
    doAssert p(0) == 0.0'f32
    doAssert p(1) == 16.0'f32
  block:
    let p = local_partition(v, L, 15)
    doAssert p(0) == 15.0'f32
    doAssert p(1) == 31.0'f32
  echo "  OK"

proc runLocalPartition3Arg3d(errors: var int) =
  echo "--- 12. local_partition 3-arg 3D (8x4x2) thrLayout (4x2) ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,4,2)))
  for i in 0..<64: t(i) = float32(i)
  let L = make_layout((4,2))
  let v = t.view()
  let p = local_partition(v, L, 0)
  doAssert p.layout.shape === (2,2,2)
  doAssert p(0,0,0) == 0.0'f32
  doAssert p(1,0,0) == 4.0'f32
  doAssert p(0,1,0) == 16.0'f32
  doAssert p(0,0,1) == 32.0'f32
  echo "  OK"

proc runLocalPartition3ArgLarge(errors: var int) =
  echo "--- 13. local_partition 3-arg large (32x32) thrLayout (16x16) ---"
  var buf = newSeq[float32](1024)
  var t = make_tensor(buf, 0, make_layout((32,32)))
  for i in 0..<1024: t(i) = float32(i)
  let L = make_layout((16,16))
  let v = t.view()
  let p = local_partition(v, L, 0)
  doAssert p.layout.shape === (2,2)
  doAssert p(0,0) == 0.0'f32
  doAssert p(1,0) == 16.0'f32
  doAssert p(0,1) == 512.0'f32
  echo "  OK"

proc runLocalPartition3ArgConst(errors: var int) =
  echo "--- 14. local_partition 3-arg const indirection ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,8)))
  for i in 0..<64: t(i) = float32(i)
  let L = make_layout((4,4))
  let v = t.view()
  block:
    doAssert local_partition(v, L, 0)(0,0) == 0.0'f32
    doAssert local_partition(v, L, 1)(0,0) == 1.0'f32
    doAssert local_partition(v, L, 4)(0,0) == 8.0'f32
    doAssert local_partition(v, L, 15)(1,1) == 63.0'f32
  echo "  OK"

proc runLocalPartition4ArgStep1X(errors: var int) =
  echo "--- 15. local_partition 4-arg Step<_1,X> (128x8) tC(16,16) ---"
  var buf = newSeq[float32](1024)
  var t = make_tensor(buf, 0, make_layout((128,8)))
  for i in 0..<1024: t(i) = float32(i)
  let tC = make_layout((16,16))
  let v = t.view()
  block:
    let p = local_partition(v, tC, 0, (Y, X))
    doAssert p.layout.shape === (8,8)
    doAssert p(0,0) == 0.0'f32
    doAssert p(1,0) == 16.0'f32
    doAssert p(0,1) == 128.0'f32
  block:
    let p = local_partition(v, tC, 15, (Y, X))
    doAssert p(0,0) == 15.0'f32
  echo "  OK"

proc runLocalPartition4ArgStepX1(errors: var int) =
  echo "--- 16. local_partition 4-arg Step<X,_1> (128x8) tC(16,16) ---"
  var buf = newSeq[float32](1024)
  var t = make_tensor(buf, 0, make_layout((128,8)))
  for i in 0..<1024: t(i) = float32(i)
  let tC = make_layout((16,16))
  let v = t.view()
  block:
    doAssert local_partition(v, tC, 0,   (X, Y))(0,0) == 0.0'f32
    doAssert local_partition(v, tC, 15,  (X, Y))(0,0) == 0.0'f32
    doAssert local_partition(v, tC, 16,  (X, Y))(0,0) == 1.0'f32
    doAssert local_partition(v, tC, 112, (X, Y))(0,0) == 7.0'f32
  echo "  OK"

proc runLocalPartition4ArgStep11(errors: var int) =
  echo "--- 17. local_partition 4-arg Step<_1,_1> (128x128) tC(16,16) ---"
  var buf = newSeq[float32](16384)
  var t = make_tensor(buf, 0, make_layout((128,128)))
  for i in 0..<16384: t(i) = float32(i)
  let tC = make_layout((16,16))
  let v = t.view()
  block:
    let p = local_partition(v, tC, 0,   (Y, Y))
    doAssert p.layout.shape === (8,8)
    doAssert p(0,0) == 0.0'f32
  block:
    let p = local_partition(v, tC, 255, (Y, Y))
    doAssert p(0,0) == (15.0 + 15.0*128.0).float32
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═════════════════════════════════════════════════════════════════════════

proc runTests =
  var errors = 0
  runOuter2dTupleCoord(errors)
  runOuter2dScalarCoord(errors)
  runInner2dTupleCoord(errors)
  runOuter3dScalarCoord(errors)
  runInner3dScalarCoord(errors)
  runInner2dUnderscore(errors)
  runOuter2dUnderscore(errors)
  runInner2dUnderscore3d(errors)
  runLocalTileCtaExtraction(errors)
  runLocalPartition3Arg2d(errors)
  runLocalPartition3Arg1d(errors)
  runLocalPartition3Arg3d(errors)
  runLocalPartition3ArgLarge(errors)
  runLocalPartition3ArgConst(errors)
  runLocalPartition4ArgStep1X(errors)
  runLocalPartition4ArgStepX1(errors)
  runLocalPartition4ArgStep11(errors)
  echo ""
  echo "=== SUMMARY ==="
  if errors == 0:
    echo "All tests passed"
  else:
    echo errors, " FAILED"

when isMainModule:
  runTests()
