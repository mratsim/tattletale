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
  echo ""
  echo "=== SUMMARY ==="
  if errors == 0:
    echo "All tests passed"
  else:
    echo errors, " FAILED"

when isMainModule:
  runTests()
