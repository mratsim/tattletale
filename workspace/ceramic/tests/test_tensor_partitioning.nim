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
#     rank<0>=2 rank<1>=3 → tiled(0, _, _, _)
#     → shape (2,2,2):(4,16,32)
#     NOTE: compiles and passes individually; triggers a pre-existing
#     `zip2_by`/`concat` C++ backend codegen bug in Nim 2.2.10 when
#     combined with other tests in one file.
# ═════════════════════════════════════════════════════════════════════════

proc runOuter3dScalarCoord(errors: var int) =
  echo "--- 4. outer 3D scalar (8,4,2) tiler (4,2) coord=0 ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,4,2)))
  for i in 0..<64: t(i) = float32(i)
  let v = t.view()
  let p = outer_partition(v, (4,2), 0)
  doAssert p(0,0,0) == 0.0'f32, "(0,0,0)==0"
  doAssert p(1,0,0) == 4.0'f32, "(1,0,0)==4"
  doAssert p(0,1,0) == 16.0'f32, "(0,1,0)==16"
  doAssert p(0,0,1) == 32.0'f32, "(0,0,1)==32"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  5. inner_partition — 3D, scalar coord
#     CuTe: data (8,4,2), tiler (4,2), coord=0
#     → tile part (4,2):(1,8), size=8
#     NOTE: same zip2_by backend bug as test 4.
# ═════════════════════════════════════════════════════════════════════════

proc runInner3dScalarCoord(errors: var int) =
  echo "--- 5. inner 3D scalar (8,4,2) tiler (4,2) coord=0 ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,4,2)))
  for i in 0..<64: t(i) = float32(i)
  let v = t.view()
  let p = inner_partition(v, (4,2), 0)
  doAssert p.size === 8, "size=8"
  doAssert p(0,0) == 0.0'f32, "(0,0)==0"
  doAssert p(1,0) == 1.0'f32, "(1,0)==1"
  doAssert p(0,1) == 8.0'f32, "(0,1)==8"
  echo "  OK"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  6. inner_partition — 2D, _ (X) in coord
#     CuTe: data (8,8), tiler (4,4), coord (1, _)
#     rest mode 0 fixed to 1 (offset=1*4=4), rest mode 1 kept
#     → shape (4,4,2):(1,8,32)
# ═════════════════════════════════════════════════════════════════════════

proc runInner2dUnderscore(errors: var int) =
  echo "--- 6. inner 2D _ coord (8x8) tiler (4,4) coord (1,X) ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,8)))
  for i in 0..<64: t(i) = float32(i)
  let v = t.view()
  let p = inner_partition(v, (4,4), (1, X()))
  # rest m0 fixed→1 (offset=4), rest m1 kept (2)
  doAssert p.layout.shape === (4,4,2), "shape=(4,4,2)"
  doAssert p(0,0,0) == 4.0'f32, "(0,0,0)==4"
  doAssert p(1,0,0) == 5.0'f32, "(1,0,0)==5"
  doAssert p(0,1,0) == 12.0'f32, "(0,1,0)==12"
  doAssert p(0,0,1) == 36.0'f32, "(0,0,1)==36"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  7. outer_partition — 2D, _ (X) in coord
#     CuTe: data (8,8), tiler (4,4), coord (1, _)
#     tile mode 0 fixed to 1, tile mode 1 kept free, rest kept
#     → shape (4,2,2):(8,4,32), offset=1
# ═════════════════════════════════════════════════════════════════════════

proc runOuter2dUnderscore(errors: var int) =
  echo "--- 7. outer 2D _ coord (8x8) tiler (4,4) coord (1,X) ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,8)))
  for i in 0..<64: t(i) = float32(i)
  let v = t.view()
  let p = outer_partition(v, (4,4), (1, X()))
  # tile m0 fixed→1 (offset=1), tile m1 kept (4), rest kept (2,2)
  doAssert p.layout.shape === (4,2,2), "shape=(4,2,2)"
  doAssert p(0,0,0) == 1.0'f32, "(0,0,0)==1"
  doAssert p(1,0,0) == 9.0'f32, "(1,0,0)==9"
  doAssert p(0,1,0) == 5.0'f32, "(0,1,0)==5"
  doAssert p(0,0,1) == 33.0'f32, "(0,0,1)==33"
  echo "  OK"

# ═════════════════════════════════════════════════════════════════════════
#  8. inner_partition — 2D, _ (X) in coord (3D result, offset=0)
#     CuTe: data (8,8), tiler (4,4), coord (0, _)
#     rest mode 0 fixed to 0 (offset=0), rest mode 1 kept
#     → shape (4,4,2):(1,8,32)
# ═════════════════════════════════════════════════════════════════════════

proc runInner2dUnderscore3d(errors: var int) =
  echo "--- 8. inner 2D _ coord (3D result) (8x8) tiler (4,4) coord (0,X) ---"
  var buf = newSeq[float32](64)
  var t = make_tensor(buf, 0, make_layout((8,8)))
  for i in 0..<64: t(i) = float32(i)
  let v = t.view()
  let p = inner_partition(v, (4,4), (0, X()))
  # rest m0 fixed→0 (offset=0), rest m1 kept (2)
  doAssert p.layout.shape === (4,4,2), "shape=(4,4,2)"
  doAssert p(0,0,0) == 0.0'f32, "(0,0,0)==0"
  doAssert p(0,1,0) == 8.0'f32, "(0,1,0)==8"
  doAssert p(0,0,1) == 32.0'f32, "(0,0,1)==32"
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
