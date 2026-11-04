# Test that a range leaf `a ..< b` in the tensor () selector offsets
# its view by `a * stride` and keeps the mode with size `b - a + 1`.
# HSlice stores the inclusive upper bound: the `..<` sugar lowers
# to `a .. pred(b)`. Layout application rejects range leaves.
# Run: nim c -r --hints:off --warnings:off -o:build/wip/tsr --nimcache:nimcache/tsr workspace/ceramic/tests/tensor/t_tensor_slice_ranges.nim

import std/macros
import workspace/ceramic/src/int_tuples {.all.}
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic

# ═══════════════════════════════════════════════════════════════════════
#  1. range first mode: (8,8):(8,1), (2 ..< 5, _) → (3,8):(8,1), origin +16
# ═══════════════════════════════════════════════════════════════════════

proc runRangeFirstMode(errors: var int) =
  echo "--- 1. range first mode (8,8):(8,1), 2 ..< 5 ---"
  var buf = newSeq[float32](64)
  let t = make_view(buf +% 0, make_layout((8,8), (8,1)))
  for i in 0..<64:
    buf[i] = float32(i)
  let r0 = 2
  let v = t(r0 ..< r0 + 3, _)
  doAssert v.shape === (3, 8), "shape=(3,8), got " & $v.shape
  doAssert v.stride === (8, 1), "stride=(8,1), got " & $v.stride
  doAssert v(0, 0) == 16.0'f32, "origin is row 2: v(0,0)==16, got " & $v(0,0)
  doAssert v(2, 7) == 16.0'f32 + 2.0'f32*8.0'f32 + 7.0'f32, "last element"
  echo "  OK"

# ═══════════════════════════════════════════════════════════════════════
#  2. range second mode: (8,8):(8,1), (1, 2 ..< 6) → (4,):(1,), origin 10
# ═══════════════════════════════════════════════════════════════════════

proc runRangeSecondMode(errors: var int) =
  echo "--- 2. range second mode (8,8):(8,1), (1, 2 ..< 6) ---"
  var buf = newSeq[float32](64)
  let t = make_view(buf +% 0, make_layout((8,8), (8,1)))
  for i in 0..<64:
    buf[i] = float32(i)
  let r0 = 2
  let v = t(1, r0 ..< r0 + 4)
  doAssert v.rank == 1, "rank=1, got " & $v.rank
  doAssert v.shape === (4,), "shape=(4,), got " & $v.shape
  doAssert v(0) == 10.0'f32, "v(0)==10, got " & $v(0)
  doAssert v(3) == 13.0'f32, "v(3)==13, got " & $v(3)
  echo "  OK"

# ═══════════════════════════════════════════════════════════════════════
#  3. transpose: (_, 2 ..< 5) → (8,3):(8,1), transpose(0,1) → (3,8):(1,8)
# ═══════════════════════════════════════════════════════════════════════

proc runRangeTranspose(errors: var int) =
  echo "--- 3. range + transpose ---"
  var buf = newSeq[float32](64)
  let t = make_view(buf +% 0, make_layout((8,8), (8,1)))
  for i in 0..<64:
    buf[i] = float32(i)
  let r0 = 2
  let v = t(_, r0 ..< r0 + 3).transpose(0, 1)
  doAssert v.shape === (3, 8), "shape=(3,8), got " & $v.shape
  doAssert v.stride === (1, 8), "stride=(1,8), got " & $v.stride
  doAssert v(0, 0) == 2.0'f32, "v(0,0)==2 (t(0,2)), got " & $v(0,0)
  doAssert v(0, 3) == 26.0'f32, "v(0,3)==26 (t(3,2)), got " & $v(0,3)
  echo "  OK"

# Layout () and tuple slice reject range leaves at compile time.
# Error sites live in layout_indexing and fire on every build
# carrying a range leaf, checked by the compiler at each compile.

when isMainModule:
  var errors = 0
  runRangeFirstMode(errors)
  runRangeSecondMode(errors)
  runRangeTranspose(errors)
  if errors == 0:
    echo "ALL SUBVIEW RANGE TESTS PASS"
  else:
    echo "FAILURES: ", errors
    quit(1)
