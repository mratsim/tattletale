# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## atAxisIndex and alongAxisAt: the two axis views
##
## Test that atAxisIndex fixes one axis and keeps the others,
## that alongAxisAt keeps one axis and fixes the others at the flat position,
## and that the two views coincide in 2D, one the dual of the other.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests workspace/ceramic/tests/tensor/t_tensor_axis_views.nim

import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic

proc runTest =
  block: # atAxisIndex, 2D: axis 1 fixed at 2 keeps the rows
    var buf = newSeq[float32](8 * 4)
    var t = make_view(buf +% 0, make_layout((8, 4)))
    for i in 0 ..< 8:
      for j in 0 ..< 4:
        t(i, j) = float32(10 * i + j)
    let v = atAxisIndex(t, axis = 1, idx = 2)
    doAssert v.shape === (8,), "atAxisIndex drops the fixed axis, shape (8,)"
    for i in 0 ..< 8:
      doAssert v(i) == t(i, 2), "v(" & $i & ") aliases t(" & $i & ", 2)"

  block: # atAxisIndex, 3D: the fixed axis leaves the shape, the rest stays
    var buf = newSeq[float32](2 * 3 * 4)
    var t = make_view(buf +% 0, make_layout((2, 3, 4)))
    for i in 0 ..< 2:
      for j in 0 ..< 3:
        for k in 0 ..< 4:
          t(i, j, k) = float32(100 * i + 10 * j + k)
    let v = atAxisIndex(t, axis = 0, idx = 1)
    doAssert v.shape === (3, 4), "atAxisIndex drops axis 0, shape (3, 4)"
    for j in 0 ..< 3:
      for k in 0 ..< 4:
        doAssert v(j, k) == t(1, j, k), "v(" & $j & ", " & $k & ") aliases t(1, " & $j & ", " & $k & ")"

  block: # atAxisIndex, rank 1: the only axis fixes away, the view is the full tensor
    var buf = newSeq[float32](6)
    var t = make_view(buf +% 0, make_layout((6,)))
    for i in 0 ..< 6:
      t(i) = float32(i)
    let v = atAxisIndex(t, axis = 0, idx = 3)
    doAssert v.shape === (6,), "rank-1 atAxisIndex keeps the full shape"
    for i in 0 ..< 6:
      doAssert v(i) == t(i), "rank-1 atAxisIndex aliases the whole tensor"

  block: # 2D duality: fixing the complement axis is the alongAxisAt strip
    var buf = newSeq[float32](8 * 4)
    var t = make_view(buf +% 0, make_layout((8, 4)))
    for i in 0 ..< 8:
      for j in 0 ..< 4:
        t(i, j) = float32(10 * i + j)
    for j in 0 ..< 4:
      let strip = alongAxisAt(t, axis = 0, idx = j)
      let fixed = atAxisIndex(t, axis = 1, idx = j)
      doAssert strip.shape === fixed.shape, "both views drop to (8,)"
      for i in 0 ..< 8:
        doAssert strip(i) == fixed(i), "alongAxisAt(0, " & $j & ") == atAxisIndex(1, " & $j & ")"

  echo "t_tensor_axis_views: all checks passed"

when isMainModule:
  runTest()
