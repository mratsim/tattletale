# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## alongAxisAt: the axis strip at a flat position of the other axes
##
## Test that alongAxisAt returns a view of the `axis` strip whose element r
## aliases src at the other axes' flat index `idx` and axis coordinate r,
## over compact and strided 2D/3D layouts. The rank-1 case returns the full tensor.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests workspace/ceramic/tests/tensor/t_tensor_along_axis_at.nim

import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic

proc runTest =
  block: # 2D row-major, axis 0: strip runs down the column
    var buf = newSeq[float32](8 * 4)
    var t = make_view(buf +% 0, make_layout((8, 4)))
    for i in 0 ..< 8:
      for j in 0 ..< 4:
        t(i, j) = float32(10 * i + j)
    let strip = alongAxisAt(t, axis = 0, idx = 3)
    doAssert strip.shape === (8,), "strip shape is (8,)"
    doAssert strip.rank() == 1, "strip rank is 1"
    for r in 0 ..< 8:
      doAssert strip(r) == t(r, 3), "strip(" & $r & ") aliases t(" & $r & ", 3)"

  block: # 2D row-major, axis 1: strip runs along the row
    var buf = newSeq[float32](8 * 4)
    var t = make_view(buf +% 0, make_layout((8, 4)))
    for i in 0 ..< 8:
      for j in 0 ..< 4:
        t(i, j) = float32(10 * i + j)
    let strip = alongAxisAt(t, axis = 1, idx = 5)
    doAssert strip.shape === (4,), "strip shape is (4,)"
    for r in 0 ..< 4:
      doAssert strip(r) == t(5, r), "strip(" & $r & ") aliases t(5, " & $r & ")"

  block: # 2D column-major: the strip keeps the source axis stride
    var buf = newSeq[float32](8 * 4)
    var t = make_view(buf +% 0, make_layout((8, 4), (1, 8)))
    for i in 0 ..< 8:
      for j in 0 ..< 4:
        t(i, j) = float32(10 * i + j)
    let strip = alongAxisAt(t, axis = 0, idx = 2)
    doAssert strip.shape === (8,), "strip shape is (8,)"
    doAssert strip.getLayout().stride === (1,), "strip keeps the axis stride 1"
    for r in 0 ..< 8:
      doAssert strip(r) == t(r, 2), "strip(" & $r & ") aliases t(" & $r & ", 2)"

  block: # 3D, middle axis: the flat idx unravels over the outer axes, dim 0 fastest
    var buf = newSeq[float32](2 * 3 * 4)
    var t = make_view(buf +% 0, make_layout((2, 3, 4)))
    for i in 0 ..< 2:
      for j in 0 ..< 3:
        for k in 0 ..< 4:
          t(i, j, k) = float32(100 * i + 10 * j + k)
    let strip = alongAxisAt(t, axis = 1, idx = 5)
    # others (2, 4) flat 5 -> (1, 2), strip aliases t(1, :, 2)
    doAssert strip.shape === (3,), "strip shape is (3,)"
    for r in 0 ..< 3:
      doAssert strip(r) == t(1, r, 2), "strip(" & $r & ") aliases t(1, " & $r & ", 2)"

  block: # 3D, strided source: offsets go through the others' strides
    var buf = newSeq[float32](4 * 2 * 3)
    var t = make_view(buf +% 0, make_layout((2, 3, 4), (6, 2, 1)))
    for i in 0 ..< 2:
      for j in 0 ..< 3:
        for k in 0 ..< 4:
          t(i, j, k) = float32(100 * i + 10 * j + k)
    let strip = alongAxisAt(t, axis = 2, idx = 4)
    # others (2, 3) flat 4 -> (0, 2), strip aliases t(0, 2, :)
    doAssert strip.shape === (4,), "strip shape is (4,)"
    for r in 0 ..< 4:
      doAssert strip(r) == t(0, 2, r), "strip(" & $r & ") aliases t(0, 2, " & $r & ")"

  block: # rank 1: the only strip is the whole tensor
    var buf = newSeq[float32](6)
    var t = make_view(buf +% 0, make_layout((6,)))
    for i in 0 ..< 6:
      t(i) = float32(i)
    let strip = alongAxisAt(t, axis = 0, idx = 0)
    doAssert strip.shape === (6,), "strip shape is (6,)"
    for r in 0 ..< 6:
      doAssert strip(r) == t(r), "strip(" & $r & ") aliases t(" & $r & ")"

  echo "t_tensor_along_axis_at: all checks passed"

when isMainModule:
  runTest()
