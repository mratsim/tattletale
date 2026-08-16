# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Anti-regression: the tensor `()` one-liner with a mixed
## scalar/underscore coordinate on a nested-mode layout.
##
## The view `v` sits on the copy-partition layout with the rank-3
## shape ((4, 1), 1, 8). The partition_S one-liner `v(thrIdx, _, _)`
## must return the thread's slice. The `()` template builds the
## coord (thrIdx, _, _), slices the layout, and computes the data
## offset through crd2idx. The tuple-coord crd2idx must decompose a
## scalar thread index into the nested (4, 1) thread mode, with
## recursion anchored on the shape (CuTe semantics).
##
## The expected thread-3 slice is shape (1, 8), strides (0, 16),
## origin 12. All asserted values below are cross-checked against
## the CuTe twin compiled against the pinned CUTLASS.

{.experimental: "callOperator".}

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/tests/layouts_testutils

proc runTests =
  echo "\n── tensor () mixed coord on nested layout (anti-regression) ──"
  block:
    # The copy-partition layout: the rank-3 shape ((4, 1), 1, 8) with
    # the nested (4, 1) thread mode and the chunk stride 16.
    const pL = make_layout(((4, 1), 1, 8), ((4, 16), 0, 16))
    var buf: array[128, int32]
    for i in 0 ..< 128: buf[i] = i.int32
    let v = make_view(buf[0].addr, pL)

    # Thread-3 slice: shape (1, 8), strides (0, 16), data at element 12.
    let sub = v(3, _, _)
    check sub.shape, (1, 8), (Int[1], Int[8])
    check sub.stride, (0, 16), (Int[0], Int[16])
    doAssert cast[ptr int32](sub.data) == buf[0].addr +% 12

    # Flat-thread-index origins: crd2idx of the scalar thread index in
    # this layout is 4 * thrIdx (CuTe: 0, 4, 12, 60 for threads 0, 1, 3, 15).
    check crd2idx(pL, 0), 0, Int[0]
    check crd2idx(pL, 1), 4, Int[4]
    check crd2idx(pL, 3), 12, Int[12]
    check crd2idx(pL, 15), 60, Int[60]

    # The 8 chunk origins 12 + 16*i, cross-checked against the
    # independent flat reference 4 * (3 + i * 4).
    for i in 0 ..< 8:
      doAssert sub[0, i] == (4 * (3 + i * 4)).int32
  echo "  All tests passed."

when isMainModule:
  runTests()
