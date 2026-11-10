# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/tensorcery/tests/tensorcery_testutils
import workspace/tensorcery/src/tensors_internal_types
import workspace/tensorcery/src/tensors_internals
import workspace/tensorcery/src/tensors_globals
import workspace/tensorcery/src/compute_graph/eval

export tensorcery_testutils

proc runSwapaxesTest*(run: static Runtime, name: string) =
  var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  var xq: array[8, bfloat16]
  for i in 0 ..< 8:
    xq[i] = xd[i].to(bfloat16)
  let x = toTensor(xq, @[2, 4])

  # view semantics are metadata-only until finalization
  # shape permutes at construction, data/offset/strides bind at the walk
  let sx = swapaxes(x, 0, 1)
  doAssert sx.shape()[0] == 4 and sx.shape()[1] == 2
  doAssert sx.dtype() == kBf16

  # downstream ops read the view with inherited strides
  let sw = sx + sx
  eval(sw)
  doAssert sx.strides()[0] == 1 and sx.strides()[1] == 4
  for i in 0 ..< 2:
    for j in 0 ..< 4:
      doAssert sx.item(j, i, bfloat16) == x.item(i, j, bfloat16), name & ": swapaxes " & $i & "," & $j
      let e2 = xd[i * 4 + j].to(bfloat16).to(float32) * 2.0'f32
      doAssert sw.item(j, i, bfloat16) == e2.to(bfloat16), name & ": swapaxes add " & $i & "," & $j

  doAssertRaises AssertionDefect:
    discard swapaxes(x, 0, 2)

  echo "tensorcery swapaxes (", name, "): all checks passed"
