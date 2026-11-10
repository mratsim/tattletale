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

proc runSplitTest*(run: static Runtime, name: string) =
  var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  var xq: array[8, bfloat16]
  for i in 0 ..< 8:
    xq[i] = xd[i].to(bfloat16)
  let x = toTensor(xq, @[2, 4])

  let parts = split(x, 0, 2)
  doAssert parts.len == 2
  for p in parts:
    doAssert p.shape()[0] == 1
    doAssert p.shape()[1] == 4
  for c in 0 ..< 4:
    doAssert parts[0].item(0, c, bfloat16) == xd[c].to(bfloat16), name & ": part0 " & $c
    doAssert parts[1].item(0, c, bfloat16) == xd[4 + c].to(bfloat16), name & ": part1 " & $c

  # inner-axis split makes strided views, strides inherit the source
  let cols = split(x, 1, 2)
  let w = cols[0] + cols[1]
  for r in 0 ..< 2:
    for c in 0 ..< 2:
      let e0 = xd[r * 4 + c].to(bfloat16).to(float32)
      let e1 = xd[r * 4 + c + 2].to(bfloat16).to(float32)
      doAssert w.item(r, c, bfloat16) == (e0 + e1).to(bfloat16), name & ": colsplit " & $r & "," & $c

  doAssertRaises AssertionDefect:
    discard split(x, 0, 3)

  echo "tensorcery split (", name, "): all checks passed"
