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

proc runAddTest*(run: static Runtime, name: string) =
  var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  var yd = @[2.0'f32, 2, 2, 2, 2, 2, 2, 2]
  var xq: array[8, bfloat16]
  var yq: array[8, bfloat16]
  for i in 0 ..< 8:
    xq[i] = xd[i].to(bfloat16)
    yq[i] = yd[i].to(bfloat16)
  let x = xq.toTensor(2, 4)
  let y = yq.toTensor(@[2, 4])

  doAssert x.shape().len == 2
  doAssert x.shape()[0] == 2
  doAssert x.shape()[1] == 4
  doAssert x.strides()[0] == 4
  doAssert x.strides()[1] == 1
  doAssert x.dtype() == kBf16
  doAssert x.size() == 8

  let z = x + y
  let small = yq[0 ..< 4].toTensor(@[1, 4])
  # face checks are contracts, they raise AssertionDefect
  doAssertRaises AssertionDefect:
    discard x + small
  doAssertRaises AssertionDefect:
    discard xq[0 ..< 7].toTensor(2, 4)

  var zd: array[8, bfloat16]
  for i in 0 ..< 8:
    let xi = xd[i].to(bfloat16).to(float32)
    let yi = yd[i].to(bfloat16).to(float32)
    zd[i] = (xi + yi).to(bfloat16)
  doAssert z.allClose(zd.toTensor(2, 4)), name & ": add"
  doAssert x.allClose(xq.toTensor(2, 4)), name & ": add source"

  # Memoize computation and drop ancestor nodes
  let d = x + y
  let e = d + x
  eval(e)
  eval(e)
  var ed: array[8, bfloat16]
  for i in 0 ..< 8:
    let xi = xd[i].to(bfloat16).to(float32)
    let yi = yd[i].to(bfloat16).to(float32)
    ed[i] = (xi * 2.0'f32 + yi).to(bfloat16)
  doAssert e.allClose(ed.toTensor(2, 4)), name & ": diamond"

  echo "tensorcery add (", name, "): all checks passed"
