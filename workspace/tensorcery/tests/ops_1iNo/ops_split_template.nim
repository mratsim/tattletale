# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../tensorcery_testutils

export tensorcery_testutils


proc runSplitTest*(run: static Runtime, name: string) =
  var xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  var xq: array[8, bfloat16]
  for i in 0 ..< 8:
    xq[i] = xd[i].to(bfloat16)
  let x = xq.toTensor(run, 2, 4)

  let parts = split(x, 0, 2)
  doAssert parts.len == 2
  for p in parts:
    doAssert p.shape()[0] == 1
    doAssert p.shape()[1] == 4
  doAssert parts[0].allClose(xq[0 ..< 4].toTensor(run, 1, 4)), name & ": part0"
  doAssert parts[1].allClose(xq[4 ..< 8].toTensor(run, 1, 4)), name & ": part1"

  # inner-axis split makes strided views, strides inherit the source
  let cols = split(x, 1, 2)
  let w = cols[0] + cols[1]
  var wd: array[4, bfloat16]
  for r in 0 ..< 2:
    for c in 0 ..< 2:
      let e0 = xd[r * 4 + c].to(bfloat16).to(float32)
      let e1 = xd[r * 4 + c + 2].to(bfloat16).to(float32)
      wd[r * 2 + c] = (e0 + e1).to(bfloat16)
  doAssert w.allClose(wd.toTensor(run, 2, 2)), name & ": colsplit"

  doAssertRaises AssertionDefect:
    discard split(x, 0, 3)

  # split over an unevaluated producer: the parts enqueue it at eval
  # distinct sums per part, a part-to-part offset mixup fails the assertions
  let ad = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]
  let bd = @[10.0'f32, 20, 30, 40, 50, 60, 70, 80]
  let a = ad.toTensor(run, 2, 4)
  let b = bd.toTensor(run, 2, 4)
  let lazyParts = (a + b).split(0, 2)
  eval(lazyParts[0])
  doAssert lazyParts[0].allClose(@[11.0'f32, 22, 33, 44].toTensor(run, 1, 4)), name & ": lazy part0"
  eval(lazyParts[1])
  doAssert lazyParts[1].allClose(@[55.0'f32, 66, 77, 88].toTensor(run, 1, 4)), name & ": lazy part1"

  echo "tensorcery split (", name, "): all checks passed"
