# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/math
import workspace/tensorcery/tests/tensorcery_testutils
import workspace/tensorcery/src/tensors_internal_types
import workspace/tensorcery/src/tensors_internals
import workspace/tensorcery/src/tensors_globals
import workspace/tensorcery/src/compute_graph/eval

export tensorcery_testutils

proc runallCloseTest*(run: static Runtime, name: string) =
  var xd = @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
  var yd = @[1.0'f32, 2.0'f32, 3.0'f32, 4.5'f32]
  let x = xd.toTensor(2, 2)
  let y = yd.toTensor(2, 2)

  doAssert x.allClose(x), name & ": self"
  doAssert not x.allClose(y), name & ": 0.5 off defaults"
  doAssert x.allClose(y, atol = 0.6'f32), name & ": atol covers"
  doAssert x.allClose(y, rtol = 0.2'f32), name & ": rtol scales with b"
  doAssert not x.allClose(y, rtol = 0.1'f32), name & ": rtol too small"

  # rtol scales with |b|, not |a|: 0.3 off b = 1.0 passes at rtol 0.5,
  # 0.3 off b = 0.1 does not
  var ad = @[1.3'f32, 0.4'f32]
  var bd = @[1.0'f32, 0.1'f32]
  let a = ad.toTensor(1, 2)
  let b = bd.toTensor(1, 2)
  doAssert not a.allClose(b, rtol = 0.5'f32), name & ": b is the reference side"

  # NaN pairs
  var nd = @[1.0'f32, NaN]
  var md = @[1.0'f32, NaN]
  let n = nd.toTensor(1, 2)
  let m = md.toTensor(1, 2)
  doAssert not n.allClose(m), name & ": NaN pair, equal_nan off"
  doAssert n.allClose(m, equal_nan = true), name & ": NaN pair, equal_nan on"
  var pd = @[1.0'f32, 2.0'f32]
  let p = pd.toTensor(1, 2)
  doAssert not n.allClose(p, equal_nan = true), name & ": NaN vs number"

  # infinities
  var fd = @[Inf.float32, -Inf.float32]
  var gd = @[Inf.float32, -Inf.float32]
  let f = fd.toTensor(1, 2)
  let g = gd.toTensor(1, 2)
  doAssert f.allClose(g), name & ": same-sign infinities"
  var hd = @[Inf.float32, Inf.float32]
  let h = hd.toTensor(1, 2)
  doAssert not f.allClose(h), name & ": +inf vs -inf"
  var idd = @[Inf.float32, 1.0'f32]
  var jd = @[1.0'f32, 1.0'f32]
  let i = idd.toTensor(1, 2)
  let j = jd.toTensor(1, 2)
  doAssert not i.allClose(j), name & ": inf vs finite"

  # low precision inputs upcast on read
  var xb: array[4, bfloat16]
  var yb: array[4, bfloat16]
  for i in 0 ..< 4:
    xb[i] = xd[i].to(bfloat16)
    yb[i] = yd[i].to(bfloat16)
  let xbq = xb.toTensor(2, 2)
  let ybq = yb.toTensor(2, 2)
  doAssert not xbq.allClose(ybq), name & ": bf16 0.5 off"
  doAssert xbq.allClose(ybq, atol = 0.6'f32), name & ": bf16 atol covers"

  echo "tensorcery allClose (", name, "): all checks passed"
