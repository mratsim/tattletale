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

proc runAllcloseTest*(run: static Runtime, name: string) =
  var xd = @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
  var yd = @[1.0'f32, 2.0'f32, 3.0'f32, 4.5'f32]
  let x = toTensor(xd, @[2, 2])
  let y = toTensor(yd, @[2, 2])

  doAssert allclose(x, x), name & ": self"
  doAssert not allclose(x, y), name & ": 0.5 off defaults"
  doAssert allclose(x, y, atol = 0.6'f32), name & ": atol covers"
  doAssert allclose(x, y, rtol = 0.2'f32), name & ": rtol scales with b"
  doAssert not allclose(x, y, rtol = 0.1'f32), name & ": rtol too small"

  # rtol scales with |b|, not |a|: 0.3 off b = 1.0 passes at rtol 0.5,
  # 0.3 off b = 0.1 does not
  var ad = @[1.3'f32, 0.4'f32]
  var bd = @[1.0'f32, 0.1'f32]
  let a = toTensor(ad, @[1, 2])
  let b = toTensor(bd, @[1, 2])
  doAssert not allclose(a, b, rtol = 0.5'f32), name & ": b is the reference side"

  # NaN pairs
  var nd = @[1.0'f32, NaN]
  var md = @[1.0'f32, NaN]
  let n = toTensor(nd, @[1, 2])
  let m = toTensor(md, @[1, 2])
  doAssert not allclose(n, m), name & ": NaN pair, equal_nan off"
  doAssert allclose(n, m, equal_nan = true), name & ": NaN pair, equal_nan on"
  var pd = @[1.0'f32, 2.0'f32]
  let p = toTensor(pd, @[1, 2])
  doAssert not allclose(n, p, equal_nan = true), name & ": NaN vs number"

  # infinities
  var fd = @[Inf.float32, -Inf.float32]
  var gd = @[Inf.float32, -Inf.float32]
  let f = toTensor(fd, @[1, 2])
  let g = toTensor(gd, @[1, 2])
  doAssert allclose(f, g), name & ": same-sign infinities"
  var hd = @[Inf.float32, Inf.float32]
  let h = toTensor(hd, @[1, 2])
  doAssert not allclose(f, h), name & ": +inf vs -inf"
  var idd = @[Inf.float32, 1.0'f32]
  var jd = @[1.0'f32, 1.0'f32]
  let i = toTensor(idd, @[1, 2])
  let j = toTensor(jd, @[1, 2])
  doAssert not allclose(i, j), name & ": inf vs finite"

  # low precision inputs upcast on read
  var xb: array[4, bfloat16]
  var yb: array[4, bfloat16]
  for i in 0 ..< 4:
    xb[i] = xd[i].to(bfloat16)
    yb[i] = yd[i].to(bfloat16)
  let xbq = toTensor(xb, @[2, 2])
  let ybq = toTensor(yb, @[2, 2])
  doAssert not allclose(xbq, ybq), name & ": bf16 0.5 off"
  doAssert allclose(xbq, ybq, atol = 0.6'f32), name & ": bf16 atol covers"

  echo "tensorcery allclose (", name, "): all checks passed"
