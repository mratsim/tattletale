# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import ../tensorcery_testutils

export tensorcery_testutils

from std/importutils import privateAccess
import workspace/tensorcery

proc runFactoriesTest*(run: static Runtime, name: string) =
  privateAccess(Tensor)

  block: # Zeros in fp32 and bf16
    let z32 = zeros([2, 3], run, kFp32)
    eval(z32)
    doAssert z32.allClose([0.0'f32, 0, 0, 0, 0, 0].toTensor([2, 3], run, float32)), name & ": zeros fp32"
    let zbf = zeros([2, 3], run, kBf16)
    eval(zbf)
    doAssert zbf.allClose([0.0'f32, 0, 0, 0, 0, 0].toTensor([2, 3], run, bfloat16)), name & ": zeros bf16"

  block: # Ones
    let o = ones([2, 2], run, kBf16)
    eval(o)
    doAssert o.allClose([1.0'f32, 1, 1, 1].toTensor([2, 2], run, bfloat16)), name & ": ones"

  block: # Full in bf16 and fp32
    let f = full([2, 2], 2.5'f32, run, kBf16)
    eval(f)
    doAssert f.allClose([2.5'f32, 2.5, 2.5, 2.5].toTensor([2, 2], run, bfloat16)), name & ": full bf16"
    let f32 = full([4], -1.5'f32, run, kFp32)
    eval(f32)
    doAssert f32.allClose([-1.5'f32, -1.5, -1.5, -1.5].toTensor([4], run, float32)), name & ": full fp32"

  block: # Like variants inherit shape and dtype, the dtype overload overrides
    let f = full([2, 2], 2.5'f32, run, kBf16)
    eval(f)
    let zl = zeros_like(f)
    eval(zl)
    doAssert zl.allClose([0.0'f32, 0, 0, 0].toTensor([2, 2], run, bfloat16)), name & ": zeros_like"
    doAssert zl.dtype() == kBf16, name & ": zeros_like dtype"
    let ol = ones_like(f, kFp32)
    eval(ol)
    doAssert ol.allClose([1.0'f32, 1, 1, 1].toTensor([2, 2], run, float32)), name & ": ones_like dtype"
    let z = zeros([2, 3], run, kBf16)
    let fl = full_like(z, 7.0'f32)
    eval(fl)
    doAssert fl.allClose([7.0'f32, 7, 7, 7, 7, 7].toTensor([2, 3], run, bfloat16)), name & ": full_like"
    let fl32 = full_like(z, 7.0'f32, kFp32)
    eval(fl32)
    doAssert fl32.allClose([7.0'f32, 7, 7, 7, 7, 7].toTensor([2, 3], run, float32)), name & ": full_like dtype"

  block: # Eye: square k=0, rectangular superdiagonal k=1, subdiagonal k=-1
    let e0 = eye(3, run, kFp32)
    eval(e0)
    doAssert e0.allClose([1.0'f32, 0, 0, 0, 1, 0, 0, 0, 1].toTensor([3, 3], run, float32)), name & ": eye k0"
    let e1 = eye(3, 4, run, kBf16, 1)
    eval(e1)
    doAssert e1.allClose([0.0'f32, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1].toTensor([3, 4], run, bfloat16)), name & ": eye k1"
    let em1 = eye(2, 3, run, kFp32, -1)
    eval(em1)
    doAssert em1.allClose([0.0'f32, 0, 0, 1, 0, 0].toTensor([2, 3], run, float32)), name & ": eye k-1"

  block: # Eye: 4098 squared has flat indices over 2^24 and every diagonal entry stays 1
    let eBig = eye(4098, run, kFp32)
    eval(eBig)
    var expectBig = newSeq[float32](4098 * 4098)
    for i in 0 ..< 4098:
      expectBig[i * (4098 + 1)] = 1.0'f32
    doAssert eBig.allClose(expectBig.toTensor([4098, 4098], run, float32)), name & ": eye 4098 diagonal"

  block: # Arange: stop-only, positive step, negative step
    let a0 = arange(10.0'f32, run, kFp32)
    eval(a0)
    doAssert a0.allClose([0.0'f32, 1, 2, 3, 4, 5, 6, 7, 8, 9].toTensor([10], run, float32)), name & ": arange stop"
    let a1 = arange(1.0'f32, 14.0'f32, run, 3.0'f32, kFp32)
    eval(a1)
    doAssert a1.allClose([1.0'f32, 4, 7, 10, 13].toTensor([5], run, float32)), name & ": arange step 3"
    let a2 = arange(10.0'f32, 0.0'f32, run, -1.5'f32, kBf16)
    eval(a2)
    var a2Expect: array[7, bfloat16]
    for i in 0 ..< 7:
      a2Expect[i] = (10.0'f32 - 1.5'f32 * i.float32).to(bfloat16)
    doAssert a2.allClose(a2Expect.toTensor([7], run, bfloat16)), name & ": arange negative step"

  block: # Arange keeps exact sign checks where the product (stop-start)*step underflows to 0
    let aTiny = arange(0.0'f32, 1e-30'f32, run, 1e-30'f32, kFp32)
    eval(aTiny)
    doAssert aTiny.allClose([0.0'f32].toTensor([1], run, float32)), name & ": arange 1e-30"

  block: # Refusals: arange with a zero step, a step that counts away from stop, a rank-0 shape, and eye with a negative size
    var x = [1.0'f32, 2].toTensor([2], run, bfloat16)
    doAssertRaises AssertionDefect:
      discard arange(0.0'f32, 10.0'f32, run, 0.0'f32, kBf16)
    doAssertRaises AssertionDefect:
      discard arange(10.0'f32, 0.0'f32, run, 1.0'f32, kBf16)
    doAssertRaises AssertionDefect:
      discard eye(-2, -3, run, kFp32)
    doAssertRaises AssertionDefect:
      discard zeros([], run, kBf16)

  block: # Size-0 shapes are out of scope: the node constructs, eval hits the backend allocation raise and no size-0 guard exists
    let z0 = zeros([0], run, kBf16)
    doAssertRaises AssertionDefect:
      eval(z0)

  echo "tensorcery factories (", name, "): all checks passed"
