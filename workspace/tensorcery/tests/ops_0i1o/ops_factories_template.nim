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

  block:
    ## Nim: zeros([2, 3], run, kFp32) / zeros([2, 3], run, kBf16)
    ## MLX: mx.zeros((2, 3), mx.float32) / mx.zeros((2, 3), mx.bfloat16)
    ## PyTorch: torch.zeros(2, 3) / torch.zeros(2, 3, dtype=torch.bfloat16)
    let z32 = zeros([2, 3], run, kFp32)
    eval(z32)
    doAssert z32.allClose([0.0'f32, 0, 0, 0, 0, 0].toTensor([2, 3], run, float32)), name & ": zeros fp32"
    let zbf = zeros([2, 3], run, kBf16)
    eval(zbf)
    doAssert zbf.allClose([0.0'f32, 0, 0, 0, 0, 0].toTensor([2, 3], run, bfloat16)), name & ": zeros bf16"

  block:
    ## Nim: ones([2, 2], run, kBf16)
    ## MLX: mx.ones((2, 2), mx.bfloat16)
    ## PyTorch: torch.ones(2, 2, dtype=torch.bfloat16)
    let o = ones([2, 2], run, kBf16)
    eval(o)
    doAssert o.allClose([1.0'f32, 1, 1, 1].toTensor([2, 2], run, bfloat16)), name & ": ones"

  block:
    ## Nim: full([2, 2], 2.5, run, kBf16) / full([4], -1.5, run, kFp32)
    ## MLX: mx.full((2, 2), 2.5, mx.bfloat16) / mx.full((4,), -1.5)
    ## PyTorch: torch.full((2, 2), 2.5, dtype=torch.bfloat16) / torch.full((4,), -1.5)
    let f = full([2, 2], 2.5'f32, run, kBf16)
    eval(f)
    doAssert f.allClose([2.5'f32, 2.5, 2.5, 2.5].toTensor([2, 2], run, bfloat16)), name & ": full bf16"
    let f32 = full([4], -1.5'f32, run, kFp32)
    eval(f32)
    doAssert f32.allClose([-1.5'f32, -1.5, -1.5, -1.5].toTensor([4], run, float32)), name & ": full fp32"

  block:
    ## Nim: zeros_like(f) / ones_like(f) / full_like(z, 7.0) / full_like(z, 7.0, kFp32)
    ## MLX: mx.zeros_like(f) / mx.ones_like(f) / mx.full_like(z, 7.0) / mx.full_like(z, 7.0, mx.float32)
    ## PyTorch: torch.zeros_like(f) / torch.ones_like(f) / torch.full_like(z, 7.0) / torch.full_like(z, 7.0, dtype=torch.float32)
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

  block:
    ## Nim: eye(3, run, kFp32) / eye(3, 4, run, kBf16, 1) / eye(2, 3, run, kFp16, -1)
    ## MLX: mx.eye(3) / mx.eye(3, 4, 1, mx.bfloat16) / mx.eye(2, 3, -1)
    ## PyTorch: torch.eye(3) / torch.eye(3, 4, dtype=torch.bfloat16)
    let e0 = eye(3, run, kFp32)
    eval(e0)
    doAssert e0.allClose([1.0'f32, 0, 0, 0, 1, 0, 0, 0, 1].toTensor([3, 3], run, float32)), name & ": eye k0"
    let e1 = eye(3, 4, run, kBf16, 1)
    eval(e1)
    doAssert e1.allClose([0.0'f32, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1].toTensor([3, 4], run, bfloat16)), name & ": eye k1"
    let em1 = eye(2, 3, run, kFp32, -1)
    eval(em1)
    doAssert em1.allClose([0.0'f32, 0, 0, 1, 0, 0].toTensor([2, 3], run, float32)), name & ": eye k-1"

  block:
    ## Nim: arange(10, run, kFp32) / arange(1, 14, run, 3.0, kFp32)
    ##      arange(10.0, 0.0, run, -1.5, kBf16)
    ## MLX: mx.arange(10) / mx.arange(1, 14, 3) / mx.arange(10, 0, -1.5, mx.bfloat16)
    ## PyTorch: torch.arange(10) / torch.arange(1, 14, 3) / torch.arange(10, 0, -1.5, dtype=torch.bfloat16)
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

  # Refusals: arange with a zero step, a step that counts away from stop,
  # and a rank-0 shape
  block:
    var x = [1.0'f32, 2].toTensor([2], run, bfloat16)
    doAssertRaises AssertionDefect:
      discard arange(0.0'f32, 10.0'f32, run, 0.0'f32, kBf16)
    doAssertRaises AssertionDefect:
      discard arange(10.0'f32, 0.0'f32, run, 1.0'f32, kBf16)
    doAssertRaises AssertionDefect:
      discard zeros([], run, kBf16)

  # Size-0 shapes are out of scope: the node constructs, eval hits the backend allocation raise and no size-0 guard exists
  block:
    let z0 = zeros([0], run, kBf16)
    doAssertRaises AssertionDefect:
      eval(z0)

  echo "tensorcery factories (", name, "): all checks passed"
