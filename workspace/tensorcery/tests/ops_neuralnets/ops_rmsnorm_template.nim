# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/math
import std/random
import ../tensorcery_testutils

export tensorcery_testutils


proc refRmsnorm(dst: var openArray[float32], src: openArray[float32], rows, cols: int, eps: float32, weight: openArray[float32] = []) =
  ## dst = src * rsqrt(mean(src^2 over the last axis) + eps)
  ## the weight broadcasts per column when provided
  for r in 0 ..< rows:
    var ms = 0.0'f32
    for c in 0 ..< cols:
      ms += src[r * cols + c] * src[r * cols + c]
    let scale = 1.0'f32 / sqrt(ms / cols.float32 + eps)
    for c in 0 ..< cols:
      if weight.len > 0:
        dst[r * cols + c] = src[r * cols + c] * scale * weight[c]
      else:
        dst[r * cols + c] = src[r * cols + c] * scale

proc runRmsnormTest*(run: static Runtime, name: string) =
  var rng = initRand(0xFACADE)
  for shape in [ @[3, 8], @[5, 13], @[2, 3, 7], @[2, 3, 5, 7], @[2, 3, 5, 7, 11] ]:
    let cols = shape[^1]
    var leading = 1
    for i in 0 ..< shape.len - 1:
      leading *= shape[i]

    var src = rng.randomSeq(float32, leading * cols)
    var weight = rng.randomSeq(float32, cols, 0.5'f32 .. 1.5'f32)

    var expect = newSeq[float32](src.len)
    expect.refRmsnorm(src, leading, cols, 1e-5'f32, weight)
    var expectUnweighted = newSeq[float32](src.len)
    expectUnweighted.refRmsnorm(src, leading, cols, 1e-5'f32)

    let x = src.toTensor(run, shape)
    let w = weight.toTensor(run, cols)
    let got = x.rms_norm(w)
    let gotUnweighted = x.rms_norm()

    doAssert got.allClose(expect.toTensor(run, shape), rtol = 1e-5'f32, atol = 1e-4'f32),
        name & ": weighted " & $shape
    doAssert gotUnweighted.allClose(expectUnweighted.toTensor(run, shape), rtol = 1e-5'f32, atol = 1e-4'f32),
        name & ": unweighted " & $shape

  echo "tensorcery rmsnorm (", name, "): all checks passed"
