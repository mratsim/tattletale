# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/random
import ../tensorcery_testutils

export tensorcery_testutils


proc refLinear(dst: var openArray[float32], x, w: openArray[float32], m, k, n: int, bias: openArray[float32] = []) =
  ## dst = x·wᵀ (+ bias), x (m, k), w (n, k), row-major
  for i in 0 ..< m:
    for j in 0 ..< n:
      var acc = 0.0'f32
      for p in 0 ..< k:
        acc += x[i * k + p] * w[j * k + p]
      dst[i * n + j] = acc
      if bias.len > 0:
        dst[i * n + j] += bias[j]

proc runLinearTest*(run: static Runtime, name: string) =
  var rng = initRand(0xFACADE)
  for (m, k, n) in [(3, 5, 4), (17, 33, 13), (32, 64, 32)]:
    let
      x = rng.randomSeq(float32, m * k)
      w = rng.randomSeq(float32, n * k)
      X = x.toTensor(run, m, k)
      W = w.toTensor(run, n, k)

    block: # no bias
      var expect = newSeq[float32](m * n)
      expect.refLinear(x, w, m, k, n)
      let o = linear(X, W)
      eval(o)
      doAssert o.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": linear " & $((m, k, n))

    block: # with bias
      let
        b = rng.randomSeq(float32, n)
        B = b.toTensor(run, n)
      var expect = newSeq[float32](m * n)
      expect.refLinear(x, w, m, k, n, b)
      let o = linear(X, W, B)
      eval(o)
      doAssert o.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": linear bias " & $((m, k, n))

    block: # batched: (B, M, K) x (K, N) equals B independent (M, K) x (K, N)
      const batch = 3
      let x3 = rng.randomSeq(float32, batch * m * k)
      let X3 = x3.toTensor(run, batch, m, k)

      var expect = newSeq[float32](batch * m * n)
      var slice = newSeq[float32](m * n)
      for i in 0 ..< batch:
        slice.refLinear(x3[i * m * k ..< (i + 1) * m * k], w, m, k, n)
        for j in 0 ..< m * n:
          expect[i * m * n + j] = slice[j]
      let o = linear(X3, W)
      eval(o)
      doAssert o.allClose(expect.toTensor(run, batch, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": linear batched " & $((batch, m, k, n))

      let bias = rng.randomSeq(float32, n)
      let Bt = bias.toTensor(run, n)
      for i in 0 ..< batch:
        slice.refLinear(x3[i * m * k ..< (i + 1) * m * k], w, m, k, n, bias)
        for j in 0 ..< m * n:
          expect[i * m * n + j] = slice[j]
      let ob = linear(X3, W, Bt)
      eval(ob)
      doAssert ob.allClose(expect.toTensor(run, batch, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": linear batched bias " & $((batch, m, k, n))

    block: # batched with strides: x laid out (M, B, K), its (B, M, K) view has batch stride K
      const batch = 3
      let xs = rng.randomSeq(float32, batch * m * k)
      let X = xs.toTensor(run, m, batch, k).swapaxes(0, 1)

      var expect = newSeq[float32](batch * m * n)
      var slice = newSeq[float32](m * n)
      for b in 0 ..< batch:
        var xb = newSeq[float32](m * k)
        for i in 0 ..< m:
          for p in 0 ..< k:
            xb[i * k + p] = xs[i * batch * k + b * k + p]
        slice.refLinear(xb, w, m, k, n)
        for j in 0 ..< m * n:
          expect[b * m * n + j] = slice[j]
      let o = linear(X, W)
      eval(o)
      doAssert o.allClose(expect.toTensor(run, batch, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": linear batched strided " & $((batch, m, k, n))
  echo "tensorcery linear (", name, "): all checks passed"
