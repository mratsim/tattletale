# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/random
import ../tensorcery_testutils

export tensorcery_testutils

proc refGemm(dst: var openArray[float32], a, b: openArray[float32], m, k, n: int,
    alpha: float32, beta: float32, c: openArray[float32] = []) =
  ## dst = alpha·a·b + beta·c, a (m, k), b (k, n), row-major
  for i in 0 ..< m:
    for j in 0 ..< n:
      var acc = 0.0'f32
      for p in 0 ..< k:
        acc += a[i * k + p] * b[p * n + j]
      dst[i * n + j] = alpha * acc
      if c.len > 0:
        dst[i * n + j] += beta * c[i * n + j]

proc padded(content: seq[float32], rows, cols: int, scale: float32): seq[float32] =
  result.newSeq(rows * (cols + 2))
  for i in 0 ..< rows:
    result[i * (cols + 2)] = 7.0'f32
    for j in 0 ..< cols:
      result[i * (cols + 2) + 1 + j] = scale * content[i * cols + j]
    result[i * (cols + 2) + cols + 1] = 7.0'f32

proc runGemmTest*(run: static Runtime, name: string) =
  var rng = initRand(0xFACADE)
  for (m, k, n) in [(3, 5, 4), (17, 33, 13), (32, 64, 32)]:
    let
      a = rng.randomSeq(float32, m * k)
      b = rng.randomSeq(float32, k * n)
      A = a.toTensor(run, m, k)
      B = b.toTensor(run, k, n)

    block: # matmul: alpha = 1, beta = 0
      var expect = newSeq[float32](m * n)
      expect.refGemm(a, b, m, k, n, 1.0'f32, 0.0'f32)
      let d = matmul(A, B)
      eval(d)
      doAssert d.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": matmul " & $((m, k, n))

    block: # scaled
      var expect = newSeq[float32](m * n)
      expect.refGemm(a, b, m, k, n, 0.5'f32, 0.0'f32)
      let d = gemm(0.5'f32, A, B)
      eval(d)
      doAssert d.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": scaled " & $((m, k, n))

    block: # accumulate into C
      let
        c = rng.randomSeq(float32, m * n)
        C = c.toTensor(run, m, n)
      var expect = newSeq[float32](m * n)
      expect.refGemm(a, b, m, k, n, 1.0'f32, 0.5'f32, c)
      let d = gemm(1.0'f32, A, B, 0.5'f32, C)
      eval(d)
      doAssert d.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": accumulate " & $((m, k, n))

    block: # C with beta = 0 is accepted, never loaded
      let
        garbageC = rng.randomSeq(float32, m * n)
        C = garbageC.toTensor(run, m, n)
      var expect = newSeq[float32](m * n)
      expect.refGemm(a, b, m, k, n, 1.0'f32, 0.0'f32)
      let d = gemm(1.0'f32, A, B, 0.0'f32, C)
      eval(d)
      doAssert d.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": beta-0 ignores C " & $((m, k, n))

    block:
      let
        aLeaf = a.padded(m, k, 1.0'f32).toTensor(run, m, k + 2)
        bLeaf = b.padded(k, n, 1.0'f32).toTensor(run, k, n + 2)
        aLeafHalf = a.padded(m, k, 0.5'f32).toTensor(run, m, k + 2)
        bLeafHalf = b.padded(k, n, 0.5'f32).toTensor(run, k, n + 2)
        aOp = aLeafHalf + aLeafHalf
        bOp = bLeafHalf + bLeafHalf
      var expect = newSeq[float32](m * n)
      expect.refGemm(a, b, m, k, n, 1.0'f32, 0.0'f32)
      let
        d1 = matmul(aLeaf.narrow(1, 1, k), B)
        d2 = matmul(A, bLeaf.narrow(1, 1, n))
        d3 = matmul(aLeaf.narrow(1, 1, k), bLeaf.narrow(1, 1, n))
        d4 = matmul(aOp.narrow(1, 1, k), B)
        d5 = matmul(A, bOp.narrow(1, 1, n))
        d6 = matmul(aOp.narrow(1, 1, k), bOp.narrow(1, 1, n))
      eval(d1, d2, d3, d4, d5, d6)
      doAssert d1.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": narrow(a) @ b over an eager leaf " & $((m, k, n))
      doAssert d2.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": a @ narrow(b) over an eager leaf " & $((m, k, n))
      doAssert d3.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": narrow(a) @ narrow(b) over an eager leaf " & $((m, k, n))
      doAssert d4.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": narrow(a) @ b over an unallocated op output " & $((m, k, n))
      doAssert d5.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": a @ narrow(b) over an unallocated op output " & $((m, k, n))
      doAssert d6.allClose(expect.toTensor(run, m, n), rtol = 1e-5'f32, atol = 1e-3'f32),
        name & ": narrow(a) @ narrow(b) over an unallocated op output " & $((m, k, n))

    block: # nil C with beta != 0 is a construction error
      try:
        discard gemm(1.0'f32, A, B, 2.0'f32)
        raise newException(CatchableError, name & ": nil C with beta != 0 must be rejected")
      except AssertionDefect:
        discard
  echo "tensorcery gemm (", name, "): all checks passed"
