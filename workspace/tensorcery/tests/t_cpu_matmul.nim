# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU matmul placeholder against an independent f64 reference.
##
## Test that matmulStrided matches an independent f64-accumulating GEMM on seeded random matrices.
## Test that the square, rectangular, strided-view and beta-combined cases all agree.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --nimcache:nimcache/t_cpu_matmul workspace/tensorcery/tests/t_cpu_matmul.nim

import std/random

import workspace/tensorcery/src/kernels/k_matmul_cpu

proc seededMatrix(rng: var Rand, rows, cols: int): seq[float32] =
  ## Deterministic matrix in [-1, 1), contiguous along the second axis.
  newSeq(result, rows * cols)
  for i in 0 ..< rows * cols:
    result[i] = float32(rng.rand(-1.0 .. 1.0))

proc referenceF64(M, N, K: int, alpha, beta: float64,
    a: openArray[float32], rsA, csA: int,
    b: openArray[float32], rsB, csB: int,
    cInit: openArray[float32], rsC, csC: int): seq[float64] =
  ## Independent reference: f64 accumulation over a k-i-j order, distinct from the kernel's f32 k-j-i loop.
  newSeq(result, M * N)
  for i in 0 ..< M:
    for j in 0 ..< N:
      let ci = i * rsC + j * csC
      var acc = 0.0
      for k in 0 ..< K:
        acc += float64(a[i * rsA + k * csA]) * float64(b[k * rsB + j * csB])
      result[ci] = float64(cInit[ci]) * beta + acc * alpha

proc maxAbsDiff[T](a: openArray[T], b: openArray[float64]): float64 =
  for i in 0 ..< a.len:
    result = max(result, abs(float64(a[i]) - b[i]))

proc runCase(rng: var Rand, M, N, K: int, alpha, beta: float32, rsA, csA, rsB, csB: int) =
  ## One seeded case over (rsA, csA) and (rsB, csB) strided views.
  ## The flat buffers span the full strided bounds, so column-major views fit too.
  let aBuf = seededMatrix(rng, 1, (M - 1) * rsA + (K - 1) * csA + 1)
  let bBuf = seededMatrix(rng, 1, (K - 1) * rsB + (N - 1) * csB + 1)
  let cInit = seededMatrix(rng, M, N)
  var actual = cInit
  matmulStrided(M, N, K, alpha,
    cast[ptr UncheckedArray[float32]](aBuf[0].unsafeAddr), rsA, csA,
    cast[ptr UncheckedArray[float32]](bBuf[0].unsafeAddr), rsB, csB,
    beta,
    cast[ptr UncheckedArray[float32]](actual[0].addr), N, 1)
  let expected = referenceF64(M, N, K, float64(alpha), float64(beta),
    aBuf, rsA, csA, bBuf, rsB, csB, cInit, N, 1)
  let diff = maxAbsDiff(actual, expected)
  if diff >= 1e-3:
    var worst = 0
    var worstD = 0.0
    for i in 0 ..< actual.len:
      let d = abs(float64(actual[i]) - expected[i])
      if d > worstD:
        worstD = d
        worst = i
    echo "worst at flat ", worst, " (i=", worst div N, " j=", worst mod N, ") actual=", actual[worst], " expected=", expected[worst]
  doAssert diff < 1e-3, "matmul diverged (M=" & $M & " N=" & $N & " K=" & $K &
      "), max abs diff " & $diff

proc main() =
  var rng = initRand(20261105)
  runCase(rng, 64, 64, 64, 1.0'f32, 0.0'f32, 64, 1, 64, 1)
  runCase(rng, 37, 96, 45, 0.5'f32, 0.25'f32, 45, 1, 96, 1)
  runCase(rng, 64, 64, 45, 1.0'f32, 0.0'f32, 50, 1, 70, 1)
  runCase(rng, 32, 48, 16, 1.0'f32, 1.0'f32, 1, 32, 1, 48)
  echo "t_cpu_matmul OK"

main()
