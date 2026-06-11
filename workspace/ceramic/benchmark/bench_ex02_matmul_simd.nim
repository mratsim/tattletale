{.experimental: "callOperator".}
## Benchmark: SIMD matmul — tattletale/ceramic GEMM vs naive triple-loop
##
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed under MIT and Apache 2.0 (see LICENSE).
##
## Measures GFlops on 128×128, 512×512, and 1024×1024 square matrices
## with column-major (Fortran) layout.

import std/[monotimes, times, math, random, strutils]
import ../examples/ex02_matmul_simd/ex02_matmul_simd
import ./laser/gemm as laser_gemm

# ═══════════════════════════════════════════════════════════════════════════
#  Naive triple-loop reference (ijk order, column-major)
# ═══════════════════════════════════════════════════════════════════════════

proc gemm_naive(M, N, K: int; alpha: float32;
    A: openArray[float32]; rsA, csA: int;
    B: openArray[float32]; rsB, csB: int;
    beta: float32; C: var openArray[float32]; rsC, csC: int) =
  for j in 0 ..< N:
    for i in 0 ..< M:
      let ci = i * rsC + j * csC
      C[ci] = if beta == 0.0'f32: 0.0'f32
              elif beta != 1.0'f32: C[ci] * beta
              else: C[ci]
  for j in 0 ..< N:
    for k in 0 ..< K:
      let bv = B[k * rsB + j * csB]
      if bv != 0.0'f32:
        for i in 0 ..< M:
          C[i * rsC + j * csC] += alpha * A[i * rsA + k * csA] * bv

# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark helpers
# ═══════════════════════════════════════════════════════════════════════════

proc gflops(elapsed: float64; ops: float64): float64 =
  (ops / 1e9) / elapsed

when isMainModule:
  randomize(42)

  echo "SIMD: ", simdArchString(), "\n"
  for N in [128, 512, 1024]:
    let
      alpha = 1.0'f32
      beta  = 1.0'f32
      rs = 1
      cs = N
      ops = 2.0 * float64(N) * float64(N) * float64(N)

    var
      A = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
      B = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
      C = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
    for i in 0 ..< A.len: A[i] = rand(1.0'f32)
    for i in 0 ..< B.len: B[i] = rand(1.0'f32)
    for i in 0 ..< C.len: C[i] = rand(1.0'f32)

    echo "\n--- ", N, "x", N, " column-major (", simdArchString(), ") ---"

    # ── Naive triple-loop ──
    gemm_naive(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
    let t0n = getMonoTime()
    gemm_naive(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
    let t1n = getMonoTime()
    let secN = float64((t1n - t0n).inNanoseconds) / 1e9
    echo "triple-loop        ", N, "x", N, ": ",
      gflops(secN, ops).formatFloat(ffDecimal, 2), " GFlop/s  (",
      (secN*1e6).formatFloat(ffDecimal, 1), " us)"

    # ── gemm_strided (identity) ──
    gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
    let t0s = getMonoTime()
    gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
    let t1s = getMonoTime()
    let secS = float64((t1s - t0s).inNanoseconds) / 1e9
    echo "gemm_strided id    ", N, "x", N, ": ",
      gflops(secS, ops).formatFloat(ffDecimal, 2), " GFlop/s  (",
      (secS*1e6).formatFloat(ffDecimal, 1), " us)"

    # ── gemm_strided (ReLU) ──
    gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs, akReLU)
    let t0r = getMonoTime()
    gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs, akReLU)
    let t1r = getMonoTime()
    let secR = float64((t1r - t0r).inNanoseconds) / 1e9
    echo "gemm_strided relu  ", N, "x", N, ": ",
      gflops(secR, ops).formatFloat(ffDecimal, 2), " GFlop/s  (",
      (secR*1e6).formatFloat(ffDecimal, 1), " us)"


    # ── Laser gemm_strided (ptr-based) ──
    laser_gemm.gemm_strided(N, N, N, alpha, addr A[0], rs, cs, addr B[0], rs, cs, beta, addr C[0], rs, cs)
    let t0l = getMonoTime()
    laser_gemm.gemm_strided(N, N, N, alpha, addr A[0], rs, cs, addr B[0], rs, cs, beta, addr C[0], rs, cs)
    let t1l = getMonoTime()
    let secL = float64((t1l - t0l).inNanoseconds) / 1e9
    echo "Laser gemm_strided   ", N, "x", N, ": ",
      gflops(secL, ops).formatFloat(ffDecimal, 2), " GFlop/s  (",
      (secL*1e6).formatFloat(ffDecimal, 1), " us)"
  echo "\nDone."
