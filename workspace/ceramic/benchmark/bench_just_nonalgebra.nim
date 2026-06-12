{.experimental: "callOperator".}
## Isolated benchmark of nonalgebra

import std/[monotimes, times, math, random, strutils]
import ../examples/ex02_matmul_simd/ex02_matmul_simd_nonalgebra

proc gflops(elapsed: float64; ops: float64): float64 =
  (ops / 1e9) / elapsed

when isMainModule:
  randomize(42)
  echo "SIMD: unimplemented (simdArchString commented out in nonalgebra)\n"

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

    echo "\n--- ", N, "x", N, " column-major ---"

    # Warmup
    gemm_strided_non_algebra(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)

    # Best of 3
    var best = float64.high
    for trial in 0 ..< 3:
      gemm_strided_non_algebra(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
      let t0 = getMonoTime()
      gemm_strided_non_algebra(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
      let t1 = getMonoTime()
      let sec = float64((t1 - t0).inNanoseconds) / 1e9
      best = min(best, sec)
    echo "gemm_strided_non_algebra id ", N, "x", N, ": ",
      gflops(best, ops).formatFloat(ffDecimal, 2), " GFlop/s  (",
      (best*1e6).formatFloat(ffDecimal, 1), " us)"

  echo "\nDone."
