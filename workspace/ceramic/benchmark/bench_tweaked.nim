{.experimental: "callOperator".}
## Benchmark: compare ref vs fast vs tweaked (minimal edge-case fix)

import std/[monotimes, times, math, random, strutils]
import ../examples/ex02_matmul_simd/ex02_matmul_simd as v_ref
import ../examples/ex02_matmul_simd/ex02_matmul_simd_nonalgebra as v_fast
import ../examples/ex02_matmul_simd/ex02_matmul_simd_tweaked as v_tweak

proc gflops(elapsed: float64; ops: float64): float64 =
  (ops / 1e9) / elapsed

proc bestOf5(ops: float64; cb: proc(): void): (float64, float64) =
  var best = float64.high
  for trial in 0 ..< 5:
    let t0 = getMonoTime()
    cb()
    let t1 = getMonoTime()
    let sec = float64((t1 - t0).inNanoseconds) / 1e9
    if sec < best:
      best = sec
  result = (gflops(best, ops), best * 1e6)

when isMainModule:
  randomize(42)
  echo "SIMD: avx_fma\n"

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
      C_ref = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
      C_fast = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
      C_tweak = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
    for i in 0 ..< A.len: A[i] = rand(1.0'f32)
    for i in 0 ..< B.len: B[i] = rand(1.0'f32)
    for i in 0 ..< C.len: C[i] = rand(1.0'f32)
    copyMem(addr C_ref[0], addr C[0], C.len * sizeof(float32).int)
    copyMem(addr C_fast[0], addr C[0], C.len * sizeof(float32).int)
    copyMem(addr C_tweak[0], addr C[0], C.len * sizeof(float32).int)

    echo "\n--- ", N, "x", N, " column-major ---"

    block:
      v_ref.gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C_ref, rs, cs)
      let (gf, us) = bestOf5(ops):
        v_ref.gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C_ref, rs, cs)
      echo "ref (ex02)     id ", N, "x", N, ": ", gf.formatFloat(ffDecimal, 2), " GFlop/s  (", us.formatFloat(ffDecimal, 1), " us)"

    block:
      v_fast.gemm_strided_non_algebra(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C_fast, rs, cs)
      let (gf, us) = bestOf5(ops):
        v_fast.gemm_strided_non_algebra(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C_fast, rs, cs)
      echo "fast (nonalg)  id ", N, "x", N, ": ", gf.formatFloat(ffDecimal, 2), " GFlop/s  (", us.formatFloat(ffDecimal, 1), " us)"

    block:
      v_tweak.gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C_tweak, rs, cs)
      let (gf, us) = bestOf5(ops):
        v_tweak.gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C_tweak, rs, cs)
      echo "tweaked       id ", N, "x", N, ": ", gf.formatFloat(ffDecimal, 2), " GFlop/s  (", us.formatFloat(ffDecimal, 1), " us)"

  echo "\nDone."
