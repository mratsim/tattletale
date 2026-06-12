{.experimental: "callOperator".}
## Benchmark: SIMD matmul — experiment harness
## Best-of-5, column-major (Fortran) layout.

import std/[monotimes, times, math, random, strutils]
import ../examples/ex02_matmul_simd/ex02_matmul_simd_experiment

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

proc gflops(elapsed: float64; ops: float64): float64 =
  (ops / 1e9) / elapsed

proc bestOf5(ops: float64; cb: proc(): void): (float64, float64) =
  ## Returns (best_gflops, best_us)
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
  var flags: seq[string]
  when defined(expNoAlign):          flags.add "expNoAlign"
  when defined(expNoRawEpilogue):    flags.add "expNoRawEpilogue"
  when defined(expNoExplicitPackA):  flags.add "expNoExplicitPackA"
  when defined(expNoExplicitPackB):  flags.add "expNoExplicitPackB"
  when defined(expNoExplicitInner):  flags.add "expNoExplicitInner"

  randomize(42)

  echo "SIMD: ", simdArchString()
  if flags.len > 0:
    echo "Flags: ", flags.join(", ")
  else:
    echo "Flags: (baseline — all fast opts enabled)"
  echo ""

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

    block:
      gemm_naive(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
      let (gf, us) = bestOf5(ops):
        gemm_naive(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
      echo "triple-loop        ", N, "x", N, ": ", gf.formatFloat(ffDecimal, 2), " GFlop/s  (", us.formatFloat(ffDecimal, 1), " us)"

    block:
      gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
      let (gf, us) = bestOf5(ops):
        gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
      echo "gemm_strided id    ", N, "x", N, ": ", gf.formatFloat(ffDecimal, 2), " GFlop/s  (", us.formatFloat(ffDecimal, 1), " us)"

  echo "\nDone."
