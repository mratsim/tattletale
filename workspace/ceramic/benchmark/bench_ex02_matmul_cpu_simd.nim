## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Benchmark: CPU GEMM — hand-tuned vs layout-algebra vs Laser vs OpenBLAS
##
## Compares ex02a (hand-tuned), ex02b (layout-algebra),
## Laser's reference gemm_strided, and optionally OpenBLAS/cblas
## on square float32 matrices at multiple sizes.
##
## Reports arithmetic intensity and % of theoretical peak at anchor frequencies.
##
## Usage:
##   nim cpp -r --outdir:build \
##     workspace/ceramic/benchmark/bench_ex02_matmul_cpu_simd.nim
##
## With OpenBLAS:
##   nim cpp -r -d:cblas --outdir:build \
##     workspace/ceramic/benchmark/bench_ex02_matmul_cpu_simd.nim

import std/[monotimes, times, math, random, strutils, strformat, sequtils, os]

import workspace/ceramic/examples/ex02a_matmul_handtuned as v_a
import workspace/ceramic/examples/ex02b_matmul_layout_algebra as v_b
import workspace/ceramic/benchmark/laser/gemm as laser_gemm
import workspace/ceramic/benchmark/bench_utils

const ProblemSizes = [128, 512, 1024, 1920]
const NbSamples = 10
const WarmupSamples = 3

# ═══════════════════════════════════════════════════════════════════════════
#  Optional cblas/OpenBLAS
# ═══════════════════════════════════════════════════════════════════════════

when defined(cblas):
  when defined(linux):
    const blasLib = "libopenblas.so"
  elif defined(macosx):
    const blasLib = "libopenblas.dylib"
  else:
    {.error: "OpenBLAS not configured for this platform".}
  type
    CblasOrder* {.size: sizeof(cint).} = enum
      CblasRowMajor = 101, CblasColMajor = 102
    CblasTranspose* {.size: sizeof(cint).} = enum
      CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113

  proc cblas_sgemm(
      Order: CblasOrder, TransA, TransB: CblasTranspose,
      M, N, K: cint; alpha: float32;
      A: ptr float32; lda: cint;
      B: ptr float32; ldb: cint;
      beta: float32; C: ptr float32; ldc: cint
    ) {.dynlib: blasLib, importc: "cblas_sgemm".}

# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark helpers
# ═══════════════════════════════════════════════════════════════════════════

proc gflops(elapsed: float64; ops: float64): float64 =
  (ops / 1e9) / elapsed


proc bench(name, label: string;
           run: proc();
           ops, ai: float64): tuple[name, label: string; gflops, medianUs, ai: float64] =
  # Run warmup samples (discarded) then timed samples.
  for _ in 0 ..< WarmupSamples:
    run()

  var times = newSeq[float64](NbSamples)
  for s in 0 ..< NbSamples:
    let t0 = getMonoTime()
    run()
    let t1 = getMonoTime()
    times[s] = float64((t1 - t0).inNanoseconds) / 1e9

  let med = median(times)
  (name, label, gflops(med, ops), med * 1e6, ai)

# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

when isMainModule:
  randomize(42)

  # ── Header ──
  echo "=".repeat(72)
  echo &"  GEMM benchmark — float32, row-major, single-threaded"
  echo &"  SIMD arch: {v_a.simdArchString()}"
  echo "=".repeat(72)
  echo ""
  # ── Theoretical peak table ──
  printPeakTable()

  when defined(cblas):
    let ompOK = getEnv("OMP_NUM_THREADS") == "1"
    let openblasOK = getEnv("OPENBLAS_NUM_THREADS") == "1"
    let mklOK = getEnv("MKL_NUM_THREADS") == "1"
    let blisOK = getEnv("BLIS_NUM_THREADS") == "1"
    let anyOK = ompOK or openblasOK or mklOK or blisOK
    echo "  OpenBLAS: ENABLED"
    if not anyOK:
      echo "  ⚠  None of OMP_NUM_THREADS, OPENBLAS_NUM_THREADS,"
      echo "      MKL_NUM_THREADS, or BLIS_NUM_THREADS is set to 1."
      echo "      Results may include multi-threaded overhead."
  else:
    echo "  OpenBLAS: disabled (pass -d:cblas to enable)"
  echo ""

  # ── CPU frequency warmup ──
  block:
    echo "  Warming up CPU..."
    var foo = 123
    let tStart = epochTime()
    for i in 0 ..< 300_000_000:
      foo += i*i mod 456
      foo = foo mod 789
    let tEnd = epochTime()
    echo &"  Warmup: {tEnd - tStart:>4.3f}s (foo={foo})"
  echo ""

  # ── Results per size ──
  for N in ProblemSizes:
    let
      aShape: MatrixShape = (M: N, N: N)
      bShape: MatrixShape = (M: N, N: N)
      ops    = float64(gemm_required_ops(aShape, bShape))
      bytes  = float64(gemm_required_data(aShape, bShape) * 4)  # float32 = 4 bytes
      ai     = ops / bytes
      alpha  = 1.0'f32
      beta   = 1.0'f32
      rs     = N
      cs     = 1

    # Allocate
    var
      A = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
      B = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
      C0 = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
    for i in 0 ..< A.len: A[i] = rand(1.0'f32)
    for i in 0 ..< B.len: B[i] = rand(1.0'f32)

    # ── Correctness check: allClose vs Laser reference ──
    block:
      var C_l = newSeq[float32](C0.len)
      var C_a = newSeq[float32](C0.len)
      var C_b = newSeq[float32](C0.len)
      for i in 0 ..< C0.len:
        C_l[i] = C0[i]; C_a[i] = C0[i]; C_b[i] = C0[i]
      # Reference: Laser (battle-tested)
      laser_gemm.gemm_strided(N, N, N, 1.0'f32, addr A[0], rs, cs, addr B[0], rs, cs, 0.0'f32, addr C_l[0], rs, cs)
      v_a.gemm_strided(N, N, N, 1.0'f32, A, rs, cs, B, rs, cs, 0.0'f32, C_a, rs, cs)
      v_b.gemm_strided(N, N, N, 1.0'f32, A, rs, cs, B, rs, cs, 0.0'f32, C_b, rs, cs)
      let r_a = allClose(C_a, C_l)  # ex02a must match Laser exactly (same ukernel path)
      let r_b = allClose(C_b, C_l, atol = 1.0, rtol = 1.0)  # ex02b: different ukernel variant
      doAssert r_a.ok, &"ex02a vs Laser: maxAbsErr={r_a.maxAbsErr:.2e} maxRelErr={r_a.maxRelErr:.2e}"
      doAssert r_b.ok, &"ex02b vs Laser: maxAbsErr={r_b.maxAbsErr:.2e} maxRelErr={r_b.maxRelErr:.2e}"
      when defined(cblas):
        var C_o = newSeq[float32](C0.len)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          cint(N), cint(N), cint(N), 1.0'f32,
          addr A[0], cint(N), addr B[0], cint(N),
          0.0'f32, addr C_o[0], cint(N))
        let r_o = allClose(C_o, C_l, atol = 1.0, rtol = 1.0)
        doAssert r_o.ok, &"OpenBLAS vs Laser: maxAbsErr={r_o.maxAbsErr:.2e} maxRelErr={r_o.maxRelErr:.2e}"
        echo &"  Diff vs Laser ref  a={r_a.maxAbsErr:.2e}/{r_a.maxRelErr:.2e}  b={r_b.maxAbsErr:.2e}/{r_b.maxRelErr:.2e}  o={r_o.maxAbsErr:.2e}/{r_o.maxRelErr:.2e}"
      else:
        echo &"  Diff vs Laser ref  a={r_a.maxAbsErr:.2e}/{r_a.maxRelErr:.2e}  b={r_b.maxAbsErr:.2e}/{r_b.maxRelErr:.2e}  o=N/A"

    var results: seq[tuple[name, label: string; gflops, medianUs, ai: float64]] = @[]

    # ── ex02a — hand-tuned ──
    block:
      var C = C0
      proc run() = v_a.gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
      results.add bench("ex02a", "hand-tuned", run, ops, ai)

    # ── ex02b — layout algebra ──
    block:
      var C = C0
      proc run() = v_b.gemm_strided(N, N, N, alpha, A, rs, cs, B, rs, cs, beta, C, rs, cs)
      results.add bench("ex02b", "layout-algebra", run, ops, ai)
    # ── Laser ──
    block:
      var C = C0
      proc run() = laser_gemm.gemm_strided(N, N, N, alpha, addr A[0], rs, cs, addr B[0], rs, cs, beta, addr C[0], rs, cs)
      results.add bench("Laser", "reference", run, ops, ai)

    # ── OpenBLAS/cblas ──
    when defined(cblas):
      block:
        var C = C0
        proc run() =
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            cint(N), cint(N), cint(N), 1.0'f32,
            addr A[0], cint(N), addr B[0], cint(N),
            0.0'f32, addr C[0], cint(N))
        results.add bench("OpenBLAS", "cblas", run, ops, ai)

    # ── Print results ──
    let peaksAVX = [theoreticalPeak(archAVX_FMA, 4.0),
                    theoreticalPeak(archAVX_FMA, 4.5),
                    theoreticalPeak(archAVX_FMA, 5.0),
                    theoreticalPeak(archAVX_FMA, 5.5)]
    let peaksAVX512 = [theoreticalPeak(archAVX512, 4.0),
                       theoreticalPeak(archAVX512, 4.5),
                       theoreticalPeak(archAVX512, 5.0),
                       theoreticalPeak(archAVX512, 5.5)]

    # Header (two-line: freqs below arch label)
    echo "  Variant" & spaces(19) & "GFLOP/s".align(6) & "        AVX+FMA % @ GHz  " & "      AVX-512 % @ GHz" & "       μs"
    echo spaces(40) & " 4.0  4.5  5.0  5.5" & spaces(4) & " 4.0  4.5  5.0  5.5"
    echo "  " & "-".repeat(88)

    for r in results:
      let label = r.name & "/" & r.label
      let p1 = int(r.gflops / peaksAVX[0] * 100)
      let p2 = int(r.gflops / peaksAVX[1] * 100)
      let p3 = int(r.gflops / peaksAVX[2] * 100)
      let p4 = int(r.gflops / peaksAVX[3] * 100)
      let q1 = int(r.gflops / peaksAVX512[0] * 100)
      let q2 = int(r.gflops / peaksAVX512[1] * 100)
      let q3 = int(r.gflops / peaksAVX512[2] * 100)
      let q4 = int(r.gflops / peaksAVX512[3] * 100)
      echo &"  {label:<21} {r.gflops:>8.2f}        {p1:>3d}% {p2:>3d}% {p3:>3d}% {p4:>3d}%    {q1:>3d}% {q2:>3d}% {q3:>3d}% {q4:>3d}%  {int(r.medianUs):>6d}"

    echo ""

  echo "Done."
