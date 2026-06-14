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

# ═══════════════════════════════════════════════════════════════════════════
#  Matrix shape helpers (Laser convention)
# ═══════════════════════════════════════════════════════════════════════════

type
  MatrixShape* = tuple[M, N: int]
  Matrix*[T] = seq[T]

func gemm_out_shape*(a, b: MatrixShape): MatrixShape =
  doAssert a.N == b.M
  result.M = a.M
  result.N = b.N

func gemm_required_ops*(a, b: MatrixShape): int =
  doAssert a.N == b.M
  result = a.M * a.N * b.N * 2   # 1 mul + 1 add per element

func gemm_required_data*(a, b: MatrixShape): int =
  doAssert a.N == b.M
  result = a.M * a.N + b.M * b.N

# ═══════════════════════════════════════════════════════════════════════════
#  Problem sizes
# ═══════════════════════════════════════════════════════════════════════════

const ProblemSizes = [128, 512, 1024, 1920]
const NbSamples = 10
const WarmupSamples = 3

# ═══════════════════════════════════════════════════════════════════════════
#  Theoretical peak (float32, single-core)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Formula: freq(GHz) × vectorWidth × instrPerCycle × FLOP/instr
#
#  AVX+FMA: 8 floats × 2 FMA/cycle × 2 FLOP/FMA = 32 FLOP/cycle
#  AVX-512: 16 floats × 2 FMA/cycle × 2 FLOP/FMA = 64 FLOP/cycle

type CpuArch = enum
  archAVX_FMA
  archAVX512

const AnchorFreqs = [4.0, 4.5, 5.0, 5.5]

func theoreticalPeak(arch: CpuArch; freq: float64): float64 =
  let vecWidth = case arch
    of archAVX_FMA:  8.0
    of archAVX512:  16.0
  let instrCycle = 2.0    # 2 FMAs per cycle (Intel: ports 0 & 1)
  let flopInstr  = 2.0    # FMA = 1 mul + 1 add
  freq * vecWidth * instrCycle * flopInstr

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

proc median(v: openArray[float64]): float64 =
  let n = v.len
  if n == 0: return 0.0
  if n mod 2 == 1: v[n div 2]
  else: (v[n div 2 - 1] + v[n div 2]) * 0.5

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
  echo &"  GEMM benchmark — float32, column-major, single-threaded"
  echo &"  SIMD arch: {v_a.simdArchString()}"
  echo "=".repeat(72)
  echo ""

  # ── Theoretical peak table ──
  echo "  Theoretical peak (float32, single-core):"
  echo "  " & spaces(7) & "4.0 GHz".align(9) & "4.5 GHz".align(9) & "5.0 GHz".align(9) & "5.5 GHz".align(9)
  echo "  " & "-".repeat(46)
  proc showPeak(arch: CpuArch) =
    let name = if arch == archAVX_FMA: "AVX+FMA" else: "AVX-512"
    echo &"  {name:<8} {int(theoreticalPeak(arch, 4.0)):>5d}    {int(theoreticalPeak(arch, 4.5)):>5d}    {int(theoreticalPeak(arch, 5.0)):>5d}    {int(theoreticalPeak(arch, 5.5)):>5d}    GFLOP/s"
  showPeak(archAVX_FMA)
  showPeak(archAVX512)
  echo ""
  echo &"  Formula: freq × vecWidth × 2 FMA/cycle × 2 FLOP/FMA"
  echo &"    AVX+FMA: 8 floats  — peak = freq × 32"
  echo &"    AVX-512: 16 floats — peak = freq × 64"
  echo ""
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
      rs     = 1
      cs     = N

    # Allocate
    var
      A = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
      B = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
      C0 = newSeq[float32]((N-1)*rs + (N-1)*cs + 1)
    for i in 0 ..< A.len: A[i] = rand(1.0'f32)
    for i in 0 ..< B.len: B[i] = rand(1.0'f32)

    echo &"--- {N}x{N} (col-major) | AI = {ai:>6.2f} FLOP/byte | {ops/1e9:>8.3f} GFLOP ---"
    echo ""

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
          cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
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
