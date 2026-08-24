## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Benchmark: Apple GPU tile-layer GEMM — fp16 in, fp32 accumulate
##
## Measures the tile `gemm` kernel (one 32×32 output tile per threadgroup,
## 16-wide K staging) on square fp16 matrices across sizes. Reports
## GFLOP/s, % of fp16 theoretical peak at anchor clocks, median dispatch
## time, and bit-exactness vs the fp32 host reference on sampled output
## positions (tolerance 0.0).
##
## Timing is host-side end-to-end (buffer upload, encode, dispatch, wait,
## readback) — the harness's run contract. A K=16 overhead probe with the
## same buffers separates the per-run copy/dispatch cost from the kernel
## compute (est. compute = full run − overhead).
##
## Usage:
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/bench_matmul_apple_gpu \
##     --nimcache:nimcache/bench_matmul_apple_gpu \
##     workspace/ceramic/benchmark/bench_matmul_apple_gpu.nim
##
## Unrecognized GPUs report no % of peak; pass -d:gpuCores=N to enable.

import std/[monotimes, times, math, random, strutils, strformat]

import workspace/crucible
import ./bench_utils
import ../tests/tile_test_utils
import ../src/kernels/k_tile_gemm

# ═════════════════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════════════════

const
  ProblemSizes = [512, 1024, 2048, 4096]
  NbSamples = 10
  WarmupSamples = 3
  SamplePositions = 1024

const gemmKernel = "fusedGemm"

const gemmMsl = metal:
  proc fusedGemm(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[uint16],
                 N, K, M: int32) {.global.} =
    gemm(D, N, M, K, A, K, 1, B, M, 1)

# ═════════════════════════════════════════════════════════════════════════
#  Theoretical peak (Apple GPU)
# ═════════════════════════════════════════════════════════════════════════
#
#  Formula: cores × 128 FMA/clk × 2 FLOP/FMA × freq (fp32, matches Apple
#  marketing: M2 Max 38 cores → 13.6 TFLOPS; M4 Max → ~17 @ 1.65 GHz).
#  fp16 = 2× fp32 (packed fp16x2 ALU rate). Note: the 8×8×8 simdgroup mma
#  with fp32 accumulate measures only ~1.1× fp32 in practice (MLX: fp32
#  13.1, fp16 14.7 TFLOPS @ 4096³), so % of fp16 peak understates the
#  achievable fraction of the matrix pipe.

const AnchorGpuFreqs = [1.4, 1.5, 1.6, 1.7]   # GHz

func fp16Peak(cores: int; freq: float64): float64 =
  ## GFLOP/s: cores × 128 FMA/clk × 2 (FMA = 2 FLOP) × 2 (fp16x2) × freq.
  float64(cores) * 128.0 * 2.0 * 2.0 * freq

func fp32Peak(cores: int; freq: float64): float64 =
  ## GFLOP/s: cores × 128 FMA/clk × 2 (FMA = 2 FLOP) × freq.
  float64(cores) * 128.0 * 2.0 * freq

func gpuCoreCount(deviceName: string): int =
  ## Known Apple GPU core counts, max variant per family.
  ## Returns 0 when unknown (no % of peak reported).
  if "M1 Ultra" in deviceName: 64
  elif "M2 Ultra" in deviceName: 76
  elif "M3 Ultra" in deviceName: 80
  elif "M4 Max" in deviceName: 40
  elif "M3 Max" in deviceName: 40
  elif "M2 Max" in deviceName: 38
  elif "M1 Max" in deviceName: 32
  elif "M4 Pro" in deviceName: 20
  elif "M3 Pro" in deviceName: 18
  elif "M2 Pro" in deviceName: 19
  elif "M1 Pro" in deviceName: 16
  elif "M4" in deviceName: 10
  elif "M3" in deviceName: 10
  elif "M2" in deviceName: 10
  elif "M1" in deviceName: 8
  else: 0

proc detectCores(engine: auto): int =
  when defined(gpuCores):
    gpuCores
  else:
    gpuCoreCount(engine.deviceName())

proc printPeakTable(cores: int) =
  echo "  Theoretical peak:"
  echo "  " & spaces(4) & "1.4 GHz".align(9) & "1.5 GHz".align(9) & "1.6 GHz".align(9) & "1.7 GHz".align(9)
  echo "  " & "-".repeat(46)
  if cores > 0:
    echo &"  fp16     {int(fp16Peak(cores, 1.4)):>5d} {int(fp16Peak(cores, 1.5)):>5d} {int(fp16Peak(cores, 1.6)):>5d} {int(fp16Peak(cores, 1.7)):>5d} GFLOP/s"
    echo &"  fp32     {int(fp32Peak(cores, 1.4)):>5d} {int(fp32Peak(cores, 1.5)):>5d} {int(fp32Peak(cores, 1.6)):>5d} {int(fp32Peak(cores, 1.7)):>5d} GFLOP/s"
    echo &"  {cores} cores"
  else:
    echo "  unknown core count — % of peak not reported (pass -d:gpuCores=N)"
  echo "  Formula: cores × 128 FMA/clk × 2 FLOP/FMA × freq; fp16 = 2× fp32"
  echo "  Note: the fp16x2 2× is the ALU rate; the simdgroup mma with fp32"
  echo "        accumulate sustains only ~1.1× fp32 (measured: measured fp32 13.1,"
  echo "        fp16 14.7 TFLOPS @ 4096³) — % of fp16 peak is conservative"
  echo ""

# ═════════════════════════════════════════════════════════════════════════
#  Benchmark helpers
# ═════════════════════════════════════════════════════════════════════════

proc bench(run: proc(); ops: float64): tuple[gflops, medianUs: float64] =
  ## Warmup runs (discarded) then timed samples, median of the samples.
  for _ in 0 ..< WarmupSamples:
    run()

  var times = newSeq[float64](NbSamples)
  for s in 0 ..< NbSamples:
    let t0 = getMonoTime()
    run()
    let t1 = getMonoTime()
    times[s] = float64((t1 - t0).inNanoseconds) / 1e9

  let med = median(times)
  ((ops / 1e9) / med, med * 1e6)

proc checkBitExact(M, N, K, Mp, Np, Kp: int; Ah, Bh: seq[uint16]; C: seq[float32]): tuple[maxAbs, maxRel: float32] =
  ## Bit-exactness on SamplePositions sampled (m, n) outputs.
  ## The reference is the sequential fp32 k-sum: fp16→fp32 is exact and
  ## fp16 products are exact in fp32, so it is the exact-in-fp32 result,
  ## which the kernel's k-ordered fragment accumulation must match.
  var rng = initRand(0xC0FFEE)
  result = (0.0'f32, 0.0'f32)
  for s in 0 ..< SamplePositions:
    let m = rng.rand(M - 1)
    let n = rng.rand(N - 1)
    var acc = 0.0'f32
    for k in 0 ..< K:
      acc += fp16ToFp32(Ah[m * Kp + k]) * fp16ToFp32(Bh[k * Np + n])
    let got = C[m * Np + n]
    let d = abs(got - acc)
    if d > result.maxAbs: result.maxAbs = d
    let rel = d / max(abs(got), max(abs(acc), 1e-12'f32))
    if rel > result.maxRel: result.maxRel = rel
  doAssert result.maxAbs == 0.0'f32,
    &"not bit-exact: worst |Δ| = {result.maxAbs}"

# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

proc runBench() =   # engines are RAII, so keep them function-local
  randomize(42)
  var engine = bkMetal.init()
  engine.ingest(gemmMsl)
  let cores = detectCores(engine)

  # ── Header ──
  echo "=".repeat(72)
  echo &"  GEMM benchmark — Apple GPU tile layer (fp16 in, fp32 acc)"
  echo &"  Device: {engine.deviceName()}"
  echo &"  Tiles: 32×32 output per threadgroup (32 lanes), K-block 16"
  echo &"  Timing: host-side end-to-end; K=16 probe separates copy overhead"
  echo "=".repeat(72)
  echo ""
  printPeakTable(cores)

  # ── Table header ──
  echo "  Size" & spaces(17) & "GFLOP/s" & "          fp16 peak % @ GHz" & "       μs" & "   absdiff    reldiff"
  echo spaces(30) & "1.4  1.5  1.6  1.7"
  echo "  " & "-".repeat(86)

  for N in ProblemSizes:
    let
      Mp = N
      Np = N
      Kp = N
      ops = float64(2 * N * N * N)

    var Ah = newSeq[uint16](Mp * Kp)
    var Bh = newSeq[uint16](Kp * Np)
    for i in 0 ..< Ah.len: Ah[i] = fp32ToFp16(rand(1.0'f32))
    for i in 0 ..< Bh.len: Bh[i] = fp32ToFp16(rand(1.0'f32))
    var C = newSeq[float32](Mp * Np)

    # Raw-pointer views into the host buffers (the pointer API, no
    # seq/array marshalling). Ah/Bh/C live until the next size, so
    # the views stay valid for the whole size iteration.
    var aArg = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](addr Ah[0]), len: Ah.len, off: 0)
    var bArg = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](addr Bh[0]), len: Bh.len, off: 0)
    var cArg = PtrArg[float32](buf: cast[ptr UncheckedArray[float32]](addr C[0]), len: C.len, off: 0)

    # Correctness on a fresh run, then warmup + timed samples on the same buffers.
    engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
      (gemmKernel, cArg, (aArg, bArg, int32(Mp), int32(Kp), int32(Np)))
    let chk = checkBitExact(N, N, N, Mp, Np, Kp, Ah, Bh, C)

    block:
      proc run() =
        engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
          (gemmKernel, cArg, (aArg, bArg, int32(Mp), int32(Kp), int32(Np)))
      let r = bench(run, ops)
      let pctStr =
        if cores > 0:
          let peaks = [fp16Peak(cores, 1.4), fp16Peak(cores, 1.5),
                       fp16Peak(cores, 1.6), fp16Peak(cores, 1.7)]
          &"{int(r.gflops / peaks[0] * 100):>3d}% {int(r.gflops / peaks[1] * 100):>3d}% " &
           &"{int(r.gflops / peaks[2] * 100):>3d}% {int(r.gflops / peaks[3] * 100):>3d}%"
        else:
          "  —    —    —    —"
      let label = &"{N}³"
      echo &"  {label:<21} {r.gflops:>8.2f}  {pctStr}  {int(r.medianUs):>6d}  " &
           &"{formatFloat(chk.maxAbs, ffScientific, 1):>9} {formatFloat(chk.maxRel, ffScientific, 1):>9}"
      echo &"         grid ({Np div 32},{Mp div 32}), K-loop {Kp div 16} iters, " &
           &"AI {ops / float64((Mp * Kp + Kp * Np) * 2):.1f} FLOP/B"

      # Harness overhead probe: same buffers, K = 16 (one k-block, ~zero
      # compute). The per-run upload + readback + dispatch cost at identical
      # buffer sizes, so est. compute = full run − overhead.
      proc runOverhead() =
        engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
          (gemmKernel, cArg, (aArg, bArg, int32(Mp), int32(16), int32(Np)))
      let o = bench(runOverhead, 1.0)
      let computeUs = r.medianUs - o.medianUs
      echo &"         est. compute {computeUs:.1f} μs (full {r.medianUs:.0f} − " &
           &"overhead {o.medianUs:.0f}) → {ops / (computeUs * 1e6):.1f} TFLOPS"
    echo ""

  echo "Done."

when isMainModule:
  runBench()
