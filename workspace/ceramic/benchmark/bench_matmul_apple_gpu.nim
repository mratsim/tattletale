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
## Timing is host-side end-to-end: the timed region is one full `engine.run` call
## (bind, encode, dispatch, wait). Device args live in page-aligned host buffers
## that the runtime binds no-copy, so the timed region holds no upload and no
## readback copy. A K=16 overhead probe on the same buffers separates the per-run
## dispatch cost from the kernel compute, giving est. compute = full run − overhead.
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

proc posix_memalign(memptr: ptr pointer, alignment: csize_t, size: csize_t): cint
  {.importc: "posix_memalign", header: "<stdlib.h>".}

proc getpagesize(): cint {.importc: "getpagesize", header: "<unistd.h>".}

func roundPages(nbytes: int, ps = getpagesize()): int =
  ## Byte length rounded up to a host-page multiple, 16 KiB on Apple Silicon.
  ## The device-visible length must be a page multiple for a no-copy binding.
  (nbytes + ps - 1) div ps * ps

proc allocAligned(nbytes: int): pointer =
  ## Allocation that is page-aligned and has a byte length that is a page multiple,
  ## the two requirements of no-copy binding. Any other length gets an allocated
  ## buffer and a copy.
  let ps = getpagesize()
  var p: pointer = nil
  doAssert posix_memalign(addr p, csize_t(ps), csize_t(roundPages(nbytes, ps))) == 0,
    "posix_memalign failed for " & $nbytes & " bytes"
  p

# ═════════════════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════════════════

const
  ProblemSizes = [512, 1024, 2048, 4096]
  NbSamples = 10
  WarmupSamples = 3
  SamplePositions = 1024

const gemmKernel = "fusedGemm"
const linearKernel = "fusedLinear"
const gemmStridedKernel = "fusedGemmStrided"

# One metal block = one library: ingest replaces the previous artifact,
# so this single source holds every bench kernel.
const gemmMsl = metal:
  proc fusedGemm(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                 N, K, M: int32) {.global.} =
    matmul(D, A, B, N, K, M)

  proc fusedLinear(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                   Bias: ptr UncheckedArray[float32], N, K, M: int32) {.global.} =
    linear(D, A, B, Bias, N, K, M)

  proc fusedGemmStrided(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                        C: ptr UncheckedArray[float32],
                        M, N, K: int32, alpha: float32,
                        rsa, csa, rsb, csb: int32, beta: float32,
                        rsc, csc: int32) {.global.} =
    gemm(D, M, N, K, alpha, A, rsa, csa, B, rsb, csb, beta, C, rsc, csc)

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

func biasVal(n: int): float32 =
  ## Deterministic per-column bias mix, negative on every third column.
  (if n mod 3 == 0: -float32(1 + 2 * n) else: float32(1 + 2 * n))

proc checkBitExactBias(M, N, K, Mp, Np, Kp: int, Ah, Bh: ptr UncheckedArray[uint16],
                       Bias: ptr UncheckedArray[float32],
                       C: ptr UncheckedArray[float32]): tuple[maxAbs, maxRel: float32] =
  ## Bit-exactness on SamplePositions sampled (m, n) outputs, D = A·B + bias.
  ## Reference: fp32 k-sum plus the fp32 bias add, exact in fp32,
  ## which the kernel's k-ordered accumulation must match.
  var rng = initRand(0xC0FFEE)
  result = (0.0'f32, 0.0'f32)
  for s in 0 ..< SamplePositions:
    let m = rng.rand(M - 1)
    let n = rng.rand(N - 1)
    var acc = 0.0'f32
    for k in 0 ..< K:
      acc += fp16ToFp32(Ah[m * Kp + k]) * fp16ToFp32(Bh[k * Np + n])
    let got = C[m * Np + n]
    let d = abs(got - (acc + Bias[n]))
    if d > result.maxAbs: result.maxAbs = d
    let rel = d / max(abs(got), max(abs(acc + Bias[n]), 1e-12'f32))
    if rel > result.maxRel: result.maxRel = rel
  doAssert result.maxAbs == 0.0'f32,
    &"not bit-exact: worst |Δ| = {result.maxAbs}"

proc checkBitExactAxpy(M, N, K, Mp, Np, Kp: int, Ah, Bh: ptr UncheckedArray[uint16],
                       Ch: ptr UncheckedArray[float32],
                       C: ptr UncheckedArray[float32]): tuple[maxAbs, maxRel: float32] =
  ## Bit-exactness on SamplePositions sampled (m, n) outputs, D = A·B + C.
  ## Reference: fp32 k-sum plus the fp32 C add (α = β = 1, exact in fp32).
  var rng = initRand(0xC0FFEE)
  result = (0.0'f32, 0.0'f32)
  for s in 0 ..< SamplePositions:
    let m = rng.rand(M - 1)
    let n = rng.rand(N - 1)
    var acc = 0.0'f32
    for k in 0 ..< K:
      acc += fp16ToFp32(Ah[m * Kp + k]) * fp16ToFp32(Bh[k * Np + n])
    let got = C[m * Np + n]
    let d = abs(got - (acc + Ch[m * Np + n]))
    if d > result.maxAbs: result.maxAbs = d
    let rel = d / max(abs(got), max(abs(acc + Ch[m * Np + n]), 1e-12'f32))
    if rel > result.maxRel: result.maxRel = rel
  doAssert result.maxAbs == 0.0'f32,
    &"not bit-exact: worst |Δ| = {result.maxAbs}"

proc checkBitExact(M, N, K, Mp, Np, Kp: int, Ah, Bh: ptr UncheckedArray[uint16],
                   C: ptr UncheckedArray[float32]): tuple[maxAbs, maxRel: float32] =
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
  echo &"  Timing: host-side end-to-end, no-copy args, K=16 probe separates dispatch overhead"
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

    var Ah = cast[ptr UncheckedArray[uint16]](allocAligned(Mp * Kp * sizeof(uint16)))
    var Bh = cast[ptr UncheckedArray[uint16]](allocAligned(Kp * Np * sizeof(uint16)))
    for i in 0 ..< Mp * Kp: Ah[i] = fp32ToFp16(rand(1.0'f32))
    for i in 0 ..< Kp * Np: Bh[i] = fp32ToFp16(rand(1.0'f32))
    var C = cast[ptr UncheckedArray[float32]](allocAligned(Mp * Np * sizeof(float32)))

    # Raw-pointer views into the page-aligned host buffers, the pointer API with no
    # seq/array marshalling. Ah/Bh/C live until the next size, so the views stay
    # valid for the whole size iteration.
    var aArg = PtrArg[uint16](buf: Ah, len: Mp * Kp, off: 0)
    var bArg = PtrArg[uint16](buf: Bh, len: Kp * Np, off: 0)
    var cArg = PtrArg[float32](buf: C, len: Mp * Np, off: 0)

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

      # K=16 overhead probe: same buffers at K = 16, one k-block and ~zero compute.
      # Measures the per-run bind + encode/commit/wait + dispatch cost, giving est.
      # compute = full run − overhead.
      proc runOverhead() =
        engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
          (gemmKernel, cArg, (aArg, bArg, int32(Mp), int32(16), int32(Np)))
      let o = bench(runOverhead, 1.0)
      let computeUs = r.medianUs - o.medianUs
      echo &"         est. compute {computeUs:.1f} μs (full {r.medianUs:.0f} − " &
           &"overhead {o.medianUs:.0f}) → {ops / (computeUs * 1e6):.1f} TFLOPS"
    echo ""

  # ── Linear (bias fused) and strided-C gemm paths ──
  # Same fused core with the bias / α·A·B+β·C epilogues. The strided kernel
  # takes runtime row/col strides for C (row-major here).
  echo "=".repeat(72)
  echo "  Linear (bias fused) and strided-C gemm — same fused core"
  echo "=".repeat(72)
  echo ""

  for N in ProblemSizes:
    let
      Mp = N
      Np = N
      Kp = N
      ops = float64(2 * N * N * N)

    var Ah = cast[ptr UncheckedArray[uint16]](allocAligned(Mp * Kp * sizeof(uint16)))
    var Bh = cast[ptr UncheckedArray[uint16]](allocAligned(Kp * Np * sizeof(uint16)))
    for i in 0 ..< Mp * Kp: Ah[i] = fp32ToFp16(rand(1.0'f32))
    for i in 0 ..< Kp * Np: Bh[i] = fp32ToFp16(rand(1.0'f32))
    var Ch = cast[ptr UncheckedArray[float32]](allocAligned(Mp * Np * sizeof(float32)))
    for i in 0 ..< Mp * Np: Ch[i] = rand(1.0'f32)
    var Bias = cast[ptr UncheckedArray[float32]](allocAligned(Np * sizeof(float32)))
    for n in 0 ..< Np: Bias[n] = biasVal(n)
    var D = cast[ptr UncheckedArray[float32]](allocAligned(Mp * Np * sizeof(float32)))

    var aArg = PtrArg[uint16](buf: Ah, len: Mp * Kp, off: 0)
    var bArg = PtrArg[uint16](buf: Bh, len: Kp * Np, off: 0)
    var dArg = PtrArg[float32](buf: D, len: Mp * Np, off: 0)
    var cArg = PtrArg[float32](buf: Ch, len: Mp * Np, off: 0)
    # Bias is the only arg whose byte length can land inside a single page
    # (16 KiB = 4096 fp32 lanes). A length that is not a page multiple would take
    # the allocate-and-copy path on every dispatch, so `len` is the page-rounded
    # length, in-bounds because allocAligned rounds the allocation up too. Kernel
    # code reads the `Np` named lanes, never the padding.
    var biasArg = PtrArg[float32](
      buf: Bias, len: roundPages(Np * sizeof(float32)) div sizeof(float32), off: 0)

    # ── linear (D = A·B + bias) ──
    engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
      (linearKernel, dArg, (aArg, bArg, biasArg, int32(Mp), int32(Kp), int32(Np)))
    block:
      let chk = checkBitExactBias(N, N, N, Mp, Np, Kp, Ah, Bh, Bias, D)
      proc run() =
        engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
          (linearKernel, dArg, (aArg, bArg, biasArg, int32(Mp), int32(Kp), int32(Np)))
      let r = bench(run, ops)
      let label = &"{N}³"
      echo &"  linear  {label:<21} {r.gflops:>8.2f}  {int(r.medianUs):>6d} μs  " &
           &"{formatFloat(chk.maxAbs, ffScientific, 1):>9} {formatFloat(chk.maxRel, ffScientific, 1):>9}"

    # ── gemm strided-C (D = α·A·B + β·C, row-major C) ──
    engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
      (gemmStridedKernel, dArg,
       (aArg, bArg, cArg, int32(Mp), int32(Np), int32(Kp), 1.0'f32,
        int32(Kp), int32(1), int32(Np), int32(1), 0.0'f32, int32(Np), int32(1)))
    block:
      # β = 0 probe: the epilogue skips the C read (EpiAXPBYStrided),
      # D = A·B, isolating the C-read cost vs the plain matmul core.
      let chk = checkBitExact(N, N, N, Mp, Np, Kp, Ah, Bh, D)
      proc run() =
        engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
          (gemmStridedKernel, dArg,
           (aArg, bArg, cArg, int32(Mp), int32(Np), int32(Kp), 1.0'f32,
            int32(Kp), int32(1), int32(Np), int32(1), 0.0'f32, int32(Np), int32(1)))
      let r = bench(run, ops)
      let label = &"{N}³"
      echo &"  strided-β0 {label:<18} {r.gflops:>8.2f}  {int(r.medianUs):>6d} μs  " &
           &"{formatFloat(chk.maxAbs, ffScientific, 1):>9} {formatFloat(chk.maxRel, ffScientific, 1):>9}"

    engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
      (gemmStridedKernel, dArg,
       (aArg, bArg, cArg, int32(Mp), int32(Np), int32(Kp), 1.0'f32,
        int32(Kp), int32(1), int32(Np), int32(1), 1.0'f32, int32(Np), int32(1)))
    block:
      let chk = checkBitExactAxpy(N, N, N, Mp, Np, Kp, Ah, Bh, Ch, D)
      proc run() =
        engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >>
          (gemmStridedKernel, dArg,
           (aArg, bArg, cArg, int32(Mp), int32(Np), int32(Kp), 1.0'f32,
            int32(Kp), int32(1), int32(Np), int32(1), 1.0'f32, int32(Np), int32(1)))
      let r = bench(run, ops)
      let label = &"{N}³"
      echo &"  strided {label:<21} {r.gflops:>8.2f}  {int(r.medianUs):>6d} μs  " &
           &"{formatFloat(chk.maxAbs, ffScientific, 1):>9} {formatFloat(chk.maxRel, ffScientific, 1):>9}"
    echo ""

  echo "Done."

when isMainModule:
  runBench()
