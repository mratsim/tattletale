## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## POC benchmark: ARM SME2 GEMM (ex02a_matmul_handtuned_arm64_sme2) vs a scalar reference.
## Square float32 matrices, one warmup plus three timed samples per size, total
## runtime ≤ 5 s. Each variant is checked against the naive reference before timing.
##
## The "generic" baseline is the shared naive triple-loop `gemm_reference` from
## workspace/ceramic/benchmark/bench_utils. The x86 ex02a (which falls back to the
## generic ukernel on non-x86) cannot compile on arm64:
## workspace/cpuplatforms/x86/simd_x86.nim defines builtin_prefetch only under
## `defined(i386) or defined(amd64)`. The naive baseline is timed for N < 512 only.
##
## Usage:
##   nim cpp -r -d:release --hints:off --warnings:off \
##     --outdir:build/wip/arm-sme-matmul --nimcache:nimcache/wip/arm-sme-matmul \
##     workspace/ceramic/benchmark/bench_ex02_matmul_sme.nim

import std/[algorithm, monotimes, random, strutils, strformat]

import workspace/ceramic/examples/ex02a_matmul_handtuned_arm64_sme2 as v_a
import workspace/ceramic/benchmark/bench_utils

const
  ProblemSizes = [128, 256, 512]
  NbSamples = 3
  WarmupSamples = 1
  Tol = 1e-4'f32

proc gflops(elapsed: float64; ops: float64): float64 =
  (ops / 1e9) / elapsed

proc bench(name: string; run: proc(); ops: float64): tuple[name: string; gflops, medianUs: float64] =
  ## Runs one warmup pass (discarded), then the timed samples, and reports the median.
  for _ in 0 ..< WarmupSamples:
    run()
  var times = newSeq[float64](NbSamples)
  for s in 0 ..< NbSamples:
    let t0 = getMonoTime()
    run()
    let t1 = getMonoTime()
    times[s] = float64(ticks(t1) - ticks(t0)) / 1e9
  let med = median(sorted(times))
  (name, gflops(med, ops), med * 1e6)

when isMainModule:
  randomize(42)

  echo "=".repeat(64)
  echo &"  SME GEMM benchmark — float32, row-major, single-threaded"
  echo &"  SIMD arch: {v_a.simdArchString()}"
  echo &"  sizes {ProblemSizes}, {WarmupSamples} warmup + {NbSamples} samples each"
  echo "=".repeat(64)
  echo ""

  for N in ProblemSizes:
    let
      shape: MatrixShape = (M: N, N: N)
      ops = float64(gemm_required_ops(shape, shape))
      rs = N
      cs = 1

    var
      A = newSeq[float32](N * N)
      B = newSeq[float32](N * N)
    for i in 0 ..< A.len: A[i] = rand(1.0'f32)
    for i in 0 ..< B.len: B[i] = rand(1.0'f32)

    # ── Correctness: SME path vs naive reference ──
    var
      C_naive = newSeq[float32](N * N)
      C_sme = newSeq[float32](N * N)
    gemm_reference(N, N, N, 1.0'f32, A, rs, cs, B, rs, cs, 0.0'f32, C_naive, rs, cs)
    v_a.gemm_strided(N, N, N, 1.0'f32, A, rs, cs, B, rs, cs, 0.0'f32, C_sme, rs, cs)
    let chk = allClose(C_sme, C_naive, rtol = Tol, atol = Tol)
    doAssert chk.ok, &"N={N}: SME vs naive maxAbsErr={chk.maxAbsErr:.2e}"

    # ── Timed runs (beta=0: C's input value is irrelevant, no re-zero needed) ──
    var results: seq[tuple[name: string; gflops, medianUs: float64]] = @[]
    block:
      var C = newSeq[float32](N * N)
      proc run() = v_a.gemm_strided(N, N, N, 1.0'f32, A, rs, cs, B, rs, cs, 0.0'f32, C, rs, cs)
      results.add bench("sme (ex02a-sme)", run, ops)
    if N < 512:
      # Naive baseline is cache-missing at 512³: five passes would dominate
      # the budget, so time it only up to 256³.
      block:
        var C = newSeq[float32](N * N)
        proc run() = gemm_reference(N, N, N, 1.0'f32, A, rs, cs, B, rs, cs, 0.0'f32, C, rs, cs)
        results.add bench("generic (naive)", run, ops)

    echo &"  N={N:4d}  correctness vs naive: maxAbsErr={chk.maxAbsErr:.2e} ✅"
    for r in results:
      echo &"    {r.name:<20} {r.gflops:>9.2f} GFLOP/s   {int(r.medianUs):>7d} μs"
    echo ""

  echo "Done."
