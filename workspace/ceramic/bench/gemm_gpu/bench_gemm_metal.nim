# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Run:
#   nim c -r -d:release --outdir:build/bench \
#     workspace/ceramic/bench/gemm_gpu/bench_gemm_metal.nim (from tattletale)

import std/[os, strutils]
import workspace/crucible
import workspace/crucible/src/runtime/engines/metal
import workspace/crucible/src/runtime/errors
import workspace/ceramic/bench/gemm_gpu/bench_gemm_template
import workspace/ceramic/bench/gemm_gpu/apple_nax_utils

const ProblemSizes = [256, 512, 1024, 2048, 4096]

const MlxSteelGemmSrc =
  staticRead("apple_gemm.metal") & staticRead("apple_mlx_bench_entrypoint.metal")

const NaxGemmSrc = naxGemmIngestSrc()

type MlxGemmParams = object
  m*, n*, k*: int
  lda*, ldb*, ldd*: int
  tiles_n*, tiles_m*: int
  batch_stride_a*, batch_stride_b*, batch_stride_d*: int64
  swizzle_log*: int
  gemm_k_iterations_aligned*: int
  batch_ndim*: int

proc benchAppleMlxSteel(engine: MetalEngine, M, N, K, iters: int): auto =
  let one = 1  # runtime int, the layouts stay dynamic
  var
    aHost = makeHostTensor(F16, (M, K), (K, one))
    bHost = makeHostTensor(F16, (N, K), (one, N))
    dHost = makeHostTensor(float32, (M, N), (N, one))
  fillGemmInputs(aHost.view(), bHost.view(), M, N, K)
  if engine.getArtifact() != MlxSteelGemmSrc:
    engine.ingest(MlxSteelGemmSrc)
  let
    tm = ceil_div(M, 64)
    tn = ceil_div(N, 64)
    swizzleLog = if tm <= 3: 0 else: 1
    tile = 1 shl swizzleLog
  let params = MlxGemmParams(
    m: M, n: N, k: K,
    lda: K, ldb: N, ldd: N,
    tiles_n: tn, tiles_m: tm,
    batch_stride_a: 0, batch_stride_b: 0, batch_stride_d: 0,
    swizzle_log: swizzleLog, gemm_k_iterations_aligned: K div 16,
    batch_ndim: 0)
  # warmup, the kernel loops 5 times inside one launch
  block:
    let
      aDev = engine.bridge(aHost)
      bDev = engine.bridge(bHost)
      dDev = engine.bridge(dHost)
    engine.launchKernel<<(grid: (tn * tile, ceil_div(tm, tile), 1), blk: (128, 1, 1), sharedMem: 5376)>>(
      "mlx_gemm_64x64x16", dDev.data, (aDev.data, bDev.data, params, 5))
  # timed, the launch overhead leaves the measurement
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  benchGemm("Apple MLX", M, N, K, iters):
    engine.launchKernel<<(grid: (tn * tile, ceil_div(tm, tile), 1), blk: (128, 1, 1), sharedMem: 5376)>>(
      "mlx_gemm_64x64x16", dDev.data, (aDev.data, bDev.data, params, iters))
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, M * N)
  dHost

proc benchAppleNax(engine: MetalEngine, M, N, K, iters: int): auto =
  let one = 1  # runtime int, the layouts stay dynamic
  var
    aHost = makeHostTensor(F16, (M, K), (K, one))
    bHost = makeHostTensor(F16, (N, K), (one, N))
    dHost = makeHostTensor(float32, (M, N), (N, one))
  fillGemmInputs(aHost.view(), bHost.view(), M, N, K)
  # the hardcoded wrapper config covers exactly M, N multiples of 64,
  # K multiples of 256: anything else would silently drop K-elements.
  if M mod 64 != 0 or N mod 64 != 0 or K mod 256 != 0:
    failLoud("NAX GEMM: unsupported shape " & $M & "x" & $N & "x" & $K & ", the wrapper config needs M and N multiples of 64, K a multiple of 256")
  if engine.getArtifact() != NaxGemmSrc:
    engine.ingest(NaxGemmSrc)
  let params = MlxGemmParams(
    m: M, n: N, k: K,
    lda: K, ldb: N, ldd: N,
    tiles_n: ceil_div(N, 64), tiles_m: ceil_div(M, 64),
    batch_stride_a: 0, batch_stride_b: 0, batch_stride_d: 0,
    swizzle_log: 0, gemm_k_iterations_aligned: K div 256,
    batch_ndim: 0)
  # warmup, the kernel loops 5 times inside one launch
  block:
    let
      aDev = engine.bridge(aHost)
      bDev = engine.bridge(bHost)
      dDev = engine.bridge(dHost)
    engine.launchKernel<<(grid: (ceil_div(N, 64), ceil_div(M, 64), 1), blk: (128, 1, 1))>>(
      "mlx_gemm_nax_64x64", dDev.data, (aDev.data, bDev.data, params, 5))
  # timed, the launch overhead leaves the measurement
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  benchGemm("Apple NAX", M, N, K, iters):
    engine.launchKernel<<(grid: (ceil_div(N, 64), ceil_div(M, 64), 1), blk: (128, 1, 1))>>(
      "mlx_gemm_nax_64x64", dDev.data, (aDev.data, bDev.data, params, iters))
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, M * N)
  dHost

proc main() =
  var engine = bkMetal.newEngine()
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  separator()
  for mnk in ProblemSizes:
    if wanted.len == 0 or mnk in wanted:
      let iters = if mnk >= 4096: 20 elif mnk >= 2048: 100 else: 1000
      let dTiled = benchCeramic(engine, mnk, mnk, mnk, iters)
      let dMlx = benchAppleMlxSteel(engine, mnk, mnk, mnk, iters)
      let dNax = benchAppleNax(engine, mnk, mnk, mnk, iters)
      # the three implementations must agree on the same inputs
      assertAllClose(dMlx.view(), dTiled.view(), rtol = 1e-2, abstol = 1e-2, msg = "mlx steel vs tiledMma")
      assertAllClose(dNax.view(), dTiled.view(), rtol = 1e-2, abstol = 1e-2, msg = "NAX vs tiledMma")
      separator()
  separator()

when isMainModule:
  main()
