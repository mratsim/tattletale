# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: GEMM on Metal, effective TFLOP/s.
#
# Flops: 2*M*N*K, one multiply-add per output element.
#
# Run:
#   nim c -r -d:release --outdir:build/bench workspace/ceramic/bench/gemm_gpu/bench_gemm_metal.nim

import std/[monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/src/kernels/k_tile_gemm_smem
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/bench/gemm_gpu/bench_blueprint
import workspace/ceramic/tests/kernels/k_gemm/kernel_gemm_template

export bench_blueprint
export int_tuples
export layout_algebra
export tensors
export tile_algebra
export ceramic_testutils

proc fillGemmInputs*[TVa, TVb](aTV: TVa, bTV: TVb, M, N, K: int) =
  for m in 0 ..< M:
    for k in 0 ..< K:
      aTV[m, k] = to(1 + (2 * m + 7 * k) mod 13, typeof(aTV[0, 0]))
  for n in 0 ..< N:
    for k in 0 ..< K:
      bTV[n, k] = to(1 + (3 * k + 11 * n) mod 17, typeof(bTV[0, 0]))

proc gemmLoop*(mma: static TiledMma, C, A, B: distinct TensorView, iters: int) =
  for _ in 0 ..< iters:
    matmul(mma, C, A, B)

proc smemGemmLoop*[TAcc, TA, TB, MT, NT, KT](
    mma: static TiledMma[TAcc, TA, TB, MT, NT, KT],
    mmaStep: static TiledMma[TAcc, TA, TB, MT, NT, 16],
    D, A, B: distinct TensorView, iters: int) =
  for _ in 0 ..< iters:
    matmul_smem(mma, mmaStep, D, A, B)

func computeGemmFlops*(M, N, K: int): float64 =
  ## 2*M*N*K, one multiply-add per output element
  2.0 * M.float * N.float * K.float

template benchGemm*(op: string, M, N, K, iters: int, body: untyped) =
  let
    config = $M & "x" & $N & "x" & $K
    flops = computeGemmFlops(M, N, K)
    benchStart = getMonoTime()
  body
  let benchStop = getMonoTime()
  discard report(op & ": gemm", config, benchStart, benchStop, iters, flops)

proc benchCeramic*[E: HwEngine](engine: E, M, N, K, iters: int): auto =
  let one = 1  # runtime int, the layouts stay dynamic
  var
    aHost = makeHostTensor(float16, (M, K), (K, one))
    bHost = makeHostTensor(float16, (N, K), (one, N))
    dHost = makeHostTensor(float32, (M, N), (N, one))
  fillGemmInputs(aHost.view(), bHost.view(), M, N, K)
  # warmup, the kernel loops 5 times inside one launch
  block:
    let
      aDev = engine.bridge(aHost)
      bDev = engine.bridge(bHost)
      dDev = engine.bridge(dHost)
    engine.launchDeviceFn gemmLoop(tiledMma(float32, float16, float16, 32, 32, 16), dDev, aDev, bDev, 5) do:
      grid = (ceil_div(M, 32), ceil_div(N, 32))
      blk = (32, 1)
  # timed, the launch overhead leaves the measurement
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  benchGemm("Tattletale Ceramic", M, N, K, iters):
    engine.launchDeviceFn gemmLoop(tiledMma(float32, float16, float16, 32, 32, 16), dDev, aDev, bDev, iters) do:
      grid = (ceil_div(M, 32), ceil_div(N, 32))
      blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, M * N)
  dHost

proc benchSmemCeramic*[E: HwEngine](engine: E, M, N, K, iters: int): auto =
  ## Benchmarks the staged bf16 GEMM through matmul_smem:
  ##
  ## - `iters` timed launches of D = A·B, alpha = 1, beta = 0
  ## - bf16 in, f32 accumulate
  ## - A and B k-contiguous, the staging fast path's layout
  let one = 1  # one is a runtime int so the layouts stay dynamic
  var
    aHost = makeHostTensor(bfloat16, (M, K), (K, one))
    bHost = makeHostTensor(bfloat16, (N, K), (K, one))
    dHost = makeHostTensor(float32, (M, N), (N, one))
  fillGemmInputs(aHost.view(), bHost.view(), M, N, K)
  # warmup, the kernel loops 5 times inside one launch
  block:
    let
      aDev = engine.bridge(aHost)
      bDev = engine.bridge(bHost)
      dDev = engine.bridge(dHost)
    engine.launchDeviceFn smemGemmLoop(
        tiledMma(float32, bfloat16, bfloat16, 32, 32, 64),
        tiledMma(float32, bfloat16, bfloat16, 32, 32, 16),
        dDev, aDev, bDev, 5) do:
      # one (warpsPerDim·32, warpsPerDim·32) block tile per block
      grid = (ceil_div(M, warpsPerDim * 32), ceil_div(N, warpsPerDim * 32))
      blk = (warpsPerBlock * threadsPerWarp, 1)
  # timed, the launch overhead leaves the measurement
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  benchGemm("Tattletale Ceramic smem", M, N, K, iters):
    engine.launchDeviceFn smemGemmLoop(
        tiledMma(float32, bfloat16, bfloat16, 32, 32, 64),
        tiledMma(float32, bfloat16, bfloat16, 32, 32, 16),
        dDev, aDev, bDev, iters) do:
      # one (warpsPerDim·32, warpsPerDim·32) block tile per block
      grid = (ceil_div(M, warpsPerDim * 32), ceil_div(N, warpsPerDim * 32))
      blk = (warpsPerBlock * threadsPerWarp, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, M * N)
  dHost
