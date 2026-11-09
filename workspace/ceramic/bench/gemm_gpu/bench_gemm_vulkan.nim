# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: GEMM on Vulkan (MoltenVK), effective TFLOP/s.
#
# Flops: 2*M*N*K, one multiply-add per output element.
# Protocol: one warmup launch, then 3 quiet-window rounds.
# A round times `iters` back-to-back GEMM launches, best round counted.
# End-to-end wall-clock: launches and the one-element sync copy
# sit inside the timed window, all data movement paid.
#
# Run from tattletale:
#   nim c -r -d:release --outdir:build/bench workspace/ceramic/bench/gemm_gpu/bench_gemm_vulkan.nim

import std/[os, strutils]
import workspace/crucible
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/bench/gemm_gpu/bench_gemm_template

export bench_gemm_template

const ProblemSizes = [256, 512, 1024, 2048, 4096]

proc fillGemmInputsF32*[TVa, TVb](aTV: TVa, bTV: TVb, M, N, K: int) =
  for m in 0 ..< M:
    for k in 0 ..< K:
      aTV[m, k] = float32(1 + (2 * m + 7 * k) mod 13)
  for n in 0 ..< N:
    for k in 0 ..< K:
      bTV[n, k] = float32(1 + (3 * k + 11 * n) mod 17)

proc gemmLoopF32*(mma: static TiledMma, D, A, B: distinct TensorView, iters: int) =
  for _ in 0 ..< iters:
    gemm(mma, D, 1.0'f32, A, B, 0.0'f32, D)

proc benchCeramicVulkan(engine: VulkanEngine, M, N, K, iters: int): auto =
  let one = 1  # runtime int, layouts stay dynamic
  var
    aHost = makeHostTensor(float32, (M, K), (K, one))
    bHost = makeHostTensor(float32, (N, K), (one, N))
    dHost = makeHostTensor(float32, (M, N), (N, one))
  fillGemmInputsF32(aHost.view(), bHost.view(), M, N, K)
  # warmup: the kernel loops 5 times inside one launch
  block:
    let
      aDev = engine.bridge(aHost)
      bDev = engine.bridge(bHost)
      dDev = engine.bridge(dHost)
    engine.launchDeviceFn gemmLoopF32(tiledMma(float32, float32, float32, 32, 32, 2), dDev, aDev, bDev, 5) do:
      grid = (ceil_div(M, 32), ceil_div(N, 32))
      blk = (32, 1)
      lang = glsl
  # timed: 3 quiet-window rounds, best round counts
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  for round in 0 ..< 3:
    discard bench("Tattletale Ceramic Vulkan", $M & "x" & $N & "x" & $K, iters,
          computeGemmFlops(M, N, K)):
      engine.launchDeviceFn gemmLoopF32(tiledMma(float32, float32, float32, 32, 32, 2), dDev, aDev, bDev, 1) do:
        grid = (ceil_div(M, 32), ceil_div(N, 32))
        blk = (32, 1)
        lang = glsl
      engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, M * N)
  dHost

proc main() =
  var engine = newEngine(runVulkan)
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  separator()
  for mnk in ProblemSizes:
    if wanted.len == 0 or mnk in wanted:
      let iters = if mnk >= 4096: 20 elif mnk >= 2048: 100 else: 1000
      discard benchCeramicVulkan(engine, mnk, mnk, mnk, iters)
      separator()
  separator()

when isMainModule:
  main()
