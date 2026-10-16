# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Run:
#   nim c -r -d:release --outdir:build/bench \
#     workspace/ceramic/benchmarks/gemm_gpu/bench_tile_gemm_metal.nim (from tattletale)

import std/[os, strutils]
import workspace/crucible
import workspace/crucible/src/runtime/engines/metal
import workspace/ceramic/benchmarks/gemm_gpu/bench_gemm_template

const ProblemSizes = [1024, 2048, 4096]

const MlxSteelGemmSrc =
  staticRead("mlx_steel_gemm.metal") & staticRead("steel_bench_wrappers.metal")
var steelCurrent = false

type MlxGemmParams = object
  m*, n*, k*: int
  lda*, ldb*, ldd*: int
  tiles_n*, tiles_m*: int
  batch_stride_a*, batch_stride_b*, batch_stride_d*: int64
  swizzle_log*: int
  gemm_k_iterations_aligned*: int
  batch_ndim*: int

proc gemmLoop(mma: static TiledMma, C, A, B: distinct TensorView, iters: int) =
  for _ in 0 ..< iters:
    matmul(mma, C, A, B)

proc launchGemm(engine: MetalEngine, dDev, aDev, bDev: pointer, M, N, K, iters: int) =
  let one = 1  # runtime int, the layouts stay dynamic
  let
    dC = make_view(cast[ptr UncheckedArray[float32]](dDev),
      make_layout((M, N), (N, one)))
    dA = make_view(cast[ptr UncheckedArray[float16]](aDev),
      make_layout((M, K), (K, one)))
    dB = make_view(cast[ptr UncheckedArray[float16]](bDev),
      make_layout((N, K), (one, N)))
  engine.wrap_and_run gemmLoop(tiledMma(float32, float16, float16, 32, 32, 16), dC, dA, dB, iters) do:
    grid = (ceil_div(M, 32), ceil_div(N, 32))
    blk = (32, 1)

proc launchMlxGemm(engine: MetalEngine, dDev, aDev, bDev: pointer, M, N, K, iters: int) =
  if not steelCurrent:
    engine.ingest(MlxSteelGemmSrc)
    steelCurrent = true
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
  engine.run("mlx_gemm_64x64x16", dDev, (aDev, bDev, params, iters),
    LaunchConfig(grid: dim3((tn * tile, ceil_div(tm, tile), 1)),
                 blk: dim3((128, 1, 1)), sharedMem: 5376))

proc main() =
  var engine = bkMetal.init()
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  separator()
  for mnk in ProblemSizes:
    if wanted.len == 0 or mnk in wanted:
      let iters = if mnk >= 4096: 10 elif mnk >= 2048: 25 else: 50
      gemmBench(engine, launchGemm, mnk, mnk, mnk, iters,
                "tiledMma 32x32x16 fp16")
      gemmBench(engine, launchMlxGemm, mnk, mnk, mnk, iters,
                "mlx steel gemm 64x64x16")
      separator()

when isMainModule:
  warmup()
  main()
