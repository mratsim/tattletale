# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Run:
#   nim c -r -d:release --outdir:build/bench \
#     workspace/ceramic/bench/gemm_gpu/bench_gemm_metal.nim (from tattletale)

import std/[os, strutils]
import workspace/crucible
import workspace/crucible/src/runtime/engines/metal
import workspace/crucible/src/abis/objc_abi as objc
import workspace/crucible/src/runtime/errors
import workspace/ceramic/bench/gemm_gpu/bench_gemm_template
import workspace/ceramic/bench/gemm_gpu/apple_nax_utils

const ProblemSizes = [1024, 2048, 4096]

const MlxSteelGemmSrc =
  staticRead("mlx_steel_gemm.metal") & staticRead("steel_bench_wrappers.metal")

const NaxGemmSrc = naxGemmIngestSrc()

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

proc launchGemm[H](engine: MetalEngine, dDev: DeviceMem[H, float32], aDev, bDev: DeviceMem[H, F16], M, N, K, iters: int) =
  let
    dC = devView2D(dDev, (M, N), (N, 1))
    dA = devView2D(aDev, (M, K), (K, 1))
    dB = devView2D(bDev, (N, K), (1, N))
  engine.launchDeviceFn gemmLoop(tiledMma(float32, float16, float16, 32, 32, 16), dC, dA, dB, iters) do:
    grid = (ceil_div(M, 32), ceil_div(N, 32))
    blk = (32, 1)

proc launchMlxGemm[H](engine: MetalEngine, dDev: DeviceMem[H, float32], aDev, bDev: DeviceMem[H, F16], M, N, K, iters: int) =
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
  engine.launchKernel<<(grid: (tn * tile, ceil_div(tm, tile), 1), blk: (128, 1, 1), sharedMem: 5376)>>(
    "mlx_gemm_64x64x16", dDev, (aDev, bDev, params, iters))

proc launchNaxGemm[H](engine: MetalEngine, dDev: DeviceMem[H, float32], aDev, bDev: DeviceMem[H, F16], M, N, K, iters: int) =
  # The hardcoded wrapper config covers exactly M, N multiples of 64,
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
  engine.launchKernel<<(grid: (ceil_div(N, 64), ceil_div(M, 64), 1), blk: (128, 1, 1))>>(
    "mlx_gemm_nax_64x64", dDev, (aDev, bDev, params, iters))

proc main() =
  var engine = bkMetal.init()
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  separator()
  var sizeIdx = 0
  for mnk in ProblemSizes:
    if wanted.len == 0 or mnk in wanted:
      let iters = if mnk >= 4096: 20 elif mnk >= 2048: 100 else: 1000
      # the run order rotates per size index, thermal drift stops
      # systematically favoring the implementation that runs first
      let order = [sizeIdx mod 3, (sizeIdx + 1) mod 3, (sizeIdx + 2) mod 3]
      for oi in order:
        case oi
        of 0:
          gemmBench[MetalEngine, objc.ID](engine, launchGemm, mnk, mnk, mnk, iters, "tiledMma 32x32x16 fp16")
        of 1:
          gemmBench[MetalEngine, objc.ID](engine, launchMlxGemm, mnk, mnk, mnk, iters, "mlx steel gemm 64x64x16")
        else:
          gemmBench[MetalEngine, objc.ID](engine, launchNaxGemm, mnk, mnk, mnk, iters, "mlx NAX gemm 64x64x256")
      separator()
    inc sizeIdx

when isMainModule:
  main()
