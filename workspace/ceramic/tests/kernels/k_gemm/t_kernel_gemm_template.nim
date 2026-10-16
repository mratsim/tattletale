# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GEMM kernel test driver, whole-value launches, backend agnostic
##
## Run:
##   nim c -r --outdir:build \
##     workspace/ceramic/tests/kernels/k_gemm/t_kernel_gemm.nim (from tattletale)

import std/[math, random, strformat, times]
import workspace/crucible
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/src/tile_algebra
import workspace/fp_conv/fp_conv

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════
#  fp32 host reference
# ═══════════════════════════════════════════════════════════════

func view2D[T](data: ptr UncheckedArray[T], shape, strides: (int, int)):
    TensorView[T, (int, int), (int, int)] =
  ## 2D view over a presized buffer
  make_view(data, make_layout(shape, strides))

proc gemm_ref*[ShA, StA, ShB, StB, ShC, StC](alpha: float32,
               A: TensorView[F16, ShA, StA],
               B: TensorView[F16, ShB, StB],
               beta: float32,
               C, refOut: TensorView[float32, ShC, StC]) =
  let
    M = A.getLayout().shape[0]
    N = B.getLayout().shape[0]
    K = A.getLayout().shape[1]
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        let a = A[m, k]
        let b = B[n, k]
        acc += float32.from(a) * float32.from(b)
      refOut[m, n] = (if beta != 0.0'f32: alpha * acc + beta * C[m, n]
                      else: alpha * acc)

# ═══════════════════════════════════════════════════════════════
#  Device function generation
# ═══════════════════════════════════════════════════════════════

# the mma tile of this suite, the atom cell repeated over (MT, NT, KT)
const
  TileM = 32
  TileN = 32
  TileK = 16

# ═══════════════════════════════════════════════════════════════
#  Test runner
# ═══════════════════════════════════════════════════════════════

proc run_gemm_tests*[E: HwEngine](engine: E, Iters: static int, moduleName: string) =
  ## Exercises the generated GEMM device function of the driver's engine,
  ## every output element asserted against `gemm_ref`.
  ##
  ## - cases: stride permutations, ragged tile dims
  ## - cases: excess squadrons, 1D grids, block size variants
  ## - cases: alpha scaling, beta accumulation against a prefilled C
  let seed = uint64(getTime().toUnix() and (1'i64 shl 32 - 1))
  var rng = initRand(int64(seed))
  echo moduleName, " PCG seed: ", seed

    # the presized pool, one allocation per engine run, cases reuse it
  const
    MaxM = 70
    MaxN = 64
    MaxK = 100
  var
    aHost = newSeq[F16](MaxM * MaxK)
    bHost = newSeq[F16](MaxK * MaxN)
    cHost = newSeq[float32](MaxM * MaxN)
    dHost = newSeq[float32](MaxM * MaxN)
    refHost = newSeq[float32](MaxM * MaxN)
  let
    aDev = engine.allocDevice(MaxM * MaxK * sizeof(F16))
    bDev = engine.allocDevice(MaxK * MaxN * sizeof(F16))
    cDev = engine.allocDevice(MaxM * MaxN * sizeof(float32))
    dDev = engine.allocDevice(MaxM * MaxN * sizeof(float32))

  template checkGemm(M, N, K: untyped, sA, sB, sC: (int, int),
                     gxMul: untyped = 1, gyMul: untyped = 1,
                     gridFlat: untyped = false, blkX: untyped = 32,
                     alpha: float32 = 2.0'f32, beta: float32 = 1.0'f32,
                     prefill: bool = false, launchTiles: untyped = 0) =
    ## Runs one gemm configuration against `gemm_ref`:
    ##
    ## - gxMul / gyMul multiply the tile grid, excess squadrons exit
    ## - gridFlat covers the whole tile grid from one grid axis, blkX
    ##   widens the block, extra threads exit on the thread count
    ## - prefill writes C = A·B on device first, the beta term reads it
    ## - launchTiles launches fewer blocks than the tile grid has tiles,
    ##   the blocks loop over the grid, the result is complete
    let
      aTV = view2D(cast[ptr UncheckedArray[F16]](addr aHost[0]), (M, K), sA)
      # the kernel sees B as an (N, K) view with strides (sB[0], sB[1])
      bTV = view2D(cast[ptr UncheckedArray[F16]](addr bHost[0]), (N, K), sB)
      cTV = view2D(cast[ptr UncheckedArray[float32]](addr cHost[0]), (M, N), sC)
      dTV = view2D(cast[ptr UncheckedArray[float32]](addr dHost[0]), (M, N), sC)
      refTV = view2D(cast[ptr UncheckedArray[float32]](addr refHost[0]), (M, N), (N, 1))
      dD = view2D(cast[ptr UncheckedArray[float32]](dDev), (M, N), sC)
      dC = view2D(cast[ptr UncheckedArray[float32]](cDev), (M, N), sC)
      dA = view2D(cast[ptr UncheckedArray[float16]](aDev), (M, K), sA)
      dB = view2D(cast[ptr UncheckedArray[float16]](bDev), (N, K), sB)
      ctx = "gemm " & $M & "x" & $N & "x" & $K & " A" & $sA & " B" & $sB &
        " C" & $sC &
        (if gxMul != 1 or gyMul != 1: " grid x" & $gxMul & "x" & $gyMul else: "") &
        (if gridFlat: " flat" else: "") &
        (if blkX != 32: " blk" & $blkX else: "") &
        (if launchTiles != 0: " cover" & $launchTiles else: "") &
        " alpha" & $alpha & " beta" & $beta & (if prefill: " prefilled" else: "")
    for m in 0 ..< M:
      for k in 0 ..< K:
        aTV[m, k] = F16.from(rng.rand(1.0'f32 .. 2.0'f32))
    for n in 0 ..< N:
      for k in 0 ..< K:
        bTV[n, k] = F16.from(rng.rand(1.0'f32 .. 2.0'f32))
    engine.copyToDeviceFromHost(aDev, addr aHost[0], M * K * sizeof(F16))
    engine.copyToDeviceFromHost(bDev, addr bHost[0], K * N * sizeof(F16))
    let
      tilesX = ((M + 31) div 32) * gxMul
      tilesY = ((N + 31) div 32) * gyMul
      gridDims = (if gridFlat: (tilesX * tilesY, 1) else: (tilesX, tilesY))
    if prefill:
      # C = A·B on device, the gemm's beta term reads it back
      gemm_ref(1.0'f32, aTV, bTV, 0.0'f32, dD, cTV)
      engine.copyToDeviceFromHost(cDev, addr cHost[0], M * N * sizeof(float32))
      engine.wrap_and_run gemm(tiledMma(float32, float16, float16, TileM, TileN, TileK),
                               dC, 1.0'f32, dA, dB, 0.0'f32, dD) do:
        grid = (tilesX, tilesY)
        blk = (32, 1)
    when launchTiles != 0:
      engine.wrap_and_run gemm(tiledMma(float32, float16, float16, TileM, TileN, TileK),
                               dD, alpha, dA, dB, beta, dC) do:
        grid = (launchTiles, 1)
        blk = (32, 1)
    else:
      engine.wrap_and_run gemm(tiledMma(float32, float16, float16, TileM, TileN, TileK),
                               dD, alpha, dA, dB, beta, dC) do:
        grid = gridDims
        blk = (blkX, 1)
    engine.copyToHostFromDevice(addr dHost[0], dDev, M * N * sizeof(float32))
    gemm_ref(alpha, aTV, bTV, beta, cTV, refTV)
    assertAllClose(dTV, refTV, rtol = 1e-5, abstol = 1e-3, msg = ctx)
    echo "  ", ctx, " matched"

  # stride permutations at the atom tile size
  for _ in 0 ..< Iters:
    for sA in [(32, 1), (1, 32)]:
      for sB in [(1, 32), (32, 1)]:
        for sC in [(32, 1), (1, 32)]:
          checkGemm(32, 32, 32, sA, sB, sC)

  # ragged edges, tile dims not multiples of the (32, 32, 16) mma atom
  for _ in 0 ..< Iters:
    checkGemm(32, 32, 41, (41, 1), (1, 32), (32, 1))
    checkGemm(40, 32, 32, (32, 1), (1, 32), (32, 1))
    checkGemm(32, 45, 32, (32, 1), (1, 45), (45, 1))
    checkGemm(40, 45, 41, (41, 1), (1, 45), (45, 1))
    checkGemm(70, 50, 100, (100, 1), (1, 50), (50, 1))
    checkGemm(70, 50, 96, (96, 1), (1, 50), (50, 1))

  # excess squadrons, the launch grid over-covers the tile grid
  for _ in 0 ..< Iters:
    checkGemm(64, 64, 64, (64, 1), (1, 64), (64, 1), gxMul = 3, gyMul = 2)
    checkGemm(70, 50, 100, (100, 1), (1, 50), (50, 1), gxMul = 2, gyMul = 3)

  # 1D grids, the whole tile grid covered from one axis
  for _ in 0 ..< Iters:
    checkGemm(64, 64, 64, (64, 1), (1, 64), (64, 1), gridFlat = true)
    checkGemm(70, 50, 100, (100, 1), (1, 50), (50, 1), gridFlat = true, gxMul = 2)

  # under-coverage, not enough blocks: covered tiles match, the rest
  # keep the sentinel
  for _ in 0 ..< Iters:
    checkGemm(64, 64, 64, (64, 1), (1, 64), (64, 1), launchTiles = 2,
              alpha = 1.0'f32, beta = 0.0'f32)
    checkGemm(64, 64, 64, (64, 1), (1, 64), (64, 1), launchTiles = 3,
              alpha = 1.0'f32, beta = 0.0'f32)

  # hard excess, the launch grid far over-covers the tile grid
  for _ in 0 ..< Iters:
    checkGemm(64, 64, 64, (64, 1), (1, 64), (64, 1), gxMul = 10, blkX = 256)

  # block size variants, extra threads exit on the thread count
  for _ in 0 ..< Iters:
    for blkX in [64, 128, 256]:
      checkGemm(64, 64, 64, (64, 1), (1, 64), (64, 1), blkX = blkX)

  # alpha scaling and beta accumulation, the beta case reads a prefilled C
  for _ in 0 ..< Iters:
    checkGemm(32, 32, 32, (32, 1), (1, 32), (32, 1), alpha = 1.0'f32, beta = 0.0'f32)
    checkGemm(32, 32, 32, (32, 1), (1, 32), (32, 1), prefill = true)
    checkGemm(40, 45, 41, (41, 1), (1, 45), (45, 1), prefill = true)

  engine.freeDevice(aDev)
  engine.freeDevice(bDev)
  engine.freeDevice(cDev)
  engine.freeDevice(dDev)
