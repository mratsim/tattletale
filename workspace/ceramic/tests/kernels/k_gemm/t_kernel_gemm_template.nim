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

proc run_gemm_tests*[E: HwEngine](engine: E, Iters: static int, moduleName: string) =
  ## Exercises the generated GEMM device function of the driver's engine,
  ## every output element asserted against `gemm_ref`
  let seed = uint64(getTime().toUnix() and (1'i64 shl 32 - 1))
  echo moduleName, " PCG seed: ", seed
  var rng = initRand(int64(seed))
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
  for i in 0 ..< aHost.len:
    aHost[i] = F16.from(rng.rand(1.0'f32 .. 2.0'f32))
  for i in 0 ..< bHost.len:
    bHost[i] = F16.from(rng.rand(1.0'f32 .. 2.0'f32))
  for i in 0 ..< cHost.len:
    cHost[i] = rng.rand(1.0'f32 .. 2.0'f32)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], aHost.len * sizeof(F16))
  engine.copyToDeviceFromHost(bDev, addr bHost[0], bHost.len * sizeof(F16))
  engine.copyToDeviceFromHost(cDev, addr cHost[0], cHost.len * sizeof(float32))

  proc checkGemm(M, N, K: int, sA, sB, sC: (int, int), alpha = 2.0'f32, beta = 1.0'f32) =
    let
      aTV = view2D(cast[ptr UncheckedArray[F16]](addr aHost[0]), (M, K), sA)
      bTV = view2D(cast[ptr UncheckedArray[F16]](addr bHost[0]), (N, K), sB)
      cTV = view2D(cast[ptr UncheckedArray[float32]](addr cHost[0]), (M, N), sC)
      dTV = view2D(cast[ptr UncheckedArray[float32]](addr dHost[0]), (M, N), sC)
      refTV = view2D(cast[ptr UncheckedArray[float32]](addr refHost[0]), (M, N), (N, 1))
      dD = view2D(cast[ptr UncheckedArray[float32]](dDev), (M, N), sC)
      dC = view2D(cast[ptr UncheckedArray[float32]](cDev), (M, N), sC)
      dA = view2D(cast[ptr UncheckedArray[float16]](aDev), (M, K), sA)
      dB = view2D(cast[ptr UncheckedArray[float16]](bDev), (N, K), sB)
    # the blocks loop over the tile grid, any launch grid completes it
    let
      gridX = rng.rand(1 .. 2 * ceil_div(M, 32))
      gridY = rng.rand(1 .. 2 * ceil_div(N, 32))
    engine.wrap_and_run gemm(tiledMma(float32, float16, float16, TileM, TileN, TileK),
                             dD, alpha, dA, dB, beta, dC) do:
      grid = (gridX, gridY)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr dHost[0], dDev, M * N * sizeof(float32))
    gemm_ref(alpha, aTV, bTV, beta, cTV, refTV)
    assertAllClose(dTV, refTV, rtol = 1e-5, abstol = 1e-3)

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

  # alpha scaling and beta accumulation against C
  for _ in 0 ..< Iters:
    checkGemm(32, 32, 32, (32, 1), (1, 32), (32, 1), alpha = 1.0'f32, beta = 0.0'f32)
    checkGemm(32, 32, 32, (32, 1), (1, 32), (32, 1), alpha = 0.5'f32, beta = 2.0'f32)
    checkGemm(40, 45, 41, (41, 1), (1, 45), (45, 1), beta = 2.0'f32)

  engine.freeDevice(aDev)
  engine.freeDevice(bDev)
  engine.freeDevice(cDev)
  engine.freeDevice(dDev)
