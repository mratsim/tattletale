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

import std/[math, random, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/fp_conv/fp_conv

export int_tuples, int_tuples_streams
export layout_algebra
export tensors
export tile_algebra
export k_tile_gemm

# ═══════════════════════════════════════════════════════════════
#  fp32 host reference
# ═══════════════════════════════════════════════════════════════

func gemm_ref*(refOut: distinct HostTensor, alpha: float32, A, B: distinct HostTensor, beta: float32, C: distinct HostTensor) =
  ## Host fp32 reference: refOut = alpha * A * B^T + beta * C, BLAS B^T form
  let
    M = A.layout.shape[0]
    N = B.layout.shape[0]
    K = A.layout.shape[1]
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        let a = A.view()[m, k]
        let b = B.view()[n, k]
        acc += float32.from(a) * float32.from(b)
      refOut.view()[m, n] = (if beta != 0.0'f32: alpha * acc + beta * C.view()[m, n]
                      else: alpha * acc)

# ═══════════════════════════════════════════════════════════════
#  Device function generation
# ═══════════════════════════════════════════════════════════════

# the mma tile of this suite, the atom cell repeated over (MT, NT, KT)
const
  TileM = 32
  TileN = 32
  TileK = 16

proc checkGemm*[E: HwEngine](rng: var Rand, engine: E, M, N, K: int, sA, sB, sC: (int, int), alpha = 2.0'f32, beta = 1.0'f32) =
  ## One GEMM launch over random (M, K) x (K, N) tensors
  var
    aHost = randomHostTensor(rng, F16, (M, K), sA)
    bHost = randomHostTensor(rng, F16, (N, K), sB)
    cHost = randomHostTensor(rng, float32, (M, N), sC)
    dHost = makeHostTensor(float32, (M, N), sC)
    refHost = makeHostTensor(float32, (M, N), (N, 1))
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    cDev = engine.bridge(cHost)
    dDev = engine.bridge(dHost)
  # the blocks loop over the tile grid, any launch grid completes it
  let
    gridX = rng.rand(1 .. 2 * ceil_div(M, 32))
    gridY = rng.rand(1 .. 2 * ceil_div(N, 32))
  engine.launchDeviceFn gemm(tiledMma(float32, float16, float16, TileM, TileN, TileK),
                           dDev, alpha, aDev, bDev, beta, cDev) do:
    grid = (gridX, gridY)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  gemm_ref(refHost, alpha, aHost, bHost, beta, cHost)
  assertAllClose(dHost.view(), refHost.view(), rtol = 1e-5, abstol = 1e-3)

proc run_gemm_tests*[E: HwEngine](engine: E, Iters: static int, moduleName: string) =
  ## Exercises the generated GEMM device function of the driver's engine,
  ## every output element asserted against `gemm_ref`
  var rng: Rand
  let seed = uint32(getTime().toUnix() and (1'i64 shl 32 - 1)) # unixTime mod 2^32
  rng.seed(int64(seed))
  echo "\n------------------------------------------------------\n"
  echo moduleName, " xoroshiro128+ seed: ", seed
  # stride permutations at the atom tile size
  for _ in 0 ..< Iters:
    for sA in [(32, 1), (1, 32)]:
      for sB in [(1, 32), (32, 1)]:
        for sC in [(32, 1), (1, 32)]:
          checkGemm(rng, engine, 32, 32, 32, sA, sB, sC)

  # ragged edges, tile dims not multiples of the (32, 32, 16) mma atom
  for _ in 0 ..< Iters:
    checkGemm(rng, engine,
        32, 32, 41, (41, 1), (1, 32), (32, 1))
    checkGemm(rng, engine,
        40, 32, 32, (32, 1), (1, 32), (32, 1))
    checkGemm(rng, engine,
        32, 45, 32, (32, 1), (1, 45), (45, 1))
    checkGemm(rng, engine,
        40, 45, 41, (41, 1), (1, 45), (45, 1))
    checkGemm(rng, engine,
        70, 50, 100, (100, 1), (1, 50), (50, 1))
    checkGemm(rng, engine,
        70, 50, 96, (96, 1), (1, 50), (50, 1))

  # alpha scaling and beta accumulation against C
  for _ in 0 ..< Iters:
    checkGemm(rng, engine,
        32, 32, 32, (32, 1), (1, 32), (32, 1), alpha = 1.0'f32, beta = 0.0'f32)
    checkGemm(rng, engine,
        32, 32, 32, (32, 1), (1, 32), (32, 1), alpha = 0.5'f32, beta = 2.0'f32)
    checkGemm(rng, engine,
        40, 45, 41, (41, 1), (1, 45), (45, 1), beta = 2.0'f32)

