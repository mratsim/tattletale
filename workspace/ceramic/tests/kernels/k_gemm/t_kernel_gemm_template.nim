# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Template for the k_tile_gemm Metal tests, instantiate through
## run_gemm_tests from a t_ driver.
##
## Run:
##   nim c -r --outdir:build \
##     workspace/ceramic/tests/kernels/k_gemm/t_kernel_gemm.nim (from tattletale)

import std/[math, random, strformat, times]
import workspace/crucible
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

const MT = 32
const NT = 32
const KT = 16

# ═══════════════════════════════════════════════════════════════
#  Kernels
# ═══════════════════════════════════════════════════════════════

proc gemmKernels(): string =
  metal:
    proc gemmK(D, C: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(A, makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(B, makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
      let dView = make_view(D, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      matmul(mma, dView, aView, bView)
    proc gemmAlphaK(D, C: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(A, makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(B, makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
      let dView = make_view(D, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      let cView = make_view(C, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      gemm(mma, dView, 2.0'f32, aView, bView, 0.0'f32, cView)
    proc gemmAlphaBetaK(D, C: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(A, makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(B, makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
      let dView = make_view(D, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      let cView = make_view(C, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      gemm(mma, dView, 2.0'f32, aView, bView, 1.0'f32, cView)

# ═══════════════════════════════════════════════════════════════
#  Test runner
# ═══════════════════════════════════════════════════════════════

proc run_gemm_tests*(Iters: static int, moduleName: string) =
  let seed = uint64(getTime().toUnix() and (1'i64 shl 32 - 1))
  var rng = initRand(int64(seed))
  echo moduleName, " PCG seed: ", seed

  var engine = bkMetal.init()
  engine.ingest(gemmKernels())

  proc checkMatmul(M, N, K: int, sA, sB, sC: (int, int), gxMul = 1, gyMul = 1, kernel = "gemmK", scale = 1.0'f32, prefill = false) =
    ## Runs one gemm configuration and asserts every output element
    ## against the fp32 reference, worst |d| echoed as info.
    let
      gridX = ((M + MT - 1) div MT) * gxMul
      gridY = ((N + NT - 1) div NT) * gyMul
    var
      aHost = newSeq[F16](M * K)
      bHost = newSeq[F16](K * N)
      outHost = newSeq[float32](M * N)
    let
      aTV = make_view(addr aHost[0], make_layout((M, K), (sA[0], sA[1])))
      # the kernel sees B as an (N, K) view with strides (sB[0], sB[1])
      bTV = make_view(addr bHost[0], make_layout((N, K), (sB[0], sB[1])))
      outTV = make_view(addr outHost[0], make_layout((M, N), (sC[0], sC[1])))
    for m in 0 ..< M:
      for k in 0 ..< K:
        aTV[m, k] = F16.from(rng.rand(1.0'f32 .. 2.0'f32))
    for n in 0 ..< N:
      for k in 0 ..< K:
        bTV[n, k] = F16.from(rng.rand(1.0'f32 .. 2.0'f32))
    let
      aDev = engine.allocDevice(M * K * sizeof(F16))
      bDev = engine.allocDevice(K * N * sizeof(F16))
      cDev = engine.allocDevice(M * N * sizeof(float32))
      dDev = engine.allocDevice(M * N * sizeof(float32))
    engine.copyToDeviceFromHost(aDev, addr aHost[0], M * K * sizeof(F16))
    engine.copyToDeviceFromHost(bDev, addr bHost[0], K * N * sizeof(F16))
    if prefill:
      # C = A·B, the gemm then reads C from a buffer distinct from D
      engine.run << (grid: (gridX, gridY), blk: (32, 1)) >>
        ("gemmK",
         cDev,
         (dDev, aDev, bDev,
          M, N, K,
          sA[0], sA[1], sB[0], sB[1],
          sC[0], sC[1]))
    engine.run << (grid: (gridX, gridY), blk: (32, 1)) >>
      (kernel,
       dDev,
       (cDev, aDev, bDev,
        M, N, K,
        sA[0], sA[1], sB[0], sB[1],
        sC[0], sC[1]))
    engine.copyToHostFromDevice(addr outHost[0], dDev, M * N * sizeof(float32))
    engine.freeDevice(aDev)
    engine.freeDevice(bDev)
    engine.freeDevice(cDev)
    engine.freeDevice(dDev)
    var bad = 0
    var worst = 0.0'f32
    var first = ""
    for m in 0 ..< M:
      for n in 0 ..< N:
        var acc = 0.0'f32
        var accAbs = 0.0'f32
        for k in 0 ..< K:
          let
            a = float32.from(aTV[m, k])
            b = float32.from(bTV[n, k])
          acc += a * b
          accAbs += abs(a * b)
        let want = (scale + (if prefill: 1.0'f32 else: 0.0'f32)) * acc
        let got = outTV[m, n]
        let d = abs(got - want)
        worst = max(worst, d)
        # the products are exact in f32, only the accumulation order
        # differs from the reference
        if d > 1e-3'f32 + 1e-5'f32 * accAbs:
          inc bad
          if bad == 1: first = &"({m},{n}): got {got} want {want}"
    echo &"  {M}x{N}x{K} A{sA} B{sB} C{sC}" &
         (if gxMul != 1 or gyMul != 1: &" grid x{gxMul}x{gyMul}" else: "") &
         &" worst |d| = {worst}"
    doAssert bad == 0, &"{kernel} {M}x{N}x{K} A{sA} B{sB} C{sC}: " &
      &"{bad} bad, e.g. {first}, worst {worst}"

  # stride permutations at the atom tile size
  for _ in 0 ..< Iters:
    for sA in [(32, 1), (1, 32)]:
      for sB in [(1, 32), (32, 1)]:
        for sC in [(32, 1), (1, 32)]:
          checkMatmul(32, 32, 32, sA, sB, sC)

  # ragged edges, tile dims not multiples of (MT, NT, KT)
  for _ in 0 ..< Iters:
    checkMatmul(32, 32, 41, (41, 1), (1, 32), (32, 1))
    checkMatmul(40, 32, 32, (32, 1), (1, 32), (32, 1))
    checkMatmul(32, 45, 32, (32, 1), (1, 45), (45, 1))
    checkMatmul(40, 45, 41, (41, 1), (1, 45), (45, 1))
    checkMatmul(70, 50, 100, (100, 1), (1, 50), (50, 1))
    checkMatmul(70, 50, 96, (96, 1), (1, 50), (50, 1))

  # excess squadrons, the launch grid over-covers the tile grid
  for _ in 0 ..< Iters:
    checkMatmul(64, 64, 64, (64, 1), (1, 64), (64, 1), gxMul = 3, gyMul = 2)
    checkMatmul(70, 50, 100, (100, 1), (1, 50), (50, 1), gxMul = 2, gyMul = 3)

  # alpha scaling and beta accumulation, the beta case reads a prefilled C
  for _ in 0 ..< Iters:
    checkMatmul(32, 32, 32, (32, 1), (1, 32), (32, 1),
                kernel = "gemmAlphaK", scale = 2.0'f32)
    checkMatmul(32, 32, 32, (32, 1), (1, 32), (32, 1),
                kernel = "gemmAlphaBetaK", scale = 2.0'f32, prefill = true)
