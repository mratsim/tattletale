# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Template for the k_tile_gemm Metal benchmarks.
## Instantiate through tileGemmBench from a bench_ driver.

import std/[monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/layout_algebra/layouts
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/benchmark/bench_utils
import workspace/fp_conv/fp_conv

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════
#  Kernels
# ═══════════════════════════════════════════════════════════════

proc benchKernels*(): string =
  metal:
    proc gemmK(C: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(A, makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(B, makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
      let dView = make_view(C, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      matmul(mma, dView, aView, bView)
    proc copyK(C: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      ## Load-path-only companion: every loaded register cell reduces into
      ## one scalar stored at the thread's own C slot, so no load is dead,
      ## isolating the load/store cost from the mma cost
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(A, makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(B, makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
      let cGeom = make_view(C, makeIntTuple((M, N)),
                            makeIntTuple((sC0, sC1))).tileWith((32, 32))
      const tvmA = mma.getTVCrdLayout(opA)
      const tvmB = mma.getTVCrdLayout(opB)
      let aGeom = aView.tileWith((32, 16))
      let cGrid = cGeom.getTileGrid()
      let (_, kTiles) = aGeom.getTileGrid()
      let tileID = cGrid.getLocalTileID()
      let thrID = mma.getLocalThreadID()
      if tileID < cGrid.size() and thrID < mma.getThreadCount():
        let (m, n) = idx2crd(cGrid, tileID)
        var rA: rt(float16, mma.getMmaLayout(opA))
        var rB: rt(float16, mma.getMmaLayout(opB))
        var acc = 0.0'f32
        for k in 0 ..< kTiles:
          rA.loadTile(aView, tvmA, thrID, (m, k))
          rB.loadTile(bView, tvmB, thrID, (n, k))
          for i in 0 ..< mma.getMmaLayout(opA).size():
            acc += float32(cast[uint16](rA[i]))
          for i in 0 ..< mma.getMmaLayout(opB).size():
            acc += float32(cast[uint16](rB[i]))
        C[tileID * 32 + thrID] = acc

# ═══════════════════════════════════════════════════════════════
#  Measurement and report
# ═══════════════════════════════════════════════════════════════

template separator*(length: int = 72) =
  echo "=".repeat(length)

template measureMs*(reps: int, body: untyped): float64 =
  ## Elapsed per-iteration milliseconds of one timed block
  let t0 = getMonotime()
  body
  inNanoseconds(getMonotime() - t0).float / 1e6 / reps.float

proc report*(label: string, M, N, K: int, gemmMs, copyMs: float64, rounds, reps: int) =
  let gemmTf = 2.0 * M.float * N.float * K.float / (gemmMs / 1e3) / 1e12
  let netMs = gemmMs - copyMs
  let netTf = 2.0 * M.float * N.float * K.float / (netMs / 1e3) / 1e12
  echo &"  {label} {M}x{N}x{K}: {gemmMs:.3f} ms ({gemmTf:.2f} TF)" &
       &", copy {copyMs:.3f} ms, net {netMs:.3f} ms ({netTf:.2f} TF)" &
       &", {rounds} rounds x {reps} reps"

# ═══════════════════════════════════════════════════════════════
#  Benchmark
# ═══════════════════════════════════════════════════════════════

proc tileGemmBench*(engine: HwEngine, M, N, K: int, warmupSec: float, rounds: int) {.noinline.} =
  ## gemmK and the load-only copyK run interleaved rounds.
  ## Median per kernel, net TF = gemm minus copy.
  const MT = 32
  const NT = 32
  let gridX = (M + MT - 1) div MT
  let gridY = (N + NT - 1) div NT
  var
    aHost = newSeq[F16](M * K)
    bHost = newSeq[F16](K * N)
  let
    aTV = make_view(addr aHost[0], make_layout((M, K), (M, 1)))
    bTV = make_view(addr bHost[0], make_layout((K, N), (N, 1)))
  for m in 0 ..< M:
    for k in 0 ..< K:
      aTV[m, k] = F16.from(float32(1 + (2 * m + 7 * k) mod 13))
  for k in 0 ..< K:
    for n in 0 ..< N:
      bTV[k, n] = F16.from(float32(1 + (3 * k + 11 * n) mod 17))
  let
    aDev = engine.allocDevice(M * K * sizeof(F16))
    bDev = engine.allocDevice(K * N * sizeof(F16))
    dDev = engine.allocDevice(M * N * sizeof(float32))
  engine.copyToDeviceFromHost(aDev, addr aHost[0], M * K * sizeof(F16))
  engine.copyToDeviceFromHost(bDev, addr bHost[0], K * N * sizeof(F16))

  template launch(which: string): untyped =
    engine.run << (grid: (gridX, gridY), blk: (32, 1)) >>
      (which, dDev,
       (aDev, bDev, M, N, K, M, 1, 1, N, M, 1))

  # clock warmup: sustained launches so the GPU clocks ramp
  # before any timed round
  let t0 = getMonotime()
  while inSeconds(getMonotime() - t0).float < warmupSec:
    for r in 0 ..< 20:
      launch("gemmK")

  var gemmMs: seq[float64] = @[]
  var copyMs: seq[float64] = @[]
  let reps = if M >= 4096: 10 elif M >= 2048: 20 else: 50
  for round in 0 ..< rounds:
    block gemmRound:
      launch("gemmK")  # pipeline warmup, untimed
      let ms = measureMs(reps):
        for r in 0 ..< reps:
          launch("gemmK")
      gemmMs.add ms
    block copyRound:
      launch("copyK")
      let ms = measureMs(reps):
        for r in 0 ..< reps:
          launch("copyK")
      copyMs.add ms

  engine.freeDevice(aDev)
  engine.freeDevice(bDev)
  engine.freeDevice(dDev)
  report("gemm", M, N, K, median(gemmMs), median(copyMs), rounds, reps)
