# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Benchmark:
##   k_tile_gemm throughput on Metal
##
## - gemmK and the load-only copyK run interleaved rounds,
##   median per kernel, net TF = gemm minus copy
## - the GPU clocks ramp during a warmup window before any timed round
##
## Run:
##   nim c -r -d:benchQuick --outdir:build/bench \
##     workspace/ceramic/benchmark/bench_tile_gemm_metal.nim (from tattletale)
##
## -d:benchQuick: 2s warmup, 2 rounds (full: 8s warmup, 5 rounds)
## CLI size filter: ./build/bench/bench_tile_gemm_metal 4096 runs only 4096.

import std/[monotimes, strformat, times, os, strutils]
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

{.experimental: "callOperator".}

const
  ProblemSizes = [1024, 2048, 4096]
  WarmupSeconds = when defined(benchQuick): 2.0 else: 8.0
  Rounds = when defined(benchQuick): 2 else: 5

# ═══════════════════════════════════════════════════════════════
#  Kernels
# ═══════════════════════════════════════════════════════════════

proc benchKernels(): string =
  metal:
    proc gemmK(C, A, B: ptr UncheckedArray[float32], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(cast[ptr UncheckedArray[float16]](A),
                            makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(cast[ptr UncheckedArray[float16]](B),
                            makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
      let dView = make_view(C, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      matmul(mma, dView, aView, bView)
    proc copyK(C, A, B: ptr UncheckedArray[float32], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      ## Load-path-only companion: every loaded register cell reduces into
      ## one scalar stored at the thread's own C slot, so no load is dead,
      ## isolating the load/store cost from the mma cost
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(cast[ptr UncheckedArray[float16]](A),
                            makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(cast[ptr UncheckedArray[float16]](B),
                            makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
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
#  Test data
# ═══════════════════════════════════════════════════════════════

const MT = 32
const NT = 32
const KT = 16

func fp32ToFp16(x: float32): uint16 =
  let u = cast[uint32](x)
  let sign = uint16((u shr 16) and 0x8000'u32)
  let aexp = int32((u shr 23) and 0xFF)
  let mant = u and 0x7FFFFF'u32
  if aexp == 0xFF:
    return sign or 0x7C00'u16
  var bexp = aexp - 127 + 15
  if bexp >= 31:
    return sign or 0x7C00'u16
  if bexp <= 0:
    if bexp < -10: return sign
    let m2 = mant or 0x800000'u32
    let shift = uint32(126 - aexp)
    let half = 1'u32 shl (shift - 1)
    var r = m2 shr shift
    let rem = m2 and ((1'u32 shl shift) - 1)
    if rem > half or (rem == half and (r and 1) == 1):
      r += 1
    return sign or uint16(r)
  let half = 0x1000'u32
  var r = mant shr 13
  let rem = mant and 0x1FFF'u32
  if rem > half or (rem == half and (r and 1) == 1):
    r += 1
    if r == 0x400:
      bexp += 1
      r = 0
      if bexp >= 31:
        return sign or 0x7C00'u16
  return sign or uint16(bexp shl 10) or uint16(r)

func flops(M, N, K: int): float {.inline.} =
  2.0 * M.float * N.float * K.float

func repsFor(M: int): int =
  # small sizes finish fast, more reps per round smooths the launch noise
  if M >= 4096: 10 elif M >= 2048: 20 else: 50

# ═══════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════

proc report(label: string, M, N, K: int, gemmMs, copyMs: float64, reps: int) =
  let gemmTf = flops(M, N, K) / (gemmMs / 1e3) / 1e12
  let netMs = gemmMs - copyMs
  let netTf = flops(M, N, K) / (netMs / 1e3) / 1e12
  echo &"  {label} {M}x{N}x{K}: {gemmMs:.3f} ms ({gemmTf:.2f} TF)" &
       &", copy {copyMs:.3f} ms, net {netMs:.3f} ms ({netTf:.2f} TF)" &
       &", {Rounds} rounds x {reps} reps"

# ═══════════════════════════════════════════════════════════════
#  Measurement
# ═══════════════════════════════════════════════════════════════

proc benchSize(engine: HwEngine, M, N, K: int) =
  let gridX = (M + MT - 1) div MT
  let gridY = (N + NT - 1) div NT
  var
    aHost = newSeq[uint16](M * K)
    bHost = newSeq[uint16](K * N)
  for m in 0 ..< M:
    for k in 0 ..< K:
      aHost[m * K + k] = fp32ToFp16(float32(1 + (2 * m + 7 * k) mod 13))
  for k in 0 ..< K:
    for n in 0 ..< N:
      bHost[k * N + n] = fp32ToFp16(float32(1 + (3 * k + 11 * n) mod 17))
  let
    aDev = engine.allocDevice(M * K * sizeof(uint16))
    bDev = engine.allocDevice(K * N * sizeof(uint16))
    dDev = engine.allocDevice(M * N * sizeof(float32))
  engine.copyToDeviceFromHost(aDev, addr aHost[0], M * K * sizeof(uint16))
  engine.copyToDeviceFromHost(bDev, addr bHost[0], K * N * sizeof(uint16))

  template launch(which: string): untyped =
    engine.run << (grid: (gridX, gridY), blk: (32, 1)) >>
      (which, dDev,
       (aDev, bDev, M, N, K, M, 1, 1, N, M, 1))

  # clock warmup: sustained launches so the GPU clocks ramp
  # before any timed round
  let t0 = getMonotime()
  while inSeconds(getMonotime() - t0).float < WarmupSeconds:
    for r in 0 ..< 20:
      launch("gemmK")

  var gemmMs: seq[float64] = @[]
  var copyMs: seq[float64] = @[]
  let reps = repsFor(M)
  for round in 0 ..< Rounds:
    block gemmRound:
      launch("gemmK")  # pipeline warmup, untimed
      let t0 = getMonotime()
      for r in 0 ..< reps:
        launch("gemmK")
      gemmMs.add inNanoseconds(getMonotime() - t0).float / 1e6 / reps.float
    block copyRound:
      launch("copyK")
      let t0 = getMonotime()
      for r in 0 ..< reps:
        launch("copyK")
      copyMs.add inNanoseconds(getMonotime() - t0).float / 1e6 / reps.float

  let gemmBest = median(gemmMs)
  let copyBest = median(copyMs)
  engine.freeDevice(aDev)
  engine.freeDevice(bDev)
  engine.freeDevice(dDev)
  report("gemm", M, N, K, gemmBest, copyBest, reps)

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

proc main =
  var engine = bkMetal.init()
  engine.ingest(benchKernels())
  echo "BENCH_START"
  # CLI size filter: bench 4096 runs only that size
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  for mnk in ProblemSizes:
    if wanted.len == 0 or mnk in wanted:
      engine.benchSize(mnk, mnk, mnk)

when isMainModule:
  main()
