# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Bench kernels and driver for k_tile_gemm, backend agnostic
##
## Run:
##   nim c -r -d:release --outdir:build/bench \
##     workspace/ceramic/benchmark/bench_tile_gemm_metal.nim (from tattletale)

import std/[monotimes, strformat, strutils, times, sequtils]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
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
#  Kernel templates and launch descs
# ═══════════════════════════════════════════════════════════════

proc tileLoadReduce(mma: static TiledMma, C, D, A, B: distinct TensorView) =
  ## Load-path-only companion: every loaded register cell reduces into
  ## one scalar stored at the thread's own C slot, so no load is dead,
  ## isolating the load/store cost from the mma cost
  const tvmA = mma.getThreadValueMap(opA)
  const tvmB = mma.getThreadValueMap(opB)
  let cGeom = C.tileWith((32, 32))
  let aGeom = A.tileWith((32, 16))
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
      rA.loadTile(A, tvmA, thrID, (m, k))
      rB.loadTile(B, tvmB, thrID, (n, k))
      for i in 0 ..< mma.getMmaLayout(opA).size():
        acc += float32(cast[uint16](rA[i]))
      for i in 0 ..< mma.getMmaLayout(opB).size():
        acc += float32(cast[uint16](rB[i]))
    C.data[tileID * 32 + thrID] = acc

# ═══════════════════════════════════════════════════════════════
#  Measurement and report
# ═══════════════════════════════════════════════════════════════

proc report*(op, config: string, nsPerOp: float64, tflops = 0.0'f64) =
  ## One line per (op, config), the throughput then the per-op cost
  if tflops > 0.0:
    echo &"{op:<24} {config:<16} {1e9/nsPerOp:>15.3f} ops/s {nsPerOp:>12.0f} ns/op {tflops:>8.2f} TFlops(f32)"
  else:
    echo &"{op:<24} {config:<16} {1e9/nsPerOp:>15.3f} ops/s {nsPerOp:>12.0f} ns/op"

template measure*(iters: int, elapsedNs: untyped, body: untyped): untyped =
  ## Wall time of `iters` runs of `body`, in nanoseconds per op
  let start = getMonotime()
  for _ in 0 ..< iters:
    body
  elapsedNs = inNanoseconds(getMonotime() - start) div iters

proc launchGemm[E: HwEngine](engine: E, C, D, A, B: distinct TensorView) =
  engine.wrap_and_run matmul(tiledMma(float32, float16, float16, 32, 32, 16), C, A, B) do:
    grid = ((C.getLayout().shape[0] + 31) div 32, (C.getLayout().shape[1] + 31) div 32)
    blk = (32, 1)

proc launchLoadReduce[E: HwEngine](engine: E, C, D, A, B: distinct TensorView) =
  engine.wrap_and_run tileLoadReduce(tiledMma(float32, float16, float16, 32, 32, 16), C, D, A, B) do:
    grid = ((C.getLayout().shape[0] + 31) div 32, (C.getLayout().shape[1] + 31) div 32)
    blk = (32, 1)

proc tileGemmBench*[E: HwEngine](engine: E, M, N, K: int, warmupSec: float, rounds: int) {.noinline.} =
  ## gemm and the load-only loadReduce run interleaved rounds, median per kernel, net TF = gemm minus load.
  let one = 1  # runtime int, the layouts stay dynamic
  var
    aHost = newSeq[F16](M * K)
    bHost = newSeq[F16](K * N)
  let
    aTV = make_view(addr aHost[0], make_layout((M, K), (M, one)))
    bTV = make_view(addr bHost[0], make_layout((N, K), (one, N)))
  for m in 0 ..< M:
    for k in 0 ..< K:
      aTV[m, k] = F16.from(float32(1 + (2 * m + 7 * k) mod 13))
  for n in 0 ..< N:
    for k in 0 ..< K:
      bTV[n, k] = F16.from(float32(1 + (3 * k + 11 * n) mod 17))
  let
    aDev = engine.allocDevice(M * K * sizeof(F16))
    bDev = engine.allocDevice(K * N * sizeof(F16))
    dDev = engine.allocDevice(M * N * sizeof(float32))
    dC = make_view(cast[ptr UncheckedArray[float32]](dDev),
      make_layout((M, N), (M, one)))
    dA = make_view(cast[ptr UncheckedArray[float16]](aDev),
      make_layout((M, K), (M, one)))
    dB = make_view(cast[ptr UncheckedArray[float16]](bDev),
      make_layout((N, K), (one, N)))
  engine.copyToDeviceFromHost(aDev, addr aHost[0], M * K * sizeof(F16))
  engine.copyToDeviceFromHost(bDev, addr bHost[0], K * N * sizeof(F16))

  # clock warmup: sustained launches so the GPU clocks ramp
  # before any timed round
  let t0 = getMonotime()
  while inSeconds(getMonotime() - t0).float < warmupSec:
    for r in 0 ..< 20:
      engine.launchGemm(dC, dC, dA, dB)

  let
    config = $M & "x" & $N & "x" & $K
    flops = 2.0 * M.float * N.float * K.float
    reps = if M >= 4096: 10 elif M >= 2048: 25 else: 50
  var
    gemmNs: seq[int64] = @[]
    loadNs: seq[int64] = @[]
  for round in 0 ..< rounds:
    var ns: int64
    measure(reps, ns):
      engine.launchGemm(dC, dC, dA, dB)
    gemmNs.add(ns)
    measure(reps, ns):
      engine.launchLoadReduce(dC, dC, dA, dB)
    loadNs.add(ns)

  engine.freeDevice(aDev)
  engine.freeDevice(bDev)
  engine.freeDevice(dDev)
  let
    gemmMs = median(gemmNs.mapIt(it.float)) / 1e6
    loadMs = median(loadNs.mapIt(it.float)) / 1e6
    netMs = gemmMs - loadMs
  report("gemm", config, gemmMs * 1e6, flops / (gemmMs / 1e3) / 1e12)
  report("load-reduce", config, loadMs * 1e6)
  report("gemm net of load", config, netMs * 1e6, flops / (netMs / 1e3) / 1e12)
