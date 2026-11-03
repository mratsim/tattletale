# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: paged SDPA decode over Metal, effective GB/s (primary) and TFLOP/s.
#
# Effective bytes count only the K/V entries the kernel reads:
#   sum over sequences of attended_seqlen * Hkv * (Dk + Dv) * sizeof(Elem)
# attended_seqlen = seqlen - windowLo, windowLo = max(0, seqlen - windowSize).
# Apply the same byte and flop model on the MLX side (bench_paged_sdpa_decode_mlx.py).
#
# Truth cases need an L2-busting K+V working set: once the K+V bytes fit L2,
# iteration loops measure cache bandwidth, which exceeds the 540 GB/s M4 Max
# DRAM peak. A line whose computed bandwidth exceeds the peak prints INVALID.
#
# Run:
#   nim c -r -d:release \
#     -d:sdpaDtype=f32|bf16 \
#     --outdir:build/bench workspace/ceramic/bench/sdpa_gpu/bench_paged_sdpa_decode_metal.nim

import std/[algorithm, math, monotimes, strformat, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_sdpa_paged
import workspace/ceramic/tests/ceramic_testutils

const sdpaDtype {.strdefine.} = "f32"

when sdpaDtype == "bf16":
  type Elem = bfloat16
else:
  type Elem = float32

const
  ## Bench geometry: one kv tile per page, pool layer 1 of 2.
  ## SdpaPagedTile in tests/kernels/k_sdpa_paged matches this schedule.
  Hq = 32
  Hkv = 8
  Dk = 128
  Dv = 128
  PageSize = 32
  benchMT = 32
  benchNT = 32
  numLayers = 2
  benchLayer = 1
  peakBw = 540.0e9  # M4 Max DRAM peak, bytes per second

proc sdpaBenchLoop(o, q, kPool, vPool, blockTable, cacheSeqlens: distinct TensorView,
                   scale: float32, Dk, Dv, PageSize: static int,
                   Windowed: static bool, windowSize: int, layer: int32, iters: int) =
  ## `iters` attention passes in one launch, idempotent stores
  for _ in 0 ..< iters:
    paged_sdpa_decode(
      tiledMma(float32, q.T, kPool.T, benchMT, benchNT, 32),
      tiledMma(float32, o.T, vPool.T, benchMT, benchNT, 32),
      o, q, kPool, vPool, blockTable, cacheSeqlens,
      scale, Dk, Dv, PageSize, Windowed, windowSize, layer)

proc fillPattern[T, Sh, St](tv: var TensorView[T, Sh, St]) =
  ## Non-uniform deterministic data, the base SDPA benches' pattern
  for i in 0 ..< tv.size():
    tv(i) = to(float32(i mod 7) * 0.25'f32 + 0.5'f32, T)

proc makeFlatHostTensor(n: int): auto =
  ## Rank-1 int32 host tensor over `n` freshly allocated elements.
  ## `makeHostTensor` rejects rank-1 shapes, `originOffset` dies with a type
  ## mismatch on the `(n,)` layout, like `loadFlatFixtureTensor` in tests/kernels/k_sdpa_paged.
  let layout = make_layout((n,))
  result = HostTensor[int32, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[int32]](allocShared(n * sizeof(int32))),
    offset: 0,
    layout: layout)

proc attendedSeqlen(seqlen, windowSize: int, Windowed: static bool): int =
  ## Keys one sequence attends: [windowLo, seqlen), windowLo = max(0, seqlen - windowSize)
  when Windowed:
    seqlen - max(0, seqlen - windowSize)
  else:
    seqlen

proc runCase(engine: MetalEngine, label: string, numSeqs, seqlen: int, Windowed: static bool, windowSize: int, iters: int) =
  ## One decode case: warmup, 3 timed launches, median GB/s and TFLOP/s
  let
    pagesPerSeq = seqlen.ceil_div(PageSize)
    maxPages = pagesPerSeq
    numPages = numSeqs * pagesPerSeq + 1  # +1 spare page, never referenced

  let
    shapeQ = (numSeqs, Hq, 1, Dk)
    shapeO = (numSeqs, Hq, 1, Dv)
    shapePoolK = (numPages, numLayers, PageSize, Hkv, Dk)
    shapePoolV = (numPages, numLayers, PageSize, Hkv, Dv)
  var
    qHost = makeHostTensor(Elem, shapeQ, LayoutRight)
    kHost = makeHostTensor(Elem, shapePoolK, LayoutRight)
    vHost = makeHostTensor(Elem, shapePoolV, LayoutRight)
    oHost = makeHostTensor(Elem, shapeO, LayoutRight)
    btHost = makeHostTensor(int32, (numSeqs, maxPages), LayoutRight)
    csHost = makeFlatHostTensor(numSeqs)

  block:
    var
      qv = qHost.view()
      kv = kHost.view()
      vv = vHost.view()
    fillPattern(qv)
    fillPattern(kv)
    fillPattern(vv)

  # block_table: sequential page ids across sequences, the bench tables fill every slot
  block:
    var btv = btHost.view()
    var nextPage = 0
    for s in 0 ..< numSeqs:
      for p in 0 ..< pagesPerSeq:
        btv[s, p] = int32(nextPage)
        nextPage.inc
    var csv = csHost.view()
    for s in 0 ..< numSeqs:
      csv[s] = int32(seqlen)

  let scaleVal = 1.0'f32 / sqrt(float32(Dk))
  let launchGrid = (1, Hq, numSeqs)
  let windowArg = when Windowed: windowSize else: low(int32)

  # warmup
  block:
    let
      qDev = engine.bridge(qHost)
      kDev = engine.bridge(kHost)
      vDev = engine.bridge(vHost)
      oDev = engine.bridge(oHost)
      btDev = engine.bridge(btHost)
      csDev = engine.bridge(csHost)
    engine.launchDeviceFn sdpaBenchLoop(oDev, qDev, kDev, vDev, btDev, csDev,
        scaleVal, Dk, Dv, PageSize, Windowed, windowArg, int32(benchLayer), 2) do:
      grid = launchGrid
      blk = (32, 1)

  # timed, fresh bridges per rep. The launch overhead is measured.
  # Drain: wait for the iterations before the next rep.
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    block:
      let
        qDev = engine.bridge(qHost)
        kDev = engine.bridge(kHost)
        vDev = engine.bridge(vHost)
        oDev = engine.bridge(oHost)
        btDev = engine.bridge(btHost)
        csDev = engine.bridge(csHost)
      let benchStart = getMonoTime()
      engine.launchDeviceFn sdpaBenchLoop(oDev, qDev, kDev, vDev, btDev, csDev,
          scaleVal, Dk, Dv, PageSize, Windowed, windowArg, int32(benchLayer), iters) do:
        grid = launchGrid
        blk = (32, 1)
      engine.copyToHostFromDevice(addr oHost.data[0], oDev.data, oHost.len())
      timings[rep] = inNanoseconds(getMonoTime() - benchStart) div iters
  sort(timings)
  let
    med = timings[1].float64 / 1e9
    attended = attendedSeqlen(seqlen, windowSize, Windowed)
    bytes = float64(numSeqs * attended * Hkv * (Dk + Dv)) * float64(sizeof(Elem))
    flops = 2.0 * float64(numSeqs * attended) * float64(Hq * (Dk + Dv))
    tflops = flops / med / 1e12
    bw = bytes / med
    bwText = if bw > peakBw: "INVALID (above M4 Max peak)" else: &"{bw / 1e9:>8.1f} GB/s"
    winText = if Windowed: $windowSize else: "-"
  writeLine(stdout, &"{label:>12} seqs={numSeqs:>3} seqlen={seqlen:>5} win={winText:>5}  " &
    &"median {med * 1e6:>10.1f} us  effective {bwText}  {tflops:>6.2f} TFLOP/s")

proc main =
  writeLine(stdout, "Paged SDPA decode Metal bench, B=1 Hq=32 Hkv=8 Dk=Dv=128 " &
    "page=32 tile=32x32 layer=1 of 2 " & sdpaDtype & " (M4 Max)")
  var engine = newEngine(runMetal)
  # L2-resident contrast case: the 67 MB K+V working set fits L2 there.
  # Iteration loops measure cache bandwidth, expect inflated bandwidth.
  runCase(engine, "l2-resident", 8, 1024, Windowed = false, 0, iters = 50)
  # truth cases: the K+V working set busts L2 (>= 2 GB f32)
  runCase(engine, "truth", 32, 8192, Windowed = false, 0, iters = 2)
  runCase(engine, "truth", 64, 8192, Windowed = false, 0, iters = 2)
  # windowed truth case: same pool size, attended bytes = window per sequence
  runCase(engine, "truth-window", 32, 8192, Windowed = true, 1024, iters = 10)

main()
