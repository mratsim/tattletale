# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: paged SDPA decode over Metal, effective TFLOP/s.
#
# Effective flops per the non-paged SDPA bench convention (README.md):
#   2 * sum over sequences of attended_seqlen * Hq * (Dk + Dv)
# attended_seqlen[b] = cache_seqlens[b] for full decode,
# min(cache_seqlens[b], windowSize) when the window applies.
# bench_paged_sdpa_decode_mlx.py applies the identical flop model.
# Large-case timing measures steady-state DRAM streaming with the K+V working set above L2.
#
# Run:
#   nim c -r -d:release \
#     -d:sdpaDtype=f32 \
#     --outdir:build/bench workspace/ceramic/bench/sdpa_gpu/bench_paged_sdpa_decode_metal.nim

import std/[algorithm, math, monotimes, strformat, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/positron/src/kernels/k_tile_sdpa_paged
import workspace/ceramic/tests/ceramic_testutils

const sdpaDtype {.strdefine.} = "f32"

when sdpaDtype == "bf16":
  type Elem = bfloat16
else:
  type Elem = float32

const
  Hq = 32
  Hkv = 8
  Dk = 128
  Dv = 128
  PageSize = 32
  benchMT = 32
  benchNT = 32
  numLayers = 2
  benchLayer = 1

proc sdpaBenchLoop(o, q, kPool, vPool, blockTable, cacheSeqlens: distinct TensorView,
                   scale: float32, Dk, Dv, PageSize: static int,
                   Windowed: static bool, windowSize: int, layer: int32, iters: int) =
  ## Runs the identical attention pass `iters` times in one launch.
  for _ in 0 ..< iters:
    paged_sdpa_decode(
      tiledMma(float32, q.T, kPool.T, benchMT, benchNT, 32),
      tiledMma(float32, o.T, vPool.T, benchMT, benchNT, 32),
      o, q, kPool, vPool, blockTable, cacheSeqlens,
      scale, Dk, Dv, PageSize, Windowed, windowSize, layer)

proc fillPattern[T, Sh, St](tv: var TensorView[T, Sh, St]) =
  for i in 0 ..< tv.size():
    tv(i) = to(float32(i mod 7) * 0.25'f32 + 0.5'f32, T)

proc makeFlatHostTensor(n: int): auto =
  let layout = make_layout((n,))
  result = HostTensor[int32, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[int32]](allocShared(n * sizeof(int32))),
    offset: 0,
    layout: layout)

proc attendedSeqlen(seqlen, windowSize: int, Windowed: static bool): int =
  when Windowed:
    seqlen - max(0, seqlen - windowSize)
  else:
    seqlen

proc caseFlops(numSeqs, seqlen, windowSize: int, Windowed: static bool): float64 =
  ## Effective flops: 2 * sum over sequences of attended_seqlen * Hq * (Dk + Dv),
  ## the non-paged SDPA bench convention.
  2.0 * float64(numSeqs * attendedSeqlen(seqlen, windowSize, Windowed)) *
    float64(Hq * (Dk + Dv))

when defined(benchSmokeFlops):
  doAssert caseFlops(8, 1024, 0, Windowed = false) == 134217728.0
  doAssert caseFlops(32, 8192, 0, Windowed = false) == 4294967296.0
  doAssert caseFlops(64, 8192, 0, Windowed = false) == 8589934592.0
  doAssert caseFlops(32, 8192, 1024, Windowed = true) == 536870912.0

proc runCase(engine: MetalEngine, label: string, numSeqs, seqlen: int, Windowed: static bool, windowSize: int, iters: int) =
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

  # Device-to-host copies drain the launch queue, so the measured time includes launch overhead.
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
    flops = caseFlops(numSeqs, seqlen, windowSize, Windowed)
    tflops = flops / med / 1e12
    winText = if Windowed: $windowSize else: "-"
  writeLine(stdout, &"{label:>12} seqs={numSeqs:>3} seqlen={seqlen:>5} win={winText:>5}  " &
    &"median {med * 1e6:>10.1f} us  effective {tflops:>6.2f} TFLOP/s")

proc main =
  var engine = newEngine(runMetal)
  writeLine(stdout, "Paged SDPA decode Metal bench, B=1 Hq=32 Hkv=8 Dk=Dv=128 " &
    "page=32 tile=32x32 layer=1 of 2 " & sdpaDtype & " (" & engine.deviceName() & ")")
  # Contrast case: the K+V working set fits the L2, so iteration measures cache bandwidth, not DRAM bandwidth.
  runCase(engine, "l2-resident", 8, 1024, Windowed = false, 0, iters = 50)
  # Large cases: the K+V working set exceeds L2 and the timing measures steady-state DRAM streaming
  runCase(engine, "dram", 32, 8192, Windowed = false, 0, iters = 2)
  runCase(engine, "dram", 64, 8192, Windowed = false, 0, iters = 2)
  runCase(engine, "dram-window", 32, 8192, Windowed = true, 1024, iters = 10)

when not defined(benchSmokeFlops):
  main()
