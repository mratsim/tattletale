# Benchmark: positron MoE frontend (moeForward) end to end on Metal, vs the tier kernels alone.
#
# Useful flops 2·T·topk·K·N for the expert FFN contraction, the rest is routing/dispatch/combine overhead.
#
# Run:
#   nim c -r -d:release \
#     -d:moeDtype=f32|f16 \
#     --outdir:build/bench workspace/positron/bench/moe/bench_moe_metal.nim

import std/[algorithm, math, monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/positron/src/kernels/k_grouped_gemv_gemm {.all.}
import workspace/positron/src/kernels/k_moe_forward {.all.}
import workspace/positron/src/kernels/k_moe_router {.all.}
import workspace/positron/tests/positron_testutils
import workspace/positron/tests/k_moe/moe_forward

const moeDtype {.strdefine.} = "f32"

when moeDtype == "f16":
  type Elem = float16
elif moeDtype == "f32":
  type Elem = float32
else:
  {.error: "unsupported -d:moeDtype=" & moeDtype & ", use f32 or f16".}

const
  Kdim = 512
  ## Fixed contraction depth
  Ndim = 512
  ## Fixed output width
  Topk = 6
  ## Fixed routed slots per token

proc fillPattern[T](data: ptr UncheckedArray[T], size: int, period: int, step, offset: float32) =
  ## data[i] = (i mod period)·step + offset, deterministic, no rng
  for i in 0 ..< size:
    data[i] = to(float32(i mod period) * step + offset, T)

template timeMedian(engine: MetalEngine, body: untyped): float64 =
  ## 1 warmup call, median of 3 timed calls, wall-clock in microseconds
  template call: untyped = body
  call()
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    let benchStart = getMonoTime()
    call()
    let benchStop = getMonoTime()
    timings[rep] = inNanoseconds(benchStop - benchStart)
  sort(timings)
  timings[1].float64 / 1e3

proc runCell(engine: MetalEngine, tok, e: int) =
  ## One cell: end to end (both forced tiers), then the two kernel-only references.
  let
    k = int(Kdim)
    n = int(Ndim)
  var
    xHost = makeHostTensor(Elem, (tok, k), (k, 1))
    bHost = makeHostTensor(Elem, (e, k, n), (k * n, n, 1))
    scoresHost = makeHostTensor(float32, (tok, e), (e, 1))
    biasHost = makeHostTensor(float32, (e,), (1,))
    dHost = makeHostTensor(Elem, (tok, n), (n, 1))
  fillPattern(cast[ptr UncheckedArray[Elem]](addr xHost.data[0]), xHost.len, 7, 0.25'f32, 0.5'f32)
  fillPattern(cast[ptr UncheckedArray[Elem]](addr bHost.data[0]), bHost.len, 13, 0.025'f32, 0.15'f32)
  fillPattern(cast[ptr UncheckedArray[float32]](addr scoresHost.data[0]), scoresHost.len, 5, 0.1'f32, -1.0'f32)
  for i in 0 ..< e:
    biasHost.data[i] = 0'f32

  let
    routedRows = tok * Topk
    rowsPerExpert = (routedRows + e - 1) div e
    pickedTier = pickGroupedTier(rowsPerExpert)
    (form, kSplits) = groupedGemvForm(tok, n)
    flops = 2.0e0 * float64(routedRows) * float64(k) * float64(n)
  let tierName = if pickedTier == gtGemv:
    if form == ggfSplitK: "gemv/splitk/" & $kSplits else: "gemv/single"
  else: "gemm"

  template e2eCall(forceTier: int): untyped =
    moeForward(engine, dHost, xHost, scoresHost, biasHost, bHost, Topk,
      moeSoftmax, 1.0'f32, forceTier)
  let
    e2eUs = timeMedian(engine): e2eCall(-1)
    gemvForcedUs = timeMedian(engine): e2eCall(0)
    gemmForcedUs = timeMedian(engine): e2eCall(1)
  # Expert weights' staging copy alone, device buffer pre-allocated
  let bDevStage = engine.bridge(bHost)
  let bStageUs = timeMedian(engine):
    engine.copyToDeviceFromHost(bDevStage.data, bHost.data.toOpenArray(0, bHost.len() - 1))

  # Kernel-only references, staged once, launches timed
  block:
    let
      idsHost = makeHostTensor(int32, (tok, Topk), (Topk, 1))
      weightsHost = makeHostTensor(float32, (tok, Topk), (Topk, 1))
    for t in 0 ..< tok:
      for i in 0 ..< Topk:
        idsHost[t, i] = int32((t * 7 + i * 13) mod e)
        weightsHost[t, i] = 1.0'f32 / float32(Topk)
    let
      xDev = engine.bridge(xHost)
      bDev = engine.bridge(bHost)
      idsDev = engine.bridge(idsHost)
      weightsDev = engine.bridge(weightsHost)
      dDev = engine.bridge(dHost)
    template makeDevView(dev, sh, st: untyped): untyped =
      let lay = make_layout(sh, st)
      BridgeTensor[typeof(dev), typeof(lay)](data: dev, layout: lay)
    # Split-K partials stage once, outside the timed loop
    let pDev = if form == ggfSplitK:
      engine.allocDevice(float32, tok * kSplits * n)
    else:
      engine.allocDevice(float32, 1)
    let pView = makeDevView(pDev, (tok, kSplits, n),
      (kSplits * n, n, 1))
    let gemvKernelUs = timeMedian(engine):
      if form == ggfSplitK:
        engine.launchDeviceFn grouped_gemv_splitk(pView, xDev, bDev, idsDev, weightsDev, Topk, kSplits) do:
          grid = (n.ceil_div(32 * GroupedGemvOutputs) * kSplits, tok)
          blk = (32, 1)
        engine.launchDeviceFn grouped_gemv_reduce(dDev, pView, kSplits) do:
          grid = (n.ceil_div(32 * GroupedGemvOutputs), tok)
          blk = (32, 1)
      else:
        engine.launchDeviceFn grouped_gemv_via_expertIds(dDev, xDev, bDev, idsDev, weightsDev, Topk) do:
          grid = (n.ceil_div(32 * GroupedGemvOutputs), tok)
          blk = (32, 1)
    # M3's packed best case: totalM = T·topk valid lines with the landed M4 bench
    # id pattern (line r -> expert r mod e), every line valid, host pack NOT charged
    let totalM = tok * Topk
    var
      aHost = makeHostTensor(Elem, (totalM, k), (k, 1))
      idsFlatHost = makeHostTensor(int32, (totalM,), (1,))
    fillPattern(cast[ptr UncheckedArray[Elem]](addr aHost.data[0]), aHost.len, 7, 0.25'f32, 0.5'f32)
    for r in 0 ..< totalM:
      idsFlatHost.data[r] = int32(r mod e)
    let
      aDev = engine.bridge(aHost)
      idsFlatDev = engine.bridge(idsFlatHost)
      dPackedDev = engine.bridge(dHost)
      totalTiles = countGroupedGemmTiles(Elem, totalM, n)
    let gemmKernelUs = timeMedian(engine):
      engine.launchDeviceFn grouped_gemm_via_expertIds(dPackedDev, aDev, bDev, idsFlatDev) do:
        grid = (totalTiles, 1)
        blk = (32, 1)
    let kernelOnly = if pickedTier == gtGemv: gemvKernelUs else: gemmKernelUs
    let frontend = e2eUs - bStageUs - kernelOnly
    writeLine(stdout, &"T={tok:>4} topk={Topk} E={e:>3} tier={tierName:<14} " &
      &"e2e {e2eUs:>9.1f} us  forced-gemv {gemvForcedUs:>9.1f}  forced-gemm {gemmForcedUs:>9.1f}  " &
      &"b-stage {bStageUs:>8.1f}  gemv-kernel {gemvKernelUs:>9.1f}  gemm-kernel {gemmKernelUs:>9.1f}  " &
      &"{flops / (e2eUs * 1e-6) / 1e9:>8.1f} GF/s  frontend {frontend:>9.1f} us")

proc main =
  writeLine(stdout, "moeForward end to end vs tier kernels alone, K=N=512 topk=6, " & moeDtype &
    " (M4 Max), median of 3 moeForward calls after 1 warmup, staging paid every call")
  var engine = newEngine(runMetal)
  for e in [8, 64, 512]:
    for tok in [1, 8, 64, 512, 4096]:
      runCell(engine, tok, e)

main()
