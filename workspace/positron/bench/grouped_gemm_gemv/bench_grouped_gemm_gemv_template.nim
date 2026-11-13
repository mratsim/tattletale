## Grouped gemv/gemm bench machinery shared by the four drivers, per backend.
## Useful flops 2·E·M·K·N per launch, identical for every column.
import std/[algorithm, monotimes, times, os, strutils, strformat]
import workspace/math
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/positron/src/kernels/k_grouped_gemv_gemm {.all.}
import workspace/ceramic/tests/ceramic_testutils

export int_tuples, layout_algebra, tensors, tile_algebra, k_tile_gemm, k_grouped_gemv_gemm

const
  Kdim = 512
  Ndim = 512

# ═══════════════════════════════════════════════════════════════════════
#  Deterministic pattern fill
# ═══════════════════════════════════════════════════════════════════════

proc fillPattern[T](data: ptr UncheckedArray[T], size: int, period: int, step, offset: float32) =
  for i in 0 ..< size:
    data[i] = to(float32(i mod period) * step + offset, T)

# ═══════════════════════════════════════════════════════════════════════
#  Device staging
# ═══════════════════════════════════════════════════════════════════════

template makeDeviceView(dev, sh, st: untyped): untyped =
  let lay = make_layout(sh, st)
  BridgeTensor[typeof(dev), typeof(lay)](data: dev, layout: lay)

template stagePatternSlab(engine: MetalEngine, T: typedesc, pattern: ptr UncheckedArray[T], m, n: int): untyped =
  let dev = engine.allocDevice(T, int(m) * int(n))
  engine.copyToDeviceFromHost(dev, pattern.toOpenArray(0, int(m) * int(n) - 1))
  makeDeviceView(dev, (m, n), (n, 1))

template stageGroupedSlabs(engine: MetalEngine, T: typedesc, e: int, m, k, n: int): untyped =
  ## Grouped-contract slabs: a (e·m, k), b (e, k, n), offs ends at (g+1)·m, d uninitialized.
  var
    aPattern = newSeq[T](e * int(m) * int(k))
    bPattern = newSeq[T](e * int(k) * int(n))
    offsPattern = newSeq[int32](e)
  fillPattern(cast[ptr UncheckedArray[T]](addr aPattern[0]), aPattern.len, 7, 0.25'f32, 0.5'f32)
  fillPattern(cast[ptr UncheckedArray[T]](addr bPattern[0]), bPattern.len, 13, 0.025'f32, 0.15'f32)
  for g in 0 ..< e:
    offsPattern[g] = int32((g + 1) * m)
  let
    aView = stagePatternSlab(engine, T, cast[ptr UncheckedArray[T]](addr aPattern[0]),
                             e * m, k)
    bDev = engine.allocDevice(T, bPattern.len)
    offsDev = engine.allocDevice(int32, offsPattern.len)
    dDev = engine.allocDevice(T, e * int(m) * int(n))
  engine.copyToDeviceFromHost(bDev, bPattern)
  engine.copyToDeviceFromHost(offsDev, offsPattern)
  let
    bView = makeDeviceView(bDev, (e, k, n), (k * n, n, 1))
    offsView = makeDeviceView(offsDev, (e,), (1,))
    dView = makeDeviceView(dDev, (e * m, n), (n, 1))
  (aView, bView, offsView, dView)

template stageDenseB(engine: MetalEngine, T: typedesc, k, n: int): untyped =
  var bPattern = newSeq[T](int(k) * int(n))
  fillPattern(cast[ptr UncheckedArray[T]](addr bPattern[0]), bPattern.len, 13, 0.025'f32, 0.15'f32)
  let bDev = engine.allocDevice(T, bPattern.len)
  engine.copyToDeviceFromHost(bDev, bPattern)
  makeDeviceView(bDev, (n, k), (1, n))

proc denseMatmulCombo*(mma: static TiledMma, d, a, b: distinct TensorView) =
  matmul(mma, d, a, b)

# ═══════════════════════════════════════════════════════════════════════
#  via_offsets columns
# ═══════════════════════════════════════════════════════════════════════

proc runGemvColumn*[T](_: typedesc[T], engine: MetalEngine, e: int, m, k, n: int): float64 =
  ## GB/s over the traffic model: every line reads its expert's full b matrix
  ## (m·k·n b bytes dominate), plus a lines and dst writes.
  var (aView, bView, offsView, dView) = stageGroupedSlabs(engine, T, e, m, k, n)
  let elt = sizeof(T).float64
  let bytes = float64(e * m * (k * n + k + n)) * elt
  engine.launchDeviceFn grouped_gemv_via_offsets(dView, aView, bView, offsView) do:
    grid = (int(n).ceil_div(32 * GroupedGemvOutputs), e * int(m))
    blk = (32, 1)
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    let benchStart = getMonoTime()
    engine.launchDeviceFn grouped_gemv_via_offsets(dView, aView, bView, offsView) do:
      grid = (int(n).ceil_div(32 * GroupedGemvOutputs), e * int(m))
      blk = (32, 1)
    let benchStop = getMonoTime()
    timings[rep] = inNanoseconds(benchStop - benchStart)
  sort(timings)
  result = bytes / float64(timings[1])

proc runGroupedGemmColumn*[T](_: typedesc[T], engine: MetalEngine, e: int, m, k, n: int): float64 =
  var (aView, bView, offsView, dView) = stageGroupedSlabs(engine, T, e, m, k, n)
  let
    totalTiles = countGroupedGemmTiles(T, e * int(m), int(n))
    flops = 2.0e0 * float64(e) * float64(m) * float64(k) * float64(n)
  engine.launchDeviceFn grouped_gemm_via_offsets(dView, aView, bView, offsView) do:
    grid = (totalTiles, 1)
    blk = (32, 1)
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    let benchStart = getMonoTime()
    engine.launchDeviceFn grouped_gemm_via_offsets(dView, aView, bView, offsView) do:
      grid = (totalTiles, 1)
      blk = (32, 1)
    let benchStop = getMonoTime()
    timings[rep] = inNanoseconds(benchStop - benchStart)
  sort(timings)
  result = flops / float64(timings[1])

proc runDenseColumn*[T](_: typedesc[T], engine: MetalEngine, e: int, m, k, n: int): float64 =
  ## Same pattern values and tile config as the grouped columns, D fp32 for every dtype
  ## (the matmul's register accumulator is fp32): the columns differ in schedule alone.
  var
    aPattern = newSeq[T](e * int(m) * int(k))
  fillPattern(cast[ptr UncheckedArray[T]](addr aPattern[0]), aPattern.len, 7, 0.25'f32, 0.5'f32)
  let
    aView = stagePatternSlab(engine, T, cast[ptr UncheckedArray[T]](addr aPattern[0]),
                             e * m, k)
    bView = stageDenseB(engine, T, k, n)
    dDev = engine.allocDevice(float32, e * int(m) * int(n))
    dView = makeDeviceView(dDev, (e * m, n), (n, 1))
  const (MT, NT, KT) = groupedGemmTiles(T)
  let
    denseTilesM = ceil_div(e * int(m), MT)
    denseTilesN = ceil_div(int(n), NT)
    flops = 2.0e0 * float64(e) * float64(m) * float64(k) * float64(n)
  engine.launchDeviceFn denseMatmulCombo(
      tiledMma(float32, T, T, MT, NT, KT), dView, aView, bView) do:
    grid = (denseTilesM, denseTilesN)
    blk = (32, 1)
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    let benchStart = getMonoTime()
    engine.launchDeviceFn denseMatmulCombo(
        tiledMma(float32, T, T, MT, NT, KT), dView, aView, bView) do:
      grid = (denseTilesM, denseTilesN)
      blk = (32, 1)
    let benchStop = getMonoTime()
    timings[rep] = inNanoseconds(benchStop - benchStart)
  sort(timings)
  result = flops / float64(timings[1])

# ═══════════════════════════════════════════════════════════════════════
#  via_offsets cells and mains
# ═══════════════════════════════════════════════════════════════════════

proc runGemvCell(engine: MetalEngine, e: int, m: int, T: typedesc) =
  when T is float16:
    const name = "fp16"
  else:
    const name = "fp32"
  let
    k = Kdim
    n = Ndim
    gemvGbs = runGemvColumn(T, engine, e, m, k, n)
    gemmGf = runGroupedGemmColumn(T, engine, e, m, k, n)
  writeLine(stdout, &"E={e:>4} M={int(m):>5} {name}  gemv {gemvGbs:>10.2f} GB/s  gemm {gemmGf:>10.2f} GFLOP/s")

proc runGemmCell(engine: MetalEngine, e: int, m: int, T: typedesc) =
  when T is float16:
    const name = "fp16"
  else:
    const name = "fp32"
  let
    k = Kdim
    n = Ndim
    groupedGf = runGroupedGemmColumn(T, engine, e, m, k, n)
    denseGf = runDenseColumn(T, engine, e, m, k, n)
  writeLine(stdout, &"E={e:>4} M={int(m):>5} {name}  grouped {groupedGf:>10.2f}  dense {denseGf:>10.2f} GFLOP/s")

proc checkDtypeArg(dtype: string) =
  if dtype notin ["fp32", "fp16"]:
    writeLine(stderr, &"error: unknown dtype '{dtype}', expected fp32 or fp16")
    quit(1)

proc mainGroupedGemvViaOffsets* =
  ## grouped_gemv_via_offsets bench:
  ## E and M select one cell, M is the per-group M_g, no arguments run the full sweep over both dtypes.
  let args = commandLineParams()
  var engine = newEngine(runMetal)
  if args.len >= 2:
    let
      e = parseInt(args[0])
      m = parseInt(args[1])
      dtype = if args.len >= 3: args[2] else: "fp32"
    checkDtypeArg(dtype)
    writeLine(stdout, &"grouped_gemv_via_offsets bench, K={Kdim} N={Ndim}, one cell")
    if dtype == "fp16":
      runGemvCell(engine, e, m, float16)
    else:
      runGemvCell(engine, e, m, float32)
  else:
    writeLine(stdout, &"grouped_gemv_via_offsets bench, full sweep, K={Kdim} N={Ndim}, M4 Max")
    for m in [1, 8]:
      runGemvCell(engine, 64, m, float32)
      runGemvCell(engine, 64, m, float16)
    runGemvCell(engine, 512, 64, float32)
    runGemvCell(engine, 512, 64, float16)

proc mainGroupedGemmViaOffsets* =
  ## grouped_gemm_via_offsets bench:
  ## E and M select one cell, M is the per-group M_g, no arguments run the fp32 full sweep.
  let args = commandLineParams()
  var engine = newEngine(runMetal)
  if args.len >= 2:
    let
      e = parseInt(args[0])
      m = parseInt(args[1])
      dtype = if args.len >= 3: args[2] else: "fp32"
    checkDtypeArg(dtype)
    writeLine(stdout, &"grouped_gemm_via_offsets bench, K={Kdim} N={Ndim}, one cell")
    if dtype == "fp16":
      runGemmCell(engine, e, m, float16)
    else:
      runGemmCell(engine, e, m, float32)
  else:
    writeLine(stdout, &"grouped_gemm_via_offsets bench, full sweep, K={Kdim} N={Ndim}, M4 Max")
    for m in [1, 8, 64]:
      runGemmCell(engine, 512, m, float32)

# ═══════════════════════════════════════════════════════════════════════
#  via_expertIds benches
# ═══════════════════════════════════════════════════════════════════════

const groupedDtype {.strdefine.} = "f32"

when groupedDtype == "f16":
  type Elem = float16
elif groupedDtype == "f32":
  type Elem = float32
else:
  {.error: "unsupported -d:groupedDtype=" & groupedDtype & ", use f32 or f16".}

template stageSlabs(engine: MetalEngine, e: int, m: int, padded: static bool): untyped =
  ## grouped_gemm_via_expertIds operands over four slabs, declared in the caller's scope:
  ## - aView: one (totalM, k) slab, every expert block's lines concatenated, pattern-filled
  ## - bView: one (e, k, n) stack, one (k, n) matrix per expert, pattern-filled
  ## - idsView: (totalM,) int32 per-line expert ids, -1 on the padding tails
  ## - dView: one uninitialized (totalM, n) destination
  ##
  ## A template: DeviceMem handles are move-only and never cross a proc boundary.
  let perBlock = when padded: (int(m) + 31) div 32 * 32 else: int(m)
  let totalM = e * perBlock
  var
    aPattern = newSeq[Elem](totalM * int(Kdim))
    bPattern = newSeq[Elem](e * int(Kdim) * int(Ndim))
    idsPattern = newSeq[int32](totalM)
  fillPattern(cast[ptr UncheckedArray[Elem]](addr aPattern[0]), aPattern.len, 7, 0.25'f32, 0.5'f32)
  fillPattern(cast[ptr UncheckedArray[Elem]](addr bPattern[0]), bPattern.len, 13, 0.025'f32, 0.15'f32)
  for g in 0 ..< e:
    for local in 0 ..< perBlock:
      idsPattern[g * perBlock + local] =
        int32(if local < m: g else: -1)
  let
    aDev = engine.allocDevice(Elem, aPattern.len)
    bDev = engine.allocDevice(Elem, bPattern.len)
    idsDev = engine.allocDevice(int32, idsPattern.len)
    dDev = engine.allocDevice(Elem, totalM * int(Ndim))
  engine.copyToDeviceFromHost(aDev, aPattern)
  engine.copyToDeviceFromHost(bDev, bPattern)
  engine.copyToDeviceFromHost(idsDev, idsPattern)
  let
    kdim = int(Kdim)
    ndim = int(Ndim)
    aView = makeDeviceView(aDev, (totalM, kdim), (kdim, 1))
    bView = makeDeviceView(bDev, (e, kdim, ndim),
                           (kdim * ndim, ndim, 1))
    idsView = makeDeviceView(idsDev, (totalM,), (1,))
    dView = makeDeviceView(dDev, (totalM, ndim), (ndim, 1))
  (aView, bView, idsView, dView)

proc runDecodeCell(engine: MetalEngine, tok, topk, e: int) =
  ## One via_expertIds gemv cell: 1 warmup launch, median of 3 timed launches.
  var
    xPattern = newSeq[Elem](tok * int(Kdim))
    bPattern = newSeq[Elem](e * int(Kdim) * int(Ndim))
    idsPattern = newSeq[int32](tok * topk)
    weightsPattern = newSeq[float32](tok * topk)
  fillPattern(cast[ptr UncheckedArray[Elem]](addr xPattern[0]), xPattern.len, 7, 0.25'f32, 0.5'f32)
  fillPattern(cast[ptr UncheckedArray[Elem]](addr bPattern[0]), bPattern.len, 13, 0.025'f32, 0.15'f32)
  for t in 0 ..< tok:
    for i in 0 ..< topk:
      idsPattern[t * topk + i] = int32((t * 7 + i * 13) mod e)
      weightsPattern[t * topk + i] = 1.0'f32 / float32(topk)
  let
    xDev = engine.allocDevice(Elem, xPattern.len)
    bDev = engine.allocDevice(Elem, bPattern.len)
    idsDev = engine.allocDevice(int32, idsPattern.len)
    weightsDev = engine.allocDevice(float32, weightsPattern.len)
    dDev = engine.allocDevice(Elem, tok * int(Ndim))
  engine.copyToDeviceFromHost(xDev, xPattern)
  engine.copyToDeviceFromHost(bDev, bPattern)
  engine.copyToDeviceFromHost(idsDev, idsPattern)
  engine.copyToDeviceFromHost(weightsDev, weightsPattern)
  let
    kdim = int(Kdim)
    ndim = int(Ndim)
    xView = makeDeviceView(xDev, (tok, kdim), (kdim, 1))
    bView = makeDeviceView(bDev, (e, kdim, ndim),
                           (kdim * ndim, ndim, 1))
    idsView = makeDeviceView(idsDev, (tok, topk), (topk, 1))
    weightsView = makeDeviceView(weightsDev, (tok, topk), (topk, 1))
    dView = makeDeviceView(dDev, (tok, ndim), (ndim, 1))
    # traffic: every (token, slot) pair reads its expert's full b matrix
    # (tok·topk·K·N b bytes dominate), plus x lines, dst writes, ids, weights
    bytes = float64(tok * topk) * (float64(Kdim * Ndim) * sizeof(Elem).float64 + 8.0) +
      float64(tok) * float64(Kdim + Ndim) * sizeof(Elem).float64
  engine.launchDeviceFn grouped_gemv_via_expertIds(dView, xView, bView, idsView, weightsView, topk) do:
    grid = (ndim.ceil_div(32 * GroupedGemvOutputs), tok)
    blk = (32, 1)
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    let benchStart = getMonoTime()
    engine.launchDeviceFn grouped_gemv_via_expertIds(dView, xView, bView, idsView, weightsView, topk) do:
      grid = (ndim.ceil_div(32 * GroupedGemvOutputs), tok)
      blk = (32, 1)
    let benchStop = getMonoTime()
    timings[rep] = inNanoseconds(benchStop - benchStart)
  sort(timings)
  let med = timings[1].float64 / 1e9
  writeLine(stdout, &"T={tok:>4} topk={topk} E={e:>4}  gemv {bytes / med / 1e9:>9.1f} GB/s  {med * 1e6:>10.1f} us")

proc runSplitKCell(engine: MetalEngine, tok, topk, e, kSplits: int) =
  ## One split-K cell: stage 1 + stage 2 back to back, 1 warmup pair, median of 3 timed pairs.
  ## The timed unit is both launches: the form's cost is the pair.
  var
    xPattern = newSeq[Elem](tok * int(Kdim))
    bPattern = newSeq[Elem](e * int(Kdim) * int(Ndim))
    idsPattern = newSeq[int32](tok * topk)
    weightsPattern = newSeq[float32](tok * topk)
    partialsPattern = newSeq[float32](tok * kSplits * int(Ndim))
  fillPattern(cast[ptr UncheckedArray[Elem]](addr xPattern[0]), xPattern.len, 7, 0.25'f32, 0.5'f32)
  fillPattern(cast[ptr UncheckedArray[Elem]](addr bPattern[0]), bPattern.len, 13, 0.025'f32, 0.15'f32)
  for t in 0 ..< tok:
    for i in 0 ..< topk:
      idsPattern[t * topk + i] = int32((t * 7 + i * 13) mod e)
      weightsPattern[t * topk + i] = 1.0'f32 / float32(topk)
  let
    xDev = engine.allocDevice(Elem, xPattern.len)
    bDev = engine.allocDevice(Elem, bPattern.len)
    idsDev = engine.allocDevice(int32, idsPattern.len)
    weightsDev = engine.allocDevice(float32, weightsPattern.len)
    dDev = engine.allocDevice(Elem, tok * int(Ndim))
    pDev = engine.allocDevice(float32, partialsPattern.len)
  engine.copyToDeviceFromHost(xDev, xPattern)
  engine.copyToDeviceFromHost(bDev, bPattern)
  engine.copyToDeviceFromHost(idsDev, idsPattern)
  engine.copyToDeviceFromHost(weightsDev, weightsPattern)
  let
    kdim = int(Kdim)
    ndim = int(Ndim)
    xView = makeDeviceView(xDev, (tok, kdim), (kdim, 1))
    bView = makeDeviceView(bDev, (e, kdim, ndim),
                           (kdim * ndim, ndim, 1))
    idsView = makeDeviceView(idsDev, (tok, topk), (topk, 1))
    weightsView = makeDeviceView(weightsDev, (tok, topk), (topk, 1))
    dView = makeDeviceView(dDev, (tok, ndim), (ndim, 1))
    pView = makeDeviceView(pDev, (tok, kSplits, ndim),
                           (kSplits * ndim, ndim, 1))
    # traffic: the b reads of the single-launch form plus the partials round trip
    bytes = float64(tok * topk) * (float64(Kdim * Ndim) * sizeof(Elem).float64 + 8.0) +
      float64(tok) * float64(Kdim + Ndim) * sizeof(Elem).float64 +
      float64(tok) * float64(kSplits * Ndim * 8)
  template launchPair: untyped =
    engine.launchDeviceFn grouped_gemv_splitk(pView, xView, bView, idsView, weightsView, topk, kSplits) do:
      grid = (ndim.ceil_div(32 * GroupedGemvOutputs) * kSplits, tok)
      blk = (32, 1)
    engine.launchDeviceFn grouped_gemv_reduce(dView, pView, kSplits) do:
      grid = (ndim.ceil_div(32 * GroupedGemvOutputs), tok)
      blk = (32, 1)
  launchPair()
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    let benchStart = getMonoTime()
    launchPair()
    let benchStop = getMonoTime()
    timings[rep] = inNanoseconds(benchStop - benchStart)
  sort(timings)
  let med = timings[1].float64 / 1e9
  writeLine(stdout, &"                                gemv-splitk/{kSplits} {bytes / med / 1e9:>9.1f} GB/s  {med * 1e6:>10.1f} us")

proc runMatchedGemmCell(engine: MetalEngine, tok, topk, e: int) =
  ## Matched gemm cell: grouped_gemm_via_expertIds over the packed equivalent.
  ##
  ## totalM = T·topk lines, sorted expert-major, every line valid, no -1 padding.
  ## The host-side pack is not charged (best case for the gemm tier).
  let totalM = tok * topk
  var
    aPattern = newSeq[Elem](totalM * int(Kdim))
    bPattern = newSeq[Elem](e * int(Kdim) * int(Ndim))
    idsPattern = newSeq[int32](totalM)
  fillPattern(cast[ptr UncheckedArray[Elem]](addr aPattern[0]), aPattern.len, 7, 0.25'f32, 0.5'f32)
  fillPattern(cast[ptr UncheckedArray[Elem]](addr bPattern[0]), bPattern.len, 13, 0.025'f32, 0.15'f32)
  # expert-major lines: line r belongs to expert r mod e, already ascending in r
  for r in 0 ..< totalM:
    idsPattern[r] = int32(r mod e)
  let
    aDev = engine.allocDevice(Elem, aPattern.len)
    bDev = engine.allocDevice(Elem, bPattern.len)
    idsDev = engine.allocDevice(int32, idsPattern.len)
    dDev = engine.allocDevice(Elem, totalM * int(Ndim))
  engine.copyToDeviceFromHost(aDev, aPattern)
  engine.copyToDeviceFromHost(bDev, bPattern)
  engine.copyToDeviceFromHost(idsDev, idsPattern)
  let
    kdim = int(Kdim)
    ndim = int(Ndim)
    aView = makeDeviceView(aDev, (totalM, kdim), (kdim, 1))
    bView = makeDeviceView(bDev, (e, kdim, ndim),
                           (kdim * ndim, ndim, 1))
    idsView = makeDeviceView(idsDev, (totalM,), (1,))
    dView = makeDeviceView(dDev, (totalM, ndim), (ndim, 1))
    totalTiles = countGroupedGemmTiles(Elem, totalM, int(Ndim))
    flops = 2.0e0 * float64(totalM) * float64(Kdim) * float64(Ndim)
  engine.launchDeviceFn grouped_gemm_via_expertIds(dView, aView, bView, idsView) do:
    grid = (totalTiles, 1)
    blk = (32, 1)
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    let benchStart = getMonoTime()
    engine.launchDeviceFn grouped_gemm_via_expertIds(dView, aView, bView, idsView) do:
      grid = (totalTiles, 1)
      blk = (32, 1)
    let benchStop = getMonoTime()
    timings[rep] = inNanoseconds(benchStop - benchStart)
  sort(timings)
  let med = timings[1].float64 / 1e9
  writeLine(stdout, &"                                gemm {flops / med / 1e9:>9.1f} GF/s  {med * 1e6:>10.1f} us")

proc mainGroupedGemvViaExpertIds* =
  ## grouped_gemv_via_expertIds bench: kSplits ablations, then the (T, topk, E)
  ## sweep with the matched gemm column.
  when Elem is float16:
    const dtypeName = "fp16"
  else:
    const dtypeName = "fp32"
  writeLine(stdout, "grouped_gemv_via_expertIds vs packed grouped_gemm_via_expertIds, K=N=512, " &
    dtypeName & " (M4 Max), median of 3 launch-inclusive after 1 warmup")
  var engine = newEngine(runMetal)
  for kSplits in [8, 64]:
    runSplitKCell(engine, 1, 6, 64, kSplits)
  for tok in [1, 8, 64, 512]:
    for topk in [1, 6]:
      runDecodeCell(engine, tok, topk, 64)
      let (form, kSplits) = groupedGemvForm(tok, int(Ndim))
      if form == ggfSplitK:
        runSplitKCell(engine, tok, topk, 64, kSplits)
      runMatchedGemmCell(engine, tok, topk, 64)
  runDecodeCell(engine, 512, 6, 512)
  runMatchedGemmCell(engine, 512, 6, 512)

proc runExpandCell(engine: MetalEngine, e: int, m: int, padded: static bool) =
  ## One cell: 1 warmup launch, median of 3 timed launches.
  var (aView, bView, idsView, dView) = stageSlabs(engine, e, m, padded)
  let
    totalM = e * (if padded: (m + 31) div 32 * 32 else: m)
    totalTiles = countGroupedGemmTiles(Elem, totalM, int(Ndim))
    flops = 2.0e0 * float64(e) * float64(m) * float64(Kdim) * float64(Ndim)
  engine.launchDeviceFn grouped_gemm_via_expertIds(dView, aView, bView, idsView) do:
    grid = (totalTiles, 1)
    blk = (32, 1)
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    let benchStart = getMonoTime()
    engine.launchDeviceFn grouped_gemm_via_expertIds(dView, aView, bView, idsView) do:
      grid = (totalTiles, 1)
      blk = (32, 1)
    let benchStop = getMonoTime()
    timings[rep] = inNanoseconds(benchStop - benchStart)
  sort(timings)
  let
    med = timings[1].float64 / 1e9
    tflops = flops / med / 1e12
  when padded:
    const packing = "padded"
  else:
    const packing = "aligned"
  writeLine(stdout, &"E={e:>4} M={m:>5} {packing}  median {med * 1e6:>12.1f} us  effective {tflops:>7.2f} TFLOP/s")

proc mainGroupedGemmViaExpertIds* =
  ## grouped_gemm_via_expertIds bench: the (E, M) sweep, aligned and padded blocks.
  when Elem is float16:
    const dtypeName = "fp16"
  else:
    const dtypeName = "fp32"
  writeLine(stdout, "grouped_gemm_via_expertIds Metal bench, K=N=512, one table per dtype, " &
    dtypeName & " (M4 Max)")
  var engine = newEngine(runMetal)
  for e in [64, 512]:
    for m in [1, 8, 64]:
      runExpandCell(engine, e, m, padded = false)
      runExpandCell(engine, e, m, padded = true)
  runExpandCell(engine, 64, 512, padded = false)
  runExpandCell(engine, 64, 512, padded = true)
  # Prefill saturation cell: m per expert where the tiles stop being
  # boundary-bound and the form reaches the dense baseline
  runExpandCell(engine, 64, 4096, padded = false)
