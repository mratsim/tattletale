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
  MaxBenchE = 512
  MaxBenchTok = 512
  MaxBenchTopk = 8
  MaxBenchKSplits = 64
  MaxBenchTotalM = 512 * 64

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

template stageSweepSlabs(engine: MetalEngine, T: typedesc, aDev, bDev, dDev, xDev, d32Dev, pDev, offsDev, idsDev, weightsDev: untyped) =
  ## Stages the sweep's operand slabs at max size, once per dtype set:
  ## the pattern fill is position-independent and every cell reads a prefix
  ## of the same device buffer.
  ## - the slabs live in refs, DeviceMem is move-only and the per-cell views
  ##   borrow the buffer through the ref, they never own it
  ## - offs, ids and weights are per-cell content, the cells re-upload them
  ##   (a few KB)
  ## - d, d32 and p are workspaces, never read before write
  let
    aLen = MaxBenchTotalM * int(Kdim)
    bLen = MaxBenchE * int(Kdim) * int(Ndim)
    dLen = MaxBenchTotalM * int(Ndim)
    xLen = MaxBenchTok * int(Kdim)
    pLen = MaxBenchTok * MaxBenchKSplits * int(Ndim)
  var
    aPattern = newSeq[T](aLen)
    bPattern = newSeq[T](bLen)
    xPattern = newSeq[T](xLen)
  fillPattern(cast[ptr UncheckedArray[T]](addr aPattern[0]), aLen, 7, 0.25'f32, 0.5'f32)
  fillPattern(cast[ptr UncheckedArray[T]](addr bPattern[0]), bLen, 13, 0.025'f32, 0.15'f32)
  fillPattern(cast[ptr UncheckedArray[T]](addr xPattern[0]), xLen, 7, 0.25'f32, 0.5'f32)
  new(aDev)
  new(bDev)
  new(dDev)
  new(d32Dev)
  new(xDev)
  new(pDev)
  new(offsDev)
  new(idsDev)
  new(weightsDev)
  aDev[] = engine.allocDevice(T, aLen)
  bDev[] = engine.allocDevice(T, bLen)
  dDev[] = engine.allocDevice(T, dLen)
  d32Dev[] = engine.allocDevice(float32, dLen)
  xDev[] = engine.allocDevice(T, xLen)
  pDev[] = engine.allocDevice(float32, pLen)
  offsDev[] = engine.allocDevice(int32, MaxBenchE)
  idsDev[] = engine.allocDevice(int32, MaxBenchTotalM)
  weightsDev[] = engine.allocDevice(float32, MaxBenchTok * MaxBenchTopk)
  engine.copyToDeviceFromHost(aDev[], aPattern)
  engine.copyToDeviceFromHost(bDev[], bPattern)
  engine.copyToDeviceFromHost(xDev[], xPattern)

proc denseMatmulCombo*(mma: static TiledMma, d, a, b: distinct TensorView) =
  matmul(mma, d, a, b)

# ═══════════════════════════════════════════════════════════════════════
#  via_offsets columns
# ═══════════════════════════════════════════════════════════════════════

proc printGemvRow(e, m: int, dtypeName: string, gemvGbs, gemmGf: float64) =
  writeLine(stdout, &"E={e:>4} M={m:>5} {dtypeName}  gemv {gemvGbs:>10.2f} GB/s  gemm {gemmGf:>10.2f} GFLOP/s")

proc printGemmRow(e, m: int, dtypeName: string, groupedGf, denseGf: float64) =
  writeLine(stdout, &"E={e:>4} M={m:>5} {dtypeName}  grouped {groupedGf:>10.2f}  dense {denseGf:>10.2f} GFLOP/s")

proc printDecodeRow(tok, topk, e: int, gbs, us: float64) =
  writeLine(stdout, &"T={tok:>4} topk={topk} E={e:>4}  gemv {gbs:>9.1f} GB/s  {us:>10.1f} us")

proc printSplitkRow(kSplits: int, gbs, us: float64) =
  writeLine(stdout, &"                                gemv-splitk/{kSplits} {gbs:>9.1f} GB/s  {us:>10.1f} us")

proc printMatchedRow(topk: int, gfs, us: float64) =
  writeLine(stdout, &"                                gemm {gfs:>9.1f} GF/s  {us:>10.1f} us")

proc printExpandRow(e, m: int, packing: string, us, tflops: float64) =
  writeLine(stdout, &"E={e:>4} M={m:>5} {packing}  median {us:>12.1f} us  effective {tflops:>7.2f} TFLOP/s")

template runGemvColumn(T: typedesc, engine: MetalEngine, e: int, m, k, n: int, aDev, bDev, offsDev, dDev: untyped): float64 =
  ## GB/s over the traffic model: every line reads its expert's full b matrix
  ## (m·k·n b bytes dominate), plus a lines and dst writes.
  var offsPattern = newSeq[int32](e)
  for g in 0 ..< e:
    offsPattern[g] = int32((g + 1) * m)
  engine.copyToDeviceFromHost(offsDev[], offsPattern)
  let
    eR = int(e)
    mR = int(m)
    kR = int(k)
    nR = int(n)
    aView = makeDeviceView(aDev, (eR * mR, kR), (kR, 1))
    bView = makeDeviceView(bDev, (eR, kR, nR), (kR * nR, nR, 1))
    offsView = makeDeviceView(offsDev, (eR,), (1,))
    dView = makeDeviceView(dDev, (eR * mR, nR), (nR, 1))
    elt = sizeof(T).float64
    bytes = float64(e * m * (k * n + k + n)) * elt
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
  bytes / float64(timings[1])

template runGroupedGemmColumn(T: typedesc, engine: MetalEngine, e: int, m, k, n: int, aDev, bDev, offsDev, dDev: untyped): float64 =
  var offsPattern = newSeq[int32](e)
  for g in 0 ..< e:
    offsPattern[g] = int32((g + 1) * m)
  engine.copyToDeviceFromHost(offsDev[], offsPattern)
  let
    eR = int(e)
    mR = int(m)
    kR = int(k)
    nR = int(n)
    aView = makeDeviceView(aDev, (eR * mR, kR), (kR, 1))
    bView = makeDeviceView(bDev, (eR, kR, nR), (kR * nR, nR, 1))
    offsView = makeDeviceView(offsDev, (eR,), (1,))
    dView = makeDeviceView(dDev, (eR * mR, nR), (nR, 1))
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
  flops / float64(timings[1])

template runDenseColumn(T: typedesc, engine: MetalEngine, e: int, m, k, n: int, aDev, bDev, d32Dev: untyped): float64 =
  ## Same pattern values and tile config as the grouped columns, D fp32 for every dtype
  ## (the matmul's register accumulator is fp32): the columns differ in schedule alone.
  ## The b view reads the b slab's first k·n elements transposed, the same values
  ## the dedicated dense slab held.
  let
    eR = int(e)
    mR = int(m)
    kR = int(k)
    nR = int(n)
    aView = makeDeviceView(aDev, (eR * mR, kR), (kR, 1))
    bView = makeDeviceView(bDev, (nR, kR), (1, nR))
    dView = makeDeviceView(d32Dev, (eR * mR, nR), (nR, 1))
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
  flops / float64(timings[1])

# ═══════════════════════════════════════════════════════════════════════
#  via_offsets cells and mains
# ═══════════════════════════════════════════════════════════════════════

template runGemvCell(engine: MetalEngine, e: int, m: int, T: typedesc, aDev, bDev, offsDev, dDev: untyped) =
  let
    k = Kdim
    n = Ndim
    gemvGbs = runGemvColumn(T, engine, e, m, k, n, aDev, bDev, offsDev, dDev)
    gemmGf = runGroupedGemmColumn(T, engine, e, m, k, n, aDev, bDev, offsDev, dDev)
  when T is float16:
    const dtypeName = "fp16"
  else:
    const dtypeName = "fp32"
  printGemvRow(e, m, dtypeName, gemvGbs, gemmGf)

template runGemmCell(engine: MetalEngine, e: int, m: int, T: typedesc, aDev, bDev, offsDev, dDev, d32Dev: untyped) =
  let
    k = Kdim
    n = Ndim
    groupedGf = runGroupedGemmColumn(T, engine, e, m, k, n, aDev, bDev, offsDev, dDev)
    denseGf = runDenseColumn(T, engine, e, m, k, n, aDev, bDev, d32Dev)
  when T is float16:
    const dtypeName = "fp16"
  else:
    const dtypeName = "fp32"
  printGemmRow(e, m, dtypeName, groupedGf, denseGf)

proc checkDtypeArg(dtype: string) =
  if dtype notin ["fp32", "fp16"]:
    writeLine(stderr, &"error: unknown dtype '{dtype}', expected fp32 or fp16")
    quit(1)

proc mainGroupedGemvViaOffsets* =
  ## grouped_gemv_via_offsets bench:
  ## E and M select one cell, M is the per-group M_g, no arguments run the full sweep over both dtypes.
  let args = commandLineParams()
  var engine = newEngine(runMetal)
  var
    aF32, bF32, dF32, xF32: ref DeviceMem[runMetal, float32]
    d32F32, pF32, wF32: ref DeviceMem[runMetal, float32]
    offsF32, idsF32: ref DeviceMem[runMetal, int32]
    aF16, bF16, dF16, xF16: ref DeviceMem[runMetal, float16]
    d32F16, pF16, wF16: ref DeviceMem[runMetal, float32]
    offsF16, idsF16: ref DeviceMem[runMetal, int32]
  stageSweepSlabs(engine, float32, aF32, bF32, dF32, xF32, d32F32, pF32, offsF32, idsF32, wF32)
  stageSweepSlabs(engine, float16, aF16, bF16, dF16, xF16, d32F16, pF16, offsF16, idsF16, wF16)
  if args.len >= 2:
    let
      e = parseInt(args[0])
      m = parseInt(args[1])
      dtype = if args.len >= 3: args[2] else: "fp32"
    checkDtypeArg(dtype)
    writeLine(stdout, &"grouped_gemv_via_offsets bench, K={Kdim} N={Ndim}, one cell")
    if dtype == "fp16":
      runGemvCell(engine, e, m, float16, aF16, bF16, offsF16, dF16)
    else:
      runGemvCell(engine, e, m, float32, aF32, bF32, offsF32, dF32)
  else:
    writeLine(stdout, &"grouped_gemv_via_offsets bench, full sweep, K={Kdim} N={Ndim}, M4 Max")
    for m in [1, 8]:
      runGemvCell(engine, 64, m, float32, aF32, bF32, offsF32, dF32)
      runGemvCell(engine, 64, m, float16, aF16, bF16, offsF16, dF16)
    runGemvCell(engine, 512, 64, float32, aF32, bF32, offsF32, dF32)
    runGemvCell(engine, 512, 64, float16, aF16, bF16, offsF16, dF16)

proc mainGroupedGemmViaOffsets* =
  ## grouped_gemm_via_offsets bench:
  ## E and M select one cell, M is the per-group M_g, no arguments run the fp32 full sweep.
  let args = commandLineParams()
  var engine = newEngine(runMetal)
  var
    aF32, bF32, dF32, xF32: ref DeviceMem[runMetal, float32]
    d32F32, pF32, wF32: ref DeviceMem[runMetal, float32]
    offsF32, idsF32: ref DeviceMem[runMetal, int32]
    aF16, bF16, dF16, xF16: ref DeviceMem[runMetal, float16]
    d32F16, pF16, wF16: ref DeviceMem[runMetal, float32]
    offsF16, idsF16: ref DeviceMem[runMetal, int32]
  stageSweepSlabs(engine, float32, aF32, bF32, dF32, xF32, d32F32, pF32, offsF32, idsF32, wF32)
  stageSweepSlabs(engine, float16, aF16, bF16, dF16, xF16, d32F16, pF16, offsF16, idsF16, wF16)
  if args.len >= 2:
    let
      e = parseInt(args[0])
      m = parseInt(args[1])
      dtype = if args.len >= 3: args[2] else: "fp32"
    checkDtypeArg(dtype)
    writeLine(stdout, &"grouped_gemm_via_offsets bench, K={Kdim} N={Ndim}, one cell")
    if dtype == "fp16":
      runGemmCell(engine, e, m, float16, aF16, bF16, offsF16, dF16, d32F16)
    else:
      runGemmCell(engine, e, m, float32, aF32, bF32, offsF32, dF32, d32F32)
  else:
    writeLine(stdout, &"grouped_gemm_via_offsets bench, full sweep, K={Kdim} N={Ndim}, M4 Max")
    for m in [1, 8, 64]:
      runGemmCell(engine, 512, m, float32, aF32, bF32, offsF32, dF32, d32F32)

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

template runDecodeCell(engine: MetalEngine, tok, topk, e: int, xDev, bDev, dDev, idsDev, weightsDev: untyped) =
  ## One via_expertIds gemv cell: 1 warmup launch, median of 3 timed launches.
  ## x and b come off the shared slabs, ids and weights are per-cell content.
  var
    idsPattern = newSeq[int32](tok * topk)
    weightsPattern = newSeq[float32](tok * topk)
  for t in 0 ..< tok:
    for i in 0 ..< topk:
      idsPattern[t * topk + i] = int32((t * 7 + i * 13) mod e)
      weightsPattern[t * topk + i] = 1.0'f32 / float32(topk)
  engine.copyToDeviceFromHost(idsDev[], idsPattern)
  engine.copyToDeviceFromHost(weightsDev[], weightsPattern)
  let
    tokR = int(tok)
    topkR = int(topk)
    eR = int(e)
    kdim = int(Kdim)
    ndim = int(Ndim)
    xView = makeDeviceView(xDev, (tokR, kdim), (kdim, 1))
    bView = makeDeviceView(bDev, (eR, kdim, ndim),
                           (kdim * ndim, ndim, 1))
    idsView = makeDeviceView(idsDev, (tokR, topkR), (topkR, 1))
    weightsView = makeDeviceView(weightsDev, (tokR, topkR), (topkR, 1))
    dView = makeDeviceView(dDev, (tokR, ndim), (ndim, 1))
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
  printDecodeRow(tok, topk, e, bytes / med / 1e9, med * 1e6)

template runSplitKCell(engine: MetalEngine, tok, topk, e, kSplits: int, xDev, bDev, dDev, idsDev, weightsDev, pDev: untyped) =
  ## One split-K cell: stage 1 + stage 2 back to back, 1 warmup pair, median of 3 timed pairs.
  ## The timed unit is both launches: the form's cost is the pair.
  ## x and b come off the shared slabs, ids and weights are per-cell content.
  var
    idsPattern = newSeq[int32](tok * topk)
    weightsPattern = newSeq[float32](tok * topk)
  for t in 0 ..< tok:
    for i in 0 ..< topk:
      idsPattern[t * topk + i] = int32((t * 7 + i * 13) mod e)
      weightsPattern[t * topk + i] = 1.0'f32 / float32(topk)
  engine.copyToDeviceFromHost(idsDev[], idsPattern)
  engine.copyToDeviceFromHost(weightsDev[], weightsPattern)
  let
    tokR = int(tok)
    topkR = int(topk)
    eR = int(e)
    kSplitsR = int(kSplits)
    kdim = int(Kdim)
    ndim = int(Ndim)
    xView = makeDeviceView(xDev, (tokR, kdim), (kdim, 1))
    bView = makeDeviceView(bDev, (eR, kdim, ndim),
                           (kdim * ndim, ndim, 1))
    idsView = makeDeviceView(idsDev, (tokR, topkR), (topkR, 1))
    weightsView = makeDeviceView(weightsDev, (tokR, topkR), (topkR, 1))
    dView = makeDeviceView(dDev, (tokR, ndim), (ndim, 1))
    pView = makeDeviceView(pDev, (tokR, kSplitsR, ndim),
                           (kSplitsR * ndim, ndim, 1))
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
  printSplitkRow(kSplits, bytes / med / 1e9, med * 1e6)

template runMatchedGemmCell(engine: MetalEngine, tok, topk, e: int, aDev, bDev, dDev, idsDev: untyped) =
  ## Matched gemm cell: grouped_gemm_via_expertIds over the packed equivalent.
  ##
  ## totalM = T·topk lines, sorted expert-major, every line valid, no -1 padding.
  ## The host-side pack is not charged (best case for the gemm tier).
  ## a and b come off the shared slabs, ids is per-cell content.
  let totalM = tok * topk
  var idsPattern = newSeq[int32](totalM)
  # expert-major lines: line r belongs to expert r mod e, already ascending in r
  for r in 0 ..< totalM:
    idsPattern[r] = int32(r mod e)
  engine.copyToDeviceFromHost(idsDev[], idsPattern)
  let
    eR = int(e)
    kdim = int(Kdim)
    ndim = int(Ndim)
    aView = makeDeviceView(aDev, (totalM, kdim), (kdim, 1))
    bView = makeDeviceView(bDev, (eR, kdim, ndim),
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
  printMatchedRow(topk, flops / med / 1e9, med * 1e6)

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
  var
    aDev, bDev, dDev, xDev: ref DeviceMem[runMetal, Elem]
    d32Dev, pDev, weightsDev: ref DeviceMem[runMetal, float32]
    offsDev, idsDev: ref DeviceMem[runMetal, int32]
  stageSweepSlabs(engine, Elem, aDev, bDev, dDev, xDev, d32Dev, pDev, offsDev, idsDev, weightsDev)
  for kSplits in [8, 64]:
    runSplitKCell(engine, 1, 6, 64, kSplits, xDev, bDev, dDev, idsDev, weightsDev, pDev)
  for tok in [1, 8, 64, 512]:
    for topk in [1, 6]:
      runDecodeCell(engine, tok, topk, 64, xDev, bDev, dDev, idsDev, weightsDev)
      let (form, kSplits) = groupedGemvForm(tok, int(Ndim))
      if form == ggfSplitK:
        runSplitKCell(engine, tok, topk, 64, kSplits, xDev, bDev, dDev, idsDev, weightsDev, pDev)
      runMatchedGemmCell(engine, tok, topk, 64, aDev, bDev, dDev, idsDev)
  runDecodeCell(engine, 512, 6, 512, xDev, bDev, dDev, idsDev, weightsDev)
  runMatchedGemmCell(engine, 512, 6, 512, aDev, bDev, dDev, idsDev)

template runExpandCell(engine: MetalEngine, e: int, m: int, padded: static bool, aDev, bDev, dDev, idsDev: untyped) =
  ## One cell: 1 warmup launch, median of 3 timed launches.
  ## a and b come off the shared slabs, ids is per-cell content.
  let
    perBlock = when padded: (int(m) + 31) div 32 * 32 else: int(m)
    totalM = e * perBlock
  var idsPattern = newSeq[int32](totalM)
  for g in 0 ..< e:
    for local in 0 ..< perBlock:
      idsPattern[g * perBlock + local] =
        int32(if local < m: g else: -1)
  engine.copyToDeviceFromHost(idsDev[], idsPattern)
  let
    kdim = int(Kdim)
    ndim = int(Ndim)
    aView = makeDeviceView(aDev, (totalM, kdim), (kdim, 1))
    bView = makeDeviceView(bDev, (e, kdim, ndim),
                           (kdim * ndim, ndim, 1))
    idsView = makeDeviceView(idsDev, (totalM,), (1,))
    dView = makeDeviceView(dDev, (totalM, ndim), (ndim, 1))
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
  printExpandRow(e, m, packing, med * 1e6, tflops)

proc mainGroupedGemmViaExpertIds* =
  ## grouped_gemm_via_expertIds bench: the (E, M) sweep, aligned and padded blocks.
  when Elem is float16:
    const dtypeName = "fp16"
  else:
    const dtypeName = "fp32"
  writeLine(stdout, "grouped_gemm_via_expertIds Metal bench, K=N=512, one table per dtype, " &
    dtypeName & " (M4 Max)")
  var engine = newEngine(runMetal)
  var
    aDev, bDev, dDev, xDev: ref DeviceMem[runMetal, Elem]
    d32Dev, pDev, weightsDev: ref DeviceMem[runMetal, float32]
    offsDev, idsDev: ref DeviceMem[runMetal, int32]
  stageSweepSlabs(engine, Elem, aDev, bDev, dDev, xDev, d32Dev, pDev, offsDev, idsDev, weightsDev)
  for e in [64, 512]:
    for m in [1, 8, 64]:
      runExpandCell(engine, e, m, padded = false, aDev, bDev, dDev, idsDev)
      runExpandCell(engine, e, m, padded = true, aDev, bDev, dDev, idsDev)
  runExpandCell(engine, 64, 512, padded = false, aDev, bDev, dDev, idsDev)
  runExpandCell(engine, 64, 512, padded = true, aDev, bDev, dDev, idsDev)
