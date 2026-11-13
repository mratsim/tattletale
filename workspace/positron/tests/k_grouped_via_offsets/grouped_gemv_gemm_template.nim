## grouped_gemv_via_offsets and grouped_gemm_via_offsets suite machinery shared by the drivers, per backend:
## - fixtures generated host-side with a seeded rng
## - the reference is a float64 host GEMM over the same rounded inputs
## - mainGroupedGEMV and mainGroupedGEMM run the per-tier case suites
##
## Element bands, per output element: |got - want| <= Rtol*|want| + Abstol
## - fp32 (rtol 1e-4, abstol 1e-4): the whole chain accumulates in fp32,
##   the reference accumulates in fp64 over the same rounded inputs
## - fp16 (rtol 1e-2, abstol 1e-2): the operands and the stored output are
##   fp16, one fp16 ulp (2^-11) of relative error per rounded element

import std/[math, random]
import workspace/math
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/tests/ceramic_testutils
import workspace/positron/src/kernels/k_grouped_gemv_gemm {.all.}

export int_tuples, layout_algebra, tensors, tile_algebra, k_grouped_gemv_gemm

const
  Fp16Rtol* = 1e-2'f64
  Fp16Abstol* = 1e-2'f64
  Fp32Rtol* = 1e-4'f64
  Fp32Abstol* = 1e-4'f64

const FuzzSeed* = 20261111

# ═══════════════════════════════════════════════════════════════════════
#  float64 host reference
# ═══════════════════════════════════════════════════════════════════════

proc gemmRefF64[T, LA, LB](a: HostTensor[T, LA], b: HostTensor[T, LB]): auto =
  let
    m = a.layout.shape[0]
    k = a.layout.shape[1]
    n = b.layout.shape[1]
  result = makeHostTensor(float64, (m, n), (n, 1))
  for i in 0 ..< m:
    for j in 0 ..< n:
      var acc = 0.0'f64
      for e in 0 ..< k:
        acc += to(a[i, e], float64) * to(b[e, j], float64)
      result[i, j] = acc

proc copyLines[T, LD, LS](dst: var HostTensor[T, LD], src: HostTensor[T, LS], lineBase: int) =
  ## src and dst are line-major.
  let cols = src.layout.shape[1]
  for i in 0 ..< src.layout.shape[0]:
    for j in 0 ..< cols:
      dst[lineBase + int(i), int(j)] = src[i, j]

proc sliceLines[T, L](src: HostTensor[T, L], lineBase, height, cols: int): auto =
  ## Lines [lineBase, lineBase + height) of a line-major host tensor, compact layout.
  result = makeHostTensor(T, (height, cols), (cols, 1))
  for i in 0 ..< height:
    for j in 0 ..< cols:
      result[i, j] = src[lineBase + i, j]

proc sliceGroupB[T, L](stack: HostTensor[T, L], g, k, n: int): auto =
  result = makeHostTensor(T, (k, n), (n, 1))
  for i in 0 ..< k:
    for j in 0 ..< n:
      result[i, j] = stack[g, i, j]

# ═══════════════════════════════════════════════════════════════════════
#  Fixtures, launch and judgment
# ═══════════════════════════════════════════════════════════════════════

template groupedBand(T: typedesc): tuple[rtol, abstol: float64] =
  (when T is float16: (rtol: Fp16Rtol, abstol: Fp16Abstol)
   else: (rtol: Fp32Rtol, abstol: Fp32Abstol))

proc fillSentinel[T, L](ht: var HostTensor[T, L]) =
  ## Pre-fills every element with a value outside the fixtures' range.
  ## Any line the kernel fails to write keeps the sentinel and fails the band.
  for i in 0 ..< ht.len():
    ht.data[i] = to(-999.0'f32, T)

proc groupedFixtures[T](_: typedesc[T], ms: openArray[int], k, n: int, rng: var Rand): auto =
  ## Grouped-contract fixtures over per-group line counts `ms`:
  ## - a: (totalM, k) random, b: (g, k, n) random and distinct per group
  ## - offs: cumulative end offsets, an empty group repeats the previous offset
  ## - d: sentinel-filled, every output element the kernel fails to write fails the band
  let
    totalM = int32(sum(ms))
    g = ms.len
  var
    aHost = rng.randomHostTensor(T, (totalM, k), (k, 1))
    bHost = rng.randomHostTensor(T, (g, k, n),
                                 (k * n, n, 1))
    offsHost = makeHostTensor(int32, (g,), (1,))
    dHost = makeHostTensor(T, (totalM, n), (n, 1))
  var cum = 0
  for i in 0 ..< ms.len:
    cum += ms[i]
    offsHost[i] = int32(cum)
  fillSentinel(dHost)
  (aHost, bHost, offsHost, dHost)

proc judgeGrouped[T, LD](got: HostTensor[T, LD], aHost, bHost: distinct HostTensor, ms: openArray[int], msg: string) =
  ## float64 reference sliced per group over the concatenated lines,
  ## judged against the kernel output with the element band.
  when T is float16:
    const dtypeLabel = "fp16"
  else:
    const dtypeLabel = "fp32"
  let
    totalM = got.layout.shape[0].toInt()
    n = got.layout.shape[1].toInt()
    k = aHost.layout.shape[1].toInt()
  var refHost = makeHostTensor(float64, (totalM, n), (n, 1))
  var gStart = 0
  for g in 0 ..< ms.len:
    let mg = ms[g]
    if mg > 0:
      let refG = gemmRefF64(sliceLines(aHost, gStart, mg, k),
                            sliceGroupB(bHost, g, k, n))
      copyLines(refHost, refG, gStart)
    gStart += mg
  const band = groupedBand(T)
  assertAllClose(got.to(float64).view(), refHost.view(),
    rtol = band.rtol, abstol = band.abstol,
    msg = msg & " (" & dtypeLabel & ", G=" & $ms.len & ", totalM=" & $totalM &
      ", K=" & $k & ", N=" & $n & ")")

# ═══════════════════════════════════════════════════════════════════════
#  grouped_gemv_via_offsets cases
# ═══════════════════════════════════════════════════════════════════════

proc launchGroupedGemv[T, LA, LB, LG, LD](
    engine: MetalEngine,
    dHost: var HostTensor[T, LD], aHost: HostTensor[T, LA],
    bHost: HostTensor[T, LB], offsHost: HostTensor[int32, LG], gridDiv = 1) =
  ## gridDiv scales the block-axis grid, for the undersized-grid coverage case.
  if dHost.layout.shape[0].toInt() * dHost.layout.shape[1].toInt() == 0:
    # totalM = 0: every group is empty, d has no element, nothing to bridge or launch
    return
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    offsDev = engine.bridge(offsHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn grouped_gemv_via_offsets(dDev, aDev, bDev, offsDev) do:
    grid = (max(1, dHost.layout.shape[1].toInt().ceil_div(32 * GroupedGemvOutputs) div gridDiv),
            dHost.layout.shape[0].toInt())
    blk = (32, 1)
  engine.copyToHostFromDevice(dHost.data.toOpenArray(0, dHost.len() - 1), dDev.data)

proc groupedGemvCase[T](engine: MetalEngine, _: typedesc[T], ms: openArray[int], k, n: int, rng: var Rand, msg: string, gridDiv = 1) =
  var (aHost, bHost, offsHost, dHost) = groupedFixtures(T, ms, k, n, rng)
  launchGroupedGemv(engine, dHost, aHost, bHost, offsHost, gridDiv)
  judgeGrouped(dHost, aHost, bHost, ms, msg)

proc gemvBijectionCase[T](engine: MetalEngine, _: typedesc[T]) =
  ## Bijection check: grouped over 3 groups (M = 17, 0, 33, K = 24, N = 16)
  ## with the same B for every group must equal one dense GEMM of the same total.
  var rng = initRand(2026)
  let
    k = 24
    n = 16
  var
    aHost = rng.randomHostTensor(T, (50, k), (k, 1))
    b0 = rng.randomHostTensor(T, (k, n), (n, 1))
    bHost = makeHostTensor(T, (3, k, n), (k * n, n, 1))
    offsHost = makeHostTensor(int32, (3,), (1,))
    dHost = makeHostTensor(T, (50, n), (n, 1))
  for gx in 0 ..< 3:
    for i in 0 ..< k * n:
      bHost.data[gx * int(k * n) + i] = b0.data[i]
  offsHost[0] = 17
  offsHost[1] = 17
  offsHost[2] = 50
  fillSentinel(dHost)
  launchGroupedGemv(engine, dHost, aHost, bHost, offsHost)

  let dense = gemmRefF64(aHost, b0)
  const band = groupedBand(T)
  const BijectionSpans = [(0, 17), (17, 33)]
  for gx in 0 ..< BijectionSpans.len:
    let (lineBase, height) = BijectionSpans[gx]
    assertAllClose(dHost.to(float64).sliceLines(lineBase, height, int(n)).view(),
                   dense.sliceLines(lineBase, height, int(n)).view(),
      rtol = band.rtol, abstol = band.abstol,
      msg = "bijection lines [" & $lineBase & ", " & $(lineBase + height) & ")")

proc gemvEdgeCase[T](engine: MetalEngine, _: typedesc[T]) =
  ## Edge cells of the group dimension.
  var rng = initRand(17)
  groupedGemvCase(engine, T, [1], 24, 16, rng,
    "single group, single line")
  groupedGemvCase(engine, T, [33], 40, 48, rng,
    "single group, 33 lines")
  groupedGemvCase(engine, T, [5, 0, 0], 24, 16, rng,
    "trailing empty groups")
  groupedGemvCase(engine, T, [0, 0], 24, 16, rng,
    "all groups empty, launch skipped")

proc gemvWideCase(engine: MetalEngine) =
  var rng = initRand(512)
  var ms = newSeq[int](512)
  for g in 0 ..< 512:
    ms[g] = [0, 1, 3, 33][g mod 4]
  groupedGemvCase(engine, float32, ms, 64, 64, rng,
    "wide, 512 groups cycling M = {0, 1, 3, 33}")

proc gemvRaggedNCase[T](engine: MetalEngine, _: typedesc[T]) =
  ## Ragged N edges over a non-power-of-2 K = 40 depth, no tiling on the gemv path.
  var rng = initRand(48)
  groupedGemvCase(engine, T, [3, 0, 9], 40, 17, rng,
    "ragged N, one n-block, idle lanes")
  groupedGemvCase(engine, T, [3, 0, 9], 40, 48, rng,
    "ragged N, one n-block, lanes past N idle")
  groupedGemvCase(engine, T, [3, 0, 9], 40, 129, rng,
    "ragged N, two n-blocks, one output in the second block")
  groupedGemvCase(engine, T, [3, 0, 9], 40, 200, rng,
    "ragged N, two n-blocks, short last block")

proc gemvFuzzCase(engine: MetalEngine) =
  ## Seeded fuzz, fp32, float64 reference, seed printed. The forced ms = [0, 0] cases
  ## cover the launch-skip path: the seed's random cases never produce an all-empty shape.
  var rng = initRand(FuzzSeed)
  echo "fuzz seed: ", FuzzSeed
  for i in 0 ..< 8:
    var ms: seq[int]
    if i mod 4 == 0:
      ms = @[0, 0]
    else:
      let g = rng.rand(1 .. 16)
      ms = newSeq[int](g)
      for j in 0 ..< g:
        ms[j] = rng.rand(0 .. 40)
      if g >= 2:
        ms[0] = 0
        ms[1] = 1
    let
      k = rng.rand(8 .. 70)
      n = rng.rand(8 .. 70)
    groupedGemvCase(engine, float32, ms, k, n, rng, "fuzz case " & $i)

proc gemvUndersizedGridCase(engine: MetalEngine) =
  ## grouped_gemv_via_offsets on HALF the scheduling contract's block-axis grid:
  ## ceil_div(n-blocks, 2) threadgroups.
  var rng = initRand(4242)
  groupedGemvCase(engine, float32, [3, 9], 40, 129, rng,
    "half grid, two n-blocks", gridDiv = 2)
  groupedGemvCase(engine, float16, [3, 9], 40, 129, rng,
    "half grid, two n-blocks", gridDiv = 2)

# ═══════════════════════════════════════════════════════════════════════
#  grouped_gemm_via_offsets cases
# ═══════════════════════════════════════════════════════════════════════

proc launchGroupedGemm[T, LA, LB, LG, LD](
    engine: MetalEngine,
    dHost: var HostTensor[T, LD], aHost: HostTensor[T, LA],
    bHost: HostTensor[T, LB], offsHost: HostTensor[int32, LG]) =
  ## At totalM = 0 the launch width is 0, the kernel's caller-skip contract.
  let gridWidth = countGroupedGemmTiles(T, dHost.layout.shape[0].toInt(),
                                      dHost.layout.shape[1].toInt())
  if gridWidth > 0:
    let
      aDev = engine.bridge(aHost)
      bDev = engine.bridge(bHost)
      offsDev = engine.bridge(offsHost)
      dDev = engine.bridge(dHost)
    engine.launchDeviceFn grouped_gemm_via_offsets(dDev, aDev, bDev, offsDev) do:
      grid = (gridWidth, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(dHost.data.toOpenArray(0, dHost.len() - 1), dDev.data)

proc groupedGemmCase[T](engine: MetalEngine, _: typedesc[T], ms: openArray[int], k, n: int, rng: var Rand, msg: string) =
  var (aHost, bHost, offsHost, dHost) = groupedFixtures(T, ms, k, n, rng)
  launchGroupedGemm(engine, dHost, aHost, bHost, offsHost)
  judgeGrouped(dHost, aHost, bHost, ms, msg)

proc gemmBijectionCase[T](engine: MetalEngine, _: typedesc[T]) =
  ## Bijection check: grouped over 3 groups (M = 17, 0, 33, K = 24, N = 16)
  ## with the same B for every group must equal one dense GEMM of the same total.
  var rng = initRand(2026)
  let
    k = 24
    n = 16
  var
    aHost = rng.randomHostTensor(T, (50, k), (k, 1))
    b0 = rng.randomHostTensor(T, (k, n), (n, 1))
    bHost = makeHostTensor(T, (3, k, n), (k * n, n, 1))
    offsHost = makeHostTensor(int32, (3,), (1,))
    dHost = makeHostTensor(T, (50, n), (n, 1))
  for gx in 0 ..< 3:
    for i in 0 ..< k * n:
      bHost.data[gx * int(k * n) + i] = b0.data[i]
  offsHost[0] = 17
  offsHost[1] = 17
  offsHost[2] = 50
  fillSentinel(dHost)
  launchGroupedGemm(engine, dHost, aHost, bHost, offsHost)

  let dense = gemmRefF64(aHost, b0)
  const band = groupedBand(T)
  const BijectionSpans = [(0, 17), (17, 33)]
  for gx in 0 ..< BijectionSpans.len:
    let (lineBase, height) = BijectionSpans[gx]
    assertAllClose(dHost.to(float64).sliceLines(lineBase, height, int(n)).view(),
                   dense.sliceLines(lineBase, height, int(n)).view(),
      rtol = band.rtol, abstol = band.abstol,
      msg = "bijection lines [" & $lineBase & ", " & $(lineBase + height) & ")")

proc gemmBoundaryCase[T](engine: MetalEngine, _: typedesc[T]) =
  ## Boundary crossing: G = 8, M per group (1, 3, 0, 33, 1, 0, 3, 33),
  ## K = 40 crosses the KT = 16 depth tile, N = 48 crosses NT = 32,
  ## distinct B per group.
  var rng = initRand(74)
  groupedGemmCase(engine, T, [1, 3, 0, 33, 1, 0, 3, 33], 40, 48, rng,
    "boundary crossing, tiles straddle group boundaries")

proc gemmEdgeCase(engine: MetalEngine) =
  ## Edge cells of the group dimension, fp32 (the schedule is dtype-shared).
  var rng = initRand(17)
  groupedGemmCase(engine, float32, [1], 24, 16, rng,
    "single group, single line")
  groupedGemmCase(engine, float32, [33], 40, 48, rng,
    "single group, 33 lines")
  groupedGemmCase(engine, float32, [5, 0, 0], 24, 16, rng,
    "trailing empty groups")
  groupedGemmCase(engine, float32, [0, 0], 24, 16, rng,
    "all groups empty, launch skipped")

proc gemmWideCase(engine: MetalEngine) =
  var rng = initRand(512)
  var ms = newSeq[int](512)
  for g in 0 ..< 512:
    ms[g] = [0, 1, 3, 33][g mod 4]
  groupedGemmCase(engine, float32, ms, 64, 64, rng,
    "wide, 512 groups cycling M = {0, 1, 3, 33}")

proc gemmFuzzCase(engine: MetalEngine) =
  ## Seeded fuzz, fp32, float64 reference, seed printed.
  var rng = initRand(FuzzSeed)
  echo "fuzz seed: ", FuzzSeed
  for i in 0 ..< 8:
    let g = rng.rand(1 .. 16)
    var ms = newSeq[int](g)
    for j in 0 ..< g:
      ms[j] = rng.rand(0 .. 40)
    if g >= 2:
      ms[0] = 0
      ms[1] = 1
    let
      k = rng.rand(8 .. 70)
      n = rng.rand(8 .. 70)
    groupedGemmCase(engine, float32, ms, k, n, rng, "fuzz case " & $i)

# ═══════════════════════════════════════════════════════════════════════
#  Per-tier suites
# ═══════════════════════════════════════════════════════════════════════

proc mainGroupedGEMV*(engine: MetalEngine) =
  ## grouped_gemv_via_offsets suite: tier rule, bijection, edge cells, wide, ragged N,
  ## undersized-grid coverage and fuzz, fp32 and fp16 operands.
  gemvBijectionCase(engine, float32)
  gemvBijectionCase(engine, float16)
  gemvEdgeCase(engine, float32)
  gemvEdgeCase(engine, float16)
  gemvWideCase(engine)
  gemvUndersizedGridCase(engine)
  gemvRaggedNCase(engine, float32)
  gemvRaggedNCase(engine, float16)
  gemvFuzzCase(engine)
  echo "[OK] grouped_gemv_via_offsets: tier rule, bijection (fp32, fp16), edge cells (fp32, fp16), wide, ragged N, undersized grid, fuzz"

proc mainGroupedGEMM*(engine: MetalEngine) =
  ## grouped_gemm_via_offsets suite: bijection, boundary crossing, edge cells, wide and fuzz.
  # Metal simdgroup atoms: fp16 and fp32 operands, both accumulate in fp32
  gemmBijectionCase(engine, float32)
  gemmBijectionCase(engine, float16)
  gemmBoundaryCase(engine, float32)
  gemmBoundaryCase(engine, float16)
  gemmWideCase(engine)
  gemmEdgeCase(engine)
  gemmFuzzCase(engine)
  echo "[OK] grouped_gemm_via_offsets: bijection (fp32, fp16), boundary crossing (fp32, fp16), wide, edge cells, fuzz"
