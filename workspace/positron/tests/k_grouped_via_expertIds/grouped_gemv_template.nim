## grouped_gemv suite machinery shared by the drivers, per backend:
## - fixtures recorded by gen_grouped_gemv_fixtures.py, replayed and judged per element
## - analytic cases generated host-side with a seeded rng
## - the reference is a float64 host routine over the same rounded inputs,
##   ascending-i slot sums and scale-after-dot like the kernel's combine contract
##
## Element bands, per output element: |got - want| <= Rtol*|want| + Abstol
## - fp32 (rtol 1e-4, abstol 1e-4): the whole chain accumulates in fp32
##   and the reference accumulates in fp64 over the same rounded inputs
## - fp16 (rtol 1e-2, abstol 1e-2): the operands and the stored output are fp16,
##   one fp16 ulp (2^-11) of relative error per rounded element

import std/[math, os, random]
import workspace/math
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/positron/src/kernels/k_grouped_gemv_gemm {.all.}
import ../positron_testutils
import grouped_gemv_fixtures

export int_tuples, layout_algebra, tensors, tile_algebra, k_grouped_gemv_gemm
export grouped_gemv_fixtures

const
  Fp16Rtol* = 1e-2'f64
  Fp16Abstol* = 1e-2'f64
  Fp32Rtol* = 1e-4'f64
  Fp32Abstol* = 1e-4'f64

const FuzzSeed* = 20261113

const GroupedGemvSentinel = -999.0'f32
  ## Pre-fill value outside the fixtures' range: any output element the kernel fails to write keeps the sentinel and fails the judgment

template groupedGemvBand(T: typedesc): tuple[rtol, abstol: float64] =
  (when T is float16: (rtol: Fp16Rtol, abstol: Fp16Abstol)
   else: (rtol: Fp32Rtol, abstol: Fp32Abstol))

# ═══════════════════════════════════════════════════════════════════════
#  float64 host reference
# ═══════════════════════════════════════════════════════════════════════

proc fillSentinel[T, L](ht: var HostTensor[T, L]) =
  for i in 0 ..< ht.len():
    ht.data[i] = to(GroupedGemvSentinel, T)

proc groupedGemvRefF64[T, LX, LB](x: HostTensor[T, LX], b: HostTensor[T, LB], ids: openArray[int], weights: openArray[float32]): auto =
  ## float64 reference of the folded combine over the same rounded inputs.
  ##
  ## - d[t] = sum over ascending i of weights[t, i] · (x[t] · b(ids[t, i]))
  ## - a slot with id < 0 contributes 0
  let
    t = x.layout.shape[0].toInt()
    k = x.layout.shape[1].toInt()
    n = b.layout.shape[2].toInt()
    topk = ids.len div t
  result = makeHostTensor(float64, (t, n), (n, 1))
  # makeHostTensor is uninitialized and the slot sums accumulate, so zero first
  for i in 0 ..< result.len():
    result.data[i] = 0.0'f64
  for ti in 0 ..< t:
    for i in 0 ..< topk:
      let g = ids[ti * topk + i]
      if g >= 0:
        let w = weights[ti * topk + i].float64
        for j in 0 ..< n:
          var acc = 0.0'f64
          for e in 0 ..< k:
            acc += to(x[ti, e], float64) * to(b[g, e, j], float64)
          result[ti, j] += w * acc

# ═══════════════════════════════════════════════════════════════════════
#  Launch and judgment
# ═══════════════════════════════════════════════════════════════════════

proc launchGroupedGemv[T, LX, LB, LG, LW, LD](
    engine: MetalEngine, dHost: var HostTensor[T, LD], xHost: HostTensor[T, LX],
    bHost: HostTensor[T, LB], idsHost: HostTensor[int32, LG],
    weightsHost: HostTensor[float32, LW], topk: int, gridDiv = 1) =
  ## Bridges the fixtures and launches `grouped_gemv` over the full (n-block, token) grid,
  ## copying d back into `dHost`.
  ##
  ## No launch-skip guard: the contract guarantees T >= 1 and N >= 1, so the grid count is
  ## always positive and every view has at least one element.
  let
    tok = dHost.layout.shape[0].toInt()
    n = dHost.layout.shape[1].toInt()
  let
    xDev = engine.bridge(xHost)
    bDev = engine.bridge(bHost)
    idsDev = engine.bridge(idsHost)
    weightsDev = engine.bridge(weightsHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn grouped_gemv_via_expertIds(dDev, xDev, bDev, idsDev, weightsDev, topk) do:
    grid = (max(1, n.ceil_div(32 * GroupedGemvOutputs) div gridDiv), tok)
    blk = (32, 1)
  engine.copyToHostFromDevice(dHost.data.toOpenArray(0, dHost.len() - 1), dDev.data)

proc launchGroupedGemvSplitK[T, LX, LB, LG, LW, LD](
    engine: MetalEngine, dHost: var HostTensor[T, LD], xHost: HostTensor[T, LX],
    bHost: HostTensor[T, LB], idsHost: HostTensor[int32, LG],
    weightsHost: HostTensor[float32, LW], topk, kSplits: int, gridDiv = 1,
    reduceGridDiv = 1) =
  ## Bridges the fixtures and launches the split-K form, stage 1 then stage 2,
  ## copying d back into `dHost`.
  ##
  ## dPartials is a fresh (T, kSplits, N) fp32 workspace per call, sentinel-filled,
  ## so stage 1 must write every in-range element before stage 2 reads it.
  ## kSplits must be <= K. gridDiv scales stage 1's grid on the block axis,
  ## reduceGridDiv scales the reduce stage's grid, both for the undersized-grid case.
  let
    tok = dHost.layout.shape[0].toInt()
    n = dHost.layout.shape[1].toInt()
    partialsHost = makeHostTensor(float32, (tok, kSplits, n),
                                  (kSplits * n, n, 1))
    xDev = engine.bridge(xHost)
    bDev = engine.bridge(bHost)
    idsDev = engine.bridge(idsHost)
    weightsDev = engine.bridge(weightsHost)
    dDev = engine.bridge(dHost)
    pDev = engine.bridge(partialsHost)
  for i in 0 ..< partialsHost.len():
    partialsHost.data[i] = GroupedGemvSentinel
  engine.launchDeviceFn grouped_gemv_splitk(pDev, xDev, bDev, idsDev, weightsDev, topk, kSplits) do:
    grid = (max(1, (n.ceil_div(32 * GroupedGemvOutputs) * kSplits) div gridDiv), tok)
    blk = (32, 1)
  engine.launchDeviceFn grouped_gemv_reduce(dDev, pDev, kSplits) do:
    grid = (max(1, n.ceil_div(32 * GroupedGemvOutputs) div reduceGridDiv), tok)
    blk = (32, 1)
  engine.copyToHostFromDevice(dHost.data.toOpenArray(0, dHost.len() - 1), dDev.data)

proc captureHostBits[T, L](ht: HostTensor[T, L]): seq[uint32] =
  result = newSeq[uint32](ht.len())
  for i in 0 ..< ht.len():
    when T is float16:
      result[i] = uint32(cast[uint16](ht.data[i]))
    else:
      result[i] = cast[uint32](ht.data[i])

proc judgeGroupedGemv[T, LD, LX, LB](
    got: HostTensor[T, LD], xHost: HostTensor[T, LX], bHost: HostTensor[T, LB],
    ids: openArray[int], weights: openArray[float32], msg: string) =
  ## float64 reference over the per-token ids and weights, judged against the kernel
  ## output with the element band.
  ##
  ## Beyond the band: a token whose ids are all -1 must be exactly 0.0 in every element,
  ## the kernel's zero sums are stored unrounded.
  let refHost = groupedGemvRefF64(xHost, bHost, ids, weights)
  const band = groupedGemvBand(T)
  assertAllClose(got.to(float64).view(), refHost.view(),
    rtol = band.rtol, abstol = band.abstol, msg = msg)
  let topk = ids.len div got.layout.shape[0].toInt()
  for ti in 0 ..< got.layout.shape[0].toInt():
    var allInvalid = true
    for i in 0 ..< topk:
      if ids[ti * topk + i] >= 0:
        allInvalid = false
    if allInvalid:
      for j in 0 ..< got.layout.shape[1].toInt():
        doAssert to(got[ti, j], float64) == 0.0'f64,
          msg & ": token " & $ti & " has all slots invalid, d[" & $ti & ", " & $j &
          "] = " & $to(got[ti, j], float64) & " but the zero sums must store exactly 0"

proc buildGroupedGemvInputs[T](_: typedesc[T], rng: var Rand, tok, topk, e, k, n: int, forceInvalidToken: bool): auto =
  ## Builds one grouped_gemv case's inputs from `rng`:
  ## x (tok, k), b (e, k, n) distinct per expert, ids and weights (tok, topk).
  ##
  ## Returns:
  ## - xHost, bHost, idsHost, weightsHost: the device-shaped host tensors
  ## - idsSeq, weightsSeq: the flat ids and weights the float64 reference reads
  ##
  ## Ids per slot:
  ## - about one id in ten is -1, expert ids lie in [0, e)
  ## - forceInvalidToken makes token 0's slots all -1, d[0, :] must come out exactly 0
  var
    xHost = rng.randomHostTensor(T, (tok, k), (k, 1))
    bHost = rng.randomHostTensor(T, (e, k, n),
                                 (k * n, n, 1))
    idsHost = makeHostTensor(int32, (tok, topk), (topk, 1))
    weightsHost = makeHostTensor(float32, (tok, topk), (topk, 1))
  var
    idsSeq: seq[int]
    weightsSeq: seq[float32]
  for ti in 0 ..< tok:
    for i in 0 ..< topk:
      var gid = -1
      if not (forceInvalidToken and ti == 0):
        if rng.rand(1.0) >= 0.1:
          gid = rng.rand(0 .. e - 1)
      idsHost[ti, i] = int32(gid)
      weightsHost[ti, i] = to(rng.rand(0.1 .. 1.0), float32)
      idsSeq.add(gid)
      weightsSeq.add(weightsHost[ti, i])
  (xHost, bHost, idsHost, weightsHost, idsSeq, weightsSeq)

proc groupedGemvCase*[T](engine: MetalEngine, _: typedesc[T], tok, topk, e, k, n: int,
    rng: var Rand, forceInvalidToken: bool, msg: string, kSplits = 0) =
  ## One grouped_gemv case over `tok` tokens and `topk` slots, distinct B per expert.
  ##
  ## Invariants:
  ## - ids drawn per slot, about one in ten forced to -1, expert ids in [0, e)
  ## - forceInvalidToken makes token 0's slots all -1 (d[0, :] must come out exactly 0)
  ## - fixtures generated from `rng`, launched through the public API
  ## - every d element judged against the float64 reference
  ##
  ## kSplits > 0 additionally runs the split-K form over the same inputs.
  ## The float64 judgment and the relaunch bit-identity check apply unchanged
  when T is float16:
    const dtypeLabel = "fp16"
  else:
    const dtypeLabel = "fp32"
  let (xHost, bHost, idsHost, weightsHost, idsSeq, weightsSeq) =
    buildGroupedGemvInputs(T, rng, tok, topk, e, k, n, forceInvalidToken)
  var dHost = makeHostTensor(T, (tok, n), (n, 1))
  fillSentinel(dHost)
  launchGroupedGemv(engine, dHost, xHost, bHost, idsHost, weightsHost, topk)
  judgeGroupedGemv(dHost, xHost, bHost, idsSeq, weightsSeq,
    msg & " (" & dtypeLabel & ", T=" & $tok & ", topk=" & $topk & ", E=" & $e &
    ", K=" & $k & ", N=" & $n & ")")
  if kSplits > 0:
    fillSentinel(dHost)
    launchGroupedGemvSplitK(engine, dHost, xHost, bHost, idsHost, weightsHost, topk, kSplits)
    let firstBits = captureHostBits(dHost)
    fillSentinel(dHost)
    launchGroupedGemvSplitK(engine, dHost, xHost, bHost, idsHost, weightsHost, topk, kSplits)
    let secondBits = captureHostBits(dHost)
    for i in 0 ..< dHost.len():
      doAssert secondBits[i] == firstBits[i],
        msg & ": split-K d differs run to run at element " & $i
    judgeGroupedGemv(dHost, xHost, bHost, idsSeq, weightsSeq,
      msg & " split-K (kSplits=" & $kSplits & ", " & dtypeLabel & ", T=" & $tok &
      ", topk=" & $topk & ", E=" & $e & ", K=" & $k & ", N=" & $n & ")")

# ═══════════════════════════════════════════════════════════════════════
#  Fixture replay
# ═══════════════════════════════════════════════════════════════════════

proc groupedGemvReplayCase*(engine: MetalEngine, Elem: typedesc, name: string, kSplits = 0) =
  ## Replays one fixture, with relaunch:
  ##
  ## - one launch over the shape-derived grid, per-element band judgment
  ##   against the recorded f64 reference
  ## - a relaunch judged bit-identical on the raw output bits
  ## - kSplits > 0 additionally replays the split-K form over the same fixture
  ## - the split-K replay checks relaunch bit-identity, a band judgment against the recorded reference, and the float64 reference judgment
  ##
  ## Preconditions:
  ## - the destination staging is sentinel-filled
  ## - the recorded reference has the kernel's combine contract in float64
  const wantDtype = when Elem is float16: "float16" else: "float32"
  let dir = GroupedGemvFixturesDir / name
  let meta = parseGroupedGemvMetadata(dir)
  doAssert meta.dtype == wantDtype, meta.name & ": dtype " & meta.dtype

  let
    xBits = loadFixtureTensor(dir, "x.bin", (meta.t, meta.k),
      when Elem is float16: uint16 else: uint32)
    bBits = loadFixtureTensor(dir, "b.bin", (meta.e, meta.k, meta.n),
      when Elem is float16: uint16 else: uint32)
    expected = loadFixtureTensor(dir, "d.bin", (meta.t, meta.n), float32)
    idsHost = loadIds2DFixtureTensor(dir, "ids.bin", meta.t, meta.topk)
    weightsHost = loadWeights2DFixtureTensor(dir, "weights.bin", meta.t, meta.topk)
  doAssert xorHash(xBits) == meta.xorhashChecksum("x.bin"), meta.name & ": x.bin checksum"
  doAssert xorHash(bBits) == meta.xorhashChecksum("b.bin"), meta.name & ": b.bin checksum"
  doAssert xorHash(expected) == meta.xorhashChecksum("d.bin"), meta.name & ": d.bin checksum"
  doAssert xorHash(idsHost) == meta.xorhashChecksum("ids.bin"), meta.name & ": ids.bin checksum"
  doAssert xorHash(weightsHost) == meta.xorhashChecksum("weights.bin"),
    meta.name & ": weights.bin checksum"

  var
    xStage = makeHostTensor(Elem, (meta.t, meta.k), LayoutRight)
    bStage = makeHostTensor(Elem, (meta.e, meta.k, meta.n), LayoutRight)
    dStage = makeHostTensor(Elem, (meta.t, meta.n), LayoutRight)
  fillSentinel(dStage)
  for i in 0 ..< xStage.len():
    xStage.data[i] = cast[Elem](xBits.data[i])
  for i in 0 ..< bStage.len():
    bStage.data[i] = cast[Elem](bBits.data[i])

  var ids: seq[int]
  var weights: seq[float32]
  for i in 0 ..< meta.t * meta.topk:
    ids.add(int(idsHost.data[i]))
    weights.add(weightsHost.data[i])

  proc captureBits(): seq[uint32] =
    result = newSeq[uint32](dStage.len())
    for i in 0 ..< dStage.len():
      when Elem is float16:
        result[i] = uint32(cast[uint16](dStage.data[i]))
      else:
        result[i] = cast[uint32](dStage.data[i])

  launchGroupedGemv(engine, dStage, xStage, bStage, idsHost, weightsHost, meta.topk)
  let firstBits = captureBits()

  fillSentinel(dStage)
  launchGroupedGemv(engine, dStage, xStage, bStage, idsHost, weightsHost, meta.topk)
  let secondBits = captureBits()
  for i in 0 ..< dStage.len():
    doAssert secondBits[i] == firstBits[i],
      name & ": d differs run to run at element " & $i

  const band = groupedGemvBand(Elem)
  assertAllClose(dStage.to(float64).view(), expected.to(float64).view(),
    rtol = band.rtol, abstol = band.abstol, msg = name)
  judgeGroupedGemv(dStage, xStage, bStage, ids, weights, name & " exact-zero rows")
  if kSplits > 0:
    fillSentinel(dStage)
    launchGroupedGemvSplitK(engine, dStage, xStage, bStage, idsHost, weightsHost, meta.topk, kSplits)
    let splitFirstBits = captureHostBits(dStage)
    fillSentinel(dStage)
    launchGroupedGemvSplitK(engine, dStage, xStage, bStage, idsHost, weightsHost, meta.topk, kSplits)
    let splitSecondBits = captureHostBits(dStage)
    for i in 0 ..< dStage.len():
      doAssert splitSecondBits[i] == splitFirstBits[i],
        name & ": split-K d differs run to run at element " & $i
    assertAllClose(dStage.to(float64).view(), expected.to(float64).view(),
      rtol = band.rtol, abstol = band.abstol, msg = name & " split-K")
    judgeGroupedGemv(dStage, xStage, bStage, ids, weights, name & " split-K exact-zero rows")
  echo name & ": T=" & $meta.t & " topk=" & $meta.topk & " E=" & $meta.e &
    " K=" & $meta.k & " N=" & $meta.n & " relaunch bit-identical"

# ═══════════════════════════════════════════════════════════════════════
#  Analytic cases
# ═══════════════════════════════════════════════════════════════════════

proc fuzzCase*(engine: MetalEngine) =
  ## Seeded fuzz, fp32, float64 reference, seed printed.
  ##
  ## - 8 random cases, T in [1, 8], topk in [1, 6], E in [1, 64], K and N in [8, 70]
  ## - about one slot id in ten is forced to -1
  ## - every third case forces token 0's slots all -1 (d[0, :] must come out exactly 0)
  var rng = initRand(FuzzSeed)
  echo "fuzz seed: ", FuzzSeed
  for i in 0 ..< 8:
    let
      tok = rng.rand(1 .. 8)
      topk = rng.rand(1 .. 6)
      e = rng.rand(1 .. 64)
      k = rng.rand(8 .. 70)
      n = rng.rand(8 .. 70)
    groupedGemvCase(engine, float32, tok, topk, e, k, n, rng,
      forceInvalidToken = i mod 3 == 2, "fuzz case " & $i, kSplits = 4)

proc undersizedGridCase*[T](engine: MetalEngine, _: typedesc[T], tok, topk, e, k, n: int, rng: var Rand, msg: string) =
  ## Both dispatch forms on HALF the scheduling contract's block-axis grid:
  ## ceil_div(n-blocks, 2) threadgroups for the single launch and the reduce stage,
  ## ceil_div(n-blocks·kSplits, 2) for the split-K stage 1.
  ##
  ## Invariants:
  ## - the grid contract is exact-cover, so an output element the reduced grid fails
  ##   to reach keeps its sentinel prefill and fails the judgment
  ## - the kernel must cover every n-block by striding the block axis with the grid width
  when T is float16:
    const dtypeLabel = "fp16"
  else:
    const dtypeLabel = "fp32"
  doAssert n >= 2 * 32 * GroupedGemvOutputs,
    "the undersized case needs 2 or more n-blocks, else the half grid is the full grid"
  let (xHost, bHost, idsHost, weightsHost, idsSeq, weightsSeq) =
    buildGroupedGemvInputs(T, rng, tok, topk, e, k, n, forceInvalidToken = false)
  var dHost = makeHostTensor(T, (tok, n), (n, 1))

  fillSentinel(dHost)
  launchGroupedGemv(engine, dHost, xHost, bHost, idsHost, weightsHost, topk, gridDiv = 2)
  judgeGroupedGemv(dHost, xHost, bHost, idsSeq, weightsSeq,
    msg & " single launch, half grid (" & dtypeLabel & ", T=" & $tok & ", N=" & $n & ")")

  fillSentinel(dHost)
  launchGroupedGemvSplitK(engine, dHost, xHost, bHost, idsHost, weightsHost,
    topk, kSplits = 4, gridDiv = 2)
  judgeGroupedGemv(dHost, xHost, bHost, idsSeq, weightsSeq,
    msg & " split-K stage 1, half grid (" & dtypeLabel & ", T=" & $tok & ", N=" & $n & ")")

  fillSentinel(dHost)
  launchGroupedGemvSplitK(engine, dHost, xHost, bHost, idsHost, weightsHost,
    topk, kSplits = 4, reduceGridDiv = 2)
  judgeGroupedGemv(dHost, xHost, bHost, idsSeq, weightsSeq,
    msg & " split-K reduce, half grid (" & dtypeLabel & ", T=" & $tok & ", N=" & $n & ")")
