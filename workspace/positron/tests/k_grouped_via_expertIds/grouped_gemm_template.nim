## grouped_gemm suite machinery shared by the drivers, per backend:
## - fixtures recorded by gen_grouped_gemm_fixtures.py, replayed and judged per element
## - analytic cases generated host-side with a seeded rng
## - the reference is a float64 host GEMM over the same rounded inputs
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
import grouped_gemm_fixtures

export int_tuples, layout_algebra, tensors, tile_algebra, k_grouped_gemv_gemm
export grouped_gemm_fixtures

const
  Fp16Rtol* = 1e-2'f64
  Fp16Abstol* = 1e-2'f64
  Fp32Rtol* = 1e-4'f64
  Fp32Abstol* = 1e-4'f64

const FuzzSeed* = 20261112

const GroupedGemmSentinel = -999.0'f32
  ## Pre-fill value outside the fixtures' range: any line the kernel fails to write keeps the sentinel and fails the judgment

template groupedGemmBand(T: typedesc): tuple[rtol, abstol: float64] =
  (when T is float16: (rtol: Fp16Rtol, abstol: Fp16Abstol)
   else: (rtol: Fp32Rtol, abstol: Fp32Abstol))

# ═══════════════════════════════════════════════════════════════════════
#  float64 host reference
# ═══════════════════════════════════════════════════════════════════════

proc fillSentinel[T, L](ht: var HostTensor[T, L]) =
  for i in 0 ..< ht.len():
    ht.data[i] = to(GroupedGemmSentinel, T)

proc groupedGemmRefF64[T, LA, LB](a: HostTensor[T, LA], b: HostTensor[T, LB], ids: seq[int]): auto =
  ## float64 reference of the grouped GEMM over the same rounded inputs.
  ##
  ## - a line with id >= 0 receives a(id)·b(id)
  ## - a padding line (id -1) inside an m tile with at least one valid line
  ##   receives the accumulator's zero-init
  ## - a padding line inside an all-padding m tile is not stored, it keeps
  ##   the destination's pre-fill sentinel
  let
    totalM = a.layout.shape[0].toInt()
    k = a.layout.shape[1].toInt()
    n = b.layout.shape[2].toInt()
    (mt, _, _) = groupedGemmTiles(T)
  result = makeHostTensor(float64, (totalM, n), (n, 1))
  for i in 0 ..< totalM:
    let g = ids[i]
    if g >= 0:
      for j in 0 ..< n:
        var acc = 0.0'f64
        for e in 0 ..< k:
          acc += to(a[i, e], float64) * to(b[g, e, j], float64)
        result[i, j] = acc
  for i in 0 ..< totalM:
    if ids[i] < 0:
      let
        m0 = (i div mt) * mt
        mEnd = min(m0 + mt, totalM)
      var tileHasValidLine = false
      for j in m0 ..< mEnd:
        if ids[j] >= 0:
          tileHasValidLine = true
      if tileHasValidLine:
        for j in 0 ..< n:
          result[i, j] = 0.0'f64
      else:
        for j in 0 ..< n:
          result[i, j] = GroupedGemmSentinel.float64

# ═══════════════════════════════════════════════════════════════════════
#  Launch and judgment
# ═══════════════════════════════════════════════════════════════════════

proc launchGroupedGemm[T, LA, LB, LG, LD](
    engine: MetalEngine, gridWidth: int,
    dHost: var HostTensor[T, LD], aHost: HostTensor[T, LA],
    bHost: HostTensor[T, LB], idsHost: HostTensor[int32, LG]) =
  ## Bridges the fixtures and launches `grouped_gemm` over the caller's `gridWidth` threadgroups, copying d back into `dHost`.
  ##
  ## At totalM = 0 the width is 0: nothing is bridged and no launch is made,
  ## the kernel's documented caller duty.
  if gridWidth > 0:
    let
      aDev = engine.bridge(aHost)
      bDev = engine.bridge(bHost)
      idsDev = engine.bridge(idsHost)
      dDev = engine.bridge(dHost)
    engine.launchDeviceFn grouped_gemm_via_expertIds(dDev, aDev, bDev, idsDev) do:
      grid = (gridWidth, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(dHost.data.toOpenArray(0, dHost.len() - 1), dDev.data)

proc judgeGroupedGemm[T, LD, LA, LB](
    got: HostTensor[T, LD], aHost: HostTensor[T, LA], bHost: HostTensor[T, LB],
    ids: openArray[int], msg: string) =
  ## float64 reference over the per-line ids, judged against the kernel
  ## output with the element band.
  var idsSeq: seq[int]
  for i in 0 ..< ids.len:
    idsSeq.add(ids[i])
  let refHost = groupedGemmRefF64(aHost, bHost, idsSeq)
  const band = groupedGemmBand(T)
  assertAllClose(got.to(float64).view(), refHost.view(),
    rtol = band.rtol, abstol = band.abstol, msg = msg)

proc groupedGemmCase*[T](engine: MetalEngine, _: typedesc[T], ids: openArray[int], e, k, n: int, rng: var Rand, msg: string) =
  ## One grouped_gemm case: per-line expert ids `ids` over an (e, k, n) B stack,
  ## distinct B per expert.
  ##
  ## - fixtures generated from `rng`, launched through the public API over
  ##   the shape-derived static grid
  ## - every d line judged against the float64 reference
  when T is float16:
    const dtypeLabel = "fp16"
  else:
    const dtypeLabel = "fp32"
  let totalM = ids.len
  var idsSeq: seq[int]
  for i in 0 ..< totalM:
    idsSeq.add(ids[i])
  var
    aHost = rng.randomHostTensor(T, (totalM, k), (k, 1))
    bHost = rng.randomHostTensor(T, (e, k, n),
                                 (k * n, n, 1))
    idsHost = makeHostTensor(int32, (totalM,), (1,))
    dHost = makeHostTensor(T, (totalM, n), (n, 1))
  for i in 0 ..< totalM:
    idsHost.data[i] = int32(idsSeq[i])
  fillSentinel(dHost)
  launchGroupedGemm(engine, countGroupedGemmTiles(T, totalM, n), dHost, aHost, bHost, idsHost)
  judgeGroupedGemm(dHost, aHost, bHost, idsSeq,
    msg & " (" & dtypeLabel & ", E=" & $e & ", totalM=" & $totalM &
    ", K=" & $k & ", N=" & $n & ")")

# ═══════════════════════════════════════════════════════════════════════
#  Fixture replay
# ═══════════════════════════════════════════════════════════════════════

proc groupedGemmReplayCase*(engine: MetalEngine, Elem: typedesc, name: string) =
  ## Replays one fixture, with relaunch:
  ##
  ## - one launch over the shape-derived static grid, per-element band judgment
  ##   against the recorded f64 reference
  ## - a relaunch judged bit-identical on the raw output bits
  ##
  ## Preconditions:
  ## - the fixtures use the kernel's 32-line m tile for their padding-line store contract, checked below
  ## - the destination staging is sentinel-filled
  ## - the recorded reference has the same sentinel on unstored lines: an unwritten line fails the judgment
  const wantDtype = when Elem is float16: "float16" else: "float32"
  let dir = GroupedGemmFixturesDir / name
  let meta = parseGroupedGemmMetadata(dir)
  doAssert meta.dtype == wantDtype, meta.name & ": dtype " & meta.dtype
  doAssert meta.mTile == 32, meta.name & ": m tile " & $meta.mTile & " != the kernel's 32"
  doAssert meta.counts.len == meta.e and meta.pads.len == meta.e,
    meta.name & ": one (count, pad) pair per expert"

  let
    aBits = loadFixtureTensor(dir, "a.bin", (meta.totalM, meta.k),
      when Elem is float16: uint16 else: uint32)
    bBits = loadFixtureTensor(dir, "b.bin", (meta.e, meta.k, meta.n),
      when Elem is float16: uint16 else: uint32)
    expected = loadFixtureTensor(dir, "d.bin", (meta.totalM, meta.n), float32)
    idsHost = loadIdsFixtureTensor(dir, "ids.bin", meta.totalM)
  doAssert xorHash(aBits) == meta.xorhashChecksum("a.bin"), meta.name & ": a.bin checksum"
  doAssert xorHash(bBits) == meta.xorhashChecksum("b.bin"), meta.name & ": b.bin checksum"
  doAssert xorHash(expected) == meta.xorhashChecksum("d.bin"), meta.name & ": d.bin checksum"
  doAssert xorHash(idsHost) == meta.xorhashChecksum("ids.bin"), meta.name & ": ids.bin checksum"

  var
    aStage = makeHostTensor(Elem, (meta.totalM, meta.k), LayoutRight)
    bStage = makeHostTensor(Elem, (meta.e, meta.k, meta.n), LayoutRight)
    dStage = makeHostTensor(Elem, (meta.totalM, meta.n), LayoutRight)
  fillSentinel(dStage)
  for i in 0 ..< aStage.len():
    aStage.data[i] = cast[Elem](aBits.data[i])
  for i in 0 ..< bStage.len():
    bStage.data[i] = cast[Elem](bBits.data[i])

  var ids: seq[int]
  for i in 0 ..< meta.totalM:
    ids.add(int(idsHost.data[i]))

  proc captureBits(): seq[uint32] =
    result = newSeq[uint32](dStage.len())
    for i in 0 ..< dStage.len():
      when Elem is float16:
        result[i] = uint32(cast[uint16](dStage.data[i]))
      else:
        result[i] = cast[uint32](dStage.data[i])

  let gridWidth = countGroupedGemmTiles(Elem, meta.totalM, meta.n)
  launchGroupedGemm(engine, gridWidth, dStage, aStage, bStage, idsHost)
  let firstBits = captureBits()

  fillSentinel(dStage)
  launchGroupedGemm(engine, gridWidth, dStage, aStage, bStage, idsHost)
  let secondBits = captureBits()
  for i in 0 ..< dStage.len():
    doAssert secondBits[i] == firstBits[i],
      name & ": d differs run to run at element " & $i

  const band = groupedGemmBand(Elem)
  assertAllClose(dStage.to(float64).view(), expected.to(float64).view(),
    rtol = band.rtol, abstol = band.abstol, msg = name)
  echo name & ": E=" & $meta.e & " totalM=" & $meta.totalM &
    " K=" & $meta.k & " N=" & $meta.n & " relaunch bit-identical"

# ═══════════════════════════════════════════════════════════════════════
#  Analytic cases
# ═══════════════════════════════════════════════════════════════════════

proc buildIds(counts, pads: openArray[int]): seq[int] =
  ## Per-line expert ids from per-expert valid counts and -1 tail pads
  for g in 0 ..< counts.len:
    for _ in 0 ..< counts[g]:
      result.add(g)
    for _ in 0 ..< pads[g]:
      result.add(-1)

proc fuzzCase*(engine: MetalEngine) =
  ## Seeded fuzz, fp32, float64 reference, seed printed.
  ##
  ## - 8 random cases, E in [1, 16], K and N in [8, 70]
  ## - per-expert valid counts in [0, 40], one forced count 0 and one forced
  ##   count 1 when E >= 2
  ## - -1 tail pads in [0, 8] per expert
  var rng = initRand(FuzzSeed)
  echo "fuzz seed: ", FuzzSeed
  for i in 0 ..< 8:
    let e = rng.rand(1 .. 16)
    var
      counts = newSeq[int](e)
      pads = newSeq[int](e)
    for g in 0 ..< e:
      counts[g] = rng.rand(0 .. 40)
      pads[g] = rng.rand(0 .. 8)
    if e >= 2:
      counts[0] = 0
      counts[1] = 1
    let
      ids = buildIds(counts, pads)
      k = rng.rand(8 .. 70)
      n = rng.rand(8 .. 70)
    groupedGemmCase(engine, float32, ids, e, k, n, rng, "fuzz case " & $i)

proc graphCaptureCase*(engine: MetalEngine) =
  ## Graph-capture compatibility over one static grid.
  ##
  ## - the grid is computed once from the view shapes, both launches reuse it
  ## - the ids mutate between the launches: two valid lines flip to -1 and padding moves to another expert's tail
  ## - each launch is judged against its own float64 reference
  var rng = initRand(31)
  const
    e = 4
    k = 40
    n = 48
    totalM = 44
    countsA = [5, 8, 3, 12]
    padsA = [4, 0, 8, 4]
    countsB = [3, 10, 3, 12]
    padsB = [6, 0, 6, 4]
  let
    idsA = buildIds(countsA, padsA)
    idsB = buildIds(countsB, padsB)
    aHost = rng.randomHostTensor(float32, (totalM, k), (k, 1))
    bHost = rng.randomHostTensor(float32, (e, k, n),
                                 (k * n, n, 1))
    idsHostA = makeHostTensor(int32, (totalM,), (1,))
    idsHostB = makeHostTensor(int32, (totalM,), (1,))
  var dHost = makeHostTensor(float32, (totalM, n), (n, 1))
  for i in 0 ..< totalM:
    idsHostA.data[i] = int32(idsA[i])
    idsHostB.data[i] = int32(idsB[i])

  let gridWidth = countGroupedGemmTiles(float32, totalM, n)
  fillSentinel(dHost)
  launchGroupedGemm(engine, gridWidth, dHost, aHost, bHost, idsHostA)
  judgeGroupedGemm(dHost, aHost, bHost, idsA, "graph capture, state A")

  fillSentinel(dHost)
  launchGroupedGemm(engine, gridWidth, dHost, aHost, bHost, idsHostB)
  judgeGroupedGemm(dHost, aHost, bHost, idsB, "graph capture, state B")
  echo "graph-capture replay: grid=" & $gridWidth &
    " reused across both launches, both id states judged"
