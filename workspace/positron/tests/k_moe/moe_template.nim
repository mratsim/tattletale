## MoE frontend suite machinery shared by the drivers:
## router top-k, dispatch/pack, combine, and the moeForward driver.
##
## Judgment: analytic cases against a float64 host reference and fixture replays
## against the recorded python float64 reference, both over the same rounded inputs.
## Bands per output element, |got - want| <= Rtol*|want| + Abstol:
## fp32 (1e-4, 1e-4), fp16 (1e-2, 1e-2), router weights fp32 end to end, ids exact.

import std/[math, os, random]
import workspace/math
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/positron/src/kernels/k_grouped_gemv_gemm {.all.}
import workspace/positron/src/kernels/k_moe_forward {.all.}
import workspace/positron/src/kernels/k_moe_router {.all.}
import ../positron_testutils
import moe_fixtures
import moe_forward

export int_tuples, layout_algebra, tensors, tile_algebra, k_moe_router, k_grouped_gemv_gemm, k_moe_forward, moe_fixtures, moe_forward

const
  Fp16Rtol* = 1e-2'f64
  Fp16Abstol* = 1e-2'f64
  Fp32Rtol* = 1e-4'f64
  Fp32Abstol* = 1e-4'f64

const FuzzSeed* = 20261114

const MoERouterSentinel = -999.0'f32
  ## Pre-fill value outside the fixtures' range: any weight the kernel fails to write keeps the sentinel and fails the judgment

# ═══════════════════════════════════════════════════════════════════════
#  float64 host reference (router)
# ═══════════════════════════════════════════════════════════════════════

proc moeRouterRefF64[T, L](scores: HostTensor[T, L], bias: openArray[float32], topk: int, variant: MoERoute, scalingFactor: float32):
    tuple[ids: seq[int], weights: seq[float64]] =
  ## float64 reference of the router top-k over the same rounded inputs.
  ##
  ## - selection on scores + bias, ascending scan, LOWER id wins ties
  ## - weights from the unbiased scores: softmax over all E (max-shifted) or sigmoid
  ## - renorm over the selected set, w = w / max(sum w, 1e-20), then ·scalingFactor
  ## - topk > E: the excess slots carry id -1 and weight 0
  let
    t = scores.layout.shape[0].toInt()
    e = scores.layout.shape[1].toInt()
  result.ids = newSeq[int](t * topk)
  result.weights = newSeq[float64](t * topk)
  for ti in 0 ..< t:
    # selection: scores + bias, strict comparisons keep the lower id
    var
      selVal = newSeq[float64](topk)
      selIdx = newSeq[int](topk)
      nSel = 0
    for i in 0 ..< e:
      let s = to(scores[ti, i], float64) + bias[i].float64
      var insert = nSel < topk
      if not insert and s > selVal[topk - 1]:
        insert = true
      if insert:
        var pos = nSel
        if pos > topk - 1:
          pos = topk - 1
        while pos > 0 and selVal[pos - 1] < s:
          selVal[pos] = selVal[pos - 1]
          selIdx[pos] = selIdx[pos - 1]
          pos -= 1
        selVal[pos] = s
        selIdx[pos] = i
        if nSel < topk:
          nSel += 1
    # unbiased weights at the selected experts
    var
      w = newSeq[float64](nSel)
      wSum = 0'f64
    if variant == moeSoftmax:
      var m = -Inf
      for j in 0 ..< e:
        let v = to(scores[ti, j], float64)
        if v > m:
          m = v
      var z = 0'f64
      for j in 0 ..< e:
        z += exp(to(scores[ti, j], float64) - m)
      for i in 0 ..< nSel:
        w[i] = exp(to(scores[ti, selIdx[i]], float64) - m) / z
    else:
      for i in 0 ..< nSel:
        w[i] = 1'f64 / (1'f64 + exp(-to(scores[ti, selIdx[i]], float64)))
    for i in 0 ..< nSel:
      wSum += w[i]
    let denom = max(wSum, 1e-20)
    for i in 0 ..< nSel:
      result.ids[ti * topk + i] = selIdx[i]
      result.weights[ti * topk + i] = w[i] / denom * scalingFactor.float64
    for i in nSel ..< topk:
      result.ids[ti * topk + i] = -1
      result.weights[ti * topk + i] = 0'f64

# ═══════════════════════════════════════════════════════════════════════
#  Launch and judgment (router)
# ═══════════════════════════════════════════════════════════════════════

proc launchMoeRouter[LS, LB](engine: MetalEngine,
    scoresHost: HostTensor[float32, LS], biasHost: HostTensor[float32, LB],
    topk: int, variant: MoERoute, scalingFactor: float32): auto =
  ## Bridges the fixtures, launches `moe_router_topk` over the (1, T) grid, and returns the ids and weights as host tensors.
  ##
  ## ids and weights are fresh sentinel-filled host tensors. The ids stage is
  ## int32-filled with a -2 pre-fill so an unwritten slot fails the exact judgment.
  let
    tok = scoresHost.layout.shape[0].toInt()
    scoresDev = engine.bridge(scoresHost)
    biasDev = engine.bridge(biasHost)
  var
    idsHost = makeHostTensor(int32, (tok, topk), (topk, 1))
    weightsHost = makeHostTensor(float32, (tok, topk), (topk, 1))
  for i in 0 ..< idsHost.len():
    idsHost.data[i] = -2
  for i in 0 ..< weightsHost.len():
    weightsHost.data[i] = MoERouterSentinel
  let
    idsDev = engine.bridge(idsHost)
    weightsDev = engine.bridge(weightsHost)
  engine.launchDeviceFn moe_router_topk(idsDev, weightsDev, scoresDev, biasDev,
      topk, ord(variant), scalingFactor) do:
    grid = (1, tok)
    blk = (32, 1)
  engine.copyToHostFromDevice(idsHost.data.toOpenArray(0, idsHost.len() - 1), idsDev.data)
  engine.copyToHostFromDevice(weightsHost.data.toOpenArray(0, weightsHost.len() - 1), weightsDev.data)
  (idsHost, weightsHost)

proc judgeMoeRouter[LI, LW](gotIds: HostTensor[int32, LI], gotWeights: HostTensor[float32, LW],
    refIds: seq[int], refWeights: seq[float64], msg: string) =
  ## Exact id judgment and weight band judgment. The ids compare exactly:
  ## the selection is an order statistic over the same rounded fp32 inputs.
  ##
  ## The weights band is the fp32 band. A renormed weight is O(1), and the fp32
  ## exp2/exp gap stays inside one relative ulp chain.
  const band = (rtol: Fp32Rtol, abstol: Fp32Abstol)
  let topk = gotWeights.layout.shape[1].toInt()
  for ti in 0 ..< gotWeights.layout.shape[0].toInt():
    for i in 0 ..< topk:
      doAssert int(gotIds[ti, i]) == refIds[ti * topk + i],
        msg & ": ids[" & $ti & ", " & $i & "] = " & $int(gotIds[ti, i]) &
        " want " & $refIds[ti * topk + i] & " (selection must match exactly)"
  var wRef = makeHostTensor(float64, gotWeights.layout.shape, gotWeights.layout.stride)
  for i in 0 ..< wRef.len():
    wRef.data[i] = refWeights[i]
  let report = gotWeights.to(float64).view().allClose(wRef.view(),
    rtol = band.rtol, abstol = band.abstol)
  doAssert report.bad == 0,
    msg & ": " & $report.bad & " mismatched weights, worst |d| = " & $report.worst &
    ", first at " & report.firstCrd & ": got " & $report.firstGot &
    " want " & $report.firstWant

# ═══════════════════════════════════════════════════════════════════════
#  Analytic cases (router)
# ═══════════════════════════════════════════════════════════════════════

proc moeRouterCase*(engine: MetalEngine, tok, e, topk: int, variant: MoERoute, biasScale, scalingFactor: float32, rng: var Rand, msg: string) =
  ## One router case: `tok` tokens over `e` experts, `topk` slots.
  ##
  ## - scores from `rng` in [-2, 2], fp32, no rounding below fp32
  ## - biasScale 0 stages a zero-filled bias (the no-bias model), otherwise a biased selection whose scale shifts WHICH experts win
  ## - ids judged exactly against the float64 reference, weights within the fp32 band
  doAssert topk <= MoERouterMaxTopk, "topk above the router's insertion buffer"
  var
    scoresHost = rng.randomHostTensor(float32, (tok, e), (e, 1),
      bounds = Slice[float32](a: -2'f32, b: 2'f32))
    biasHost = makeHostTensor(float32, (e,), (1,))
  for i in 0 ..< e:
    biasHost[i] = to(rng.rand(-biasScale .. biasScale), float32)
  let (refIds, refWeights) = moeRouterRefF64(scoresHost, biasHost.data.toOpenArray(0, e - 1),
    topk, variant, scalingFactor)
  let (gotIds, gotWeights) = launchMoeRouter(engine, scoresHost, biasHost, topk, variant, scalingFactor)
  judgeMoeRouter(gotIds, gotWeights, refIds, refWeights,
    msg & " (T=" & $tok & ", E=" & $e & ", topk=" & $topk & ", variant=" &
    $variant & ", biasScale=" & $biasScale & ")")
  echo msg & ": T=" & $tok & " E=" & $e & " topk=" & $topk & " variant=" & $variant &
    " biasScale=" & $biasScale & " scaling=" & $scalingFactor

proc moeRouterBiasSelectCase*(engine: MetalEngine) =
  ## Bias-select-only property, constructively: a strong-enough bias changes
  ## WHICH experts are selected, with the weights still the unbiased scores'
  ## transform at the selected ids.
  ##
  ## - one token, E = 8 experts with strictly ordered scores, topk = 3
  ## - bias +10 on the two lowest-scoring experts flips the selection to them
  ## - the selection without bias differs from the selection with bias
  ## - the biased run's weights equal the unbiased softmax/sigmoid at its ids,
  ##   judged against the float64 reference like every case
  var rng = initRand(7)
  let
    e = 8
    topk = 3
  var scoresHost = makeHostTensor(float32, (1, e), (e, 1))
  for i in 0 ..< e:
    scoresHost[0, i] = to(-2.0'f32 + 0.5'f32 * float32(i), float32)
  var
    biasZero = makeHostTensor(float32, (e,), (1,))
    biasFlip = makeHostTensor(float32, (e,), (1,))
  for i in 0 ..< e:
    biasZero[i] = 0'f32
    biasFlip[i] = 0'f32
  biasFlip[0] = 10'f32
  biasFlip[1] = 10'f32

  let (idsPlain, _) = launchMoeRouter(engine, scoresHost, biasZero, topk, moeSoftmax, 1.0'f32)
  let (idsFlipped, wFlipped) = launchMoeRouter(engine, scoresHost, biasFlip, topk, moeSoftmax, 1.0'f32)
  var plainSel: seq[int]
  var flippedSel: seq[int]
  for i in 0 ..< topk:
    plainSel.add(int(idsPlain[0, i]))
    flippedSel.add(int(idsFlipped[0, i]))
  # Ids come out in descending selection-score order: experts 5, 6 and 7 hold the three highest scores, bias +10 puts experts 1 and 0 on top.
  doAssert plainSel == @[7, 6, 5],
    "no-bias selection must be the three highest scores in descending order, got " & $plainSel
  doAssert flippedSel == @[1, 0, 7],
    "the +10 bias on experts 0 and 1 must flip the selection, got " & $flippedSel
  let (refIds, refWeights) = moeRouterRefF64(scoresHost, biasFlip.data.toOpenArray(0, e - 1),
    topk, moeSoftmax, 1.0'f32)
  judgeMoeRouter(idsFlipped, wFlipped, refIds, refWeights, "bias-select-only property")
  echo "bias-select-only: selection flipped by bias, weights match the unbiased softmax at the flipped ids"

proc moeRouterTieCase*(engine: MetalEngine) =
  ## Tie-break contract, engineered: equal selection scores select the LOWER expert id, judged against the contract's literal ids.
  ##
  ## - one token, E = 4 experts, topk = 2, zero bias
  ## - all four scores equal: the ascending scan selects ids 0 and 1
  ## - scores 1.0, 0.5, 0.5, 0.3: the 0.5 tie straddles the topk boundary and expert 1 must win it over expert 2
  ## - the ids are the hand-derived contract literal, weights judged against the float64 reference (all-equal scores renorm to 0.5 per expert)
  let
    e = 4
    topk = 2
  var
    scoresEqual = makeHostTensor(float32, (1, e), (e, 1))
    scoresBoundary = makeHostTensor(float32, (1, e), (e, 1))
    biasHost = makeHostTensor(float32, (e,), (1,))
  for i in 0 ..< e:
    scoresEqual[0, i] = 0.5'f32
    biasHost[i] = 0'f32
  scoresBoundary[0, 0] = 1.0'f32
  scoresBoundary[0, 1] = 0.5'f32
  scoresBoundary[0, 2] = 0.5'f32
  scoresBoundary[0, 3] = 0.3'f32
  let (idsEqual, wEqual) = launchMoeRouter(engine, scoresEqual, biasHost, topk, moeSoftmax, 1.0'f32)
  doAssert int(idsEqual[0, 0]) == 0 and int(idsEqual[0, 1]) == 1,
    "all-equal scores must select the two lowest ids, got [" & $int(idsEqual[0, 0]) &
    ", " & $int(idsEqual[0, 1]) & "]"
  let (refIdsEq, refWEq) = moeRouterRefF64(scoresEqual, biasHost.data.toOpenArray(0, e - 1),
    topk, moeSoftmax, 1.0'f32)
  judgeMoeRouter(idsEqual, wEqual, refIdsEq, refWEq, "tie all-equal scores")
  let (idsBound, wBound) = launchMoeRouter(engine, scoresBoundary, biasHost, topk, moeSoftmax, 1.0'f32)
  doAssert int(idsBound[0, 0]) == 0 and int(idsBound[0, 1]) == 1,
    "the 0.5 tie straddling the topk boundary must select expert 1 over expert 2, got [" &
    $int(idsBound[0, 0]) & ", " & $int(idsBound[0, 1]) & "]"
  let (refIdsBd, refWBd) = moeRouterRefF64(scoresBoundary, biasHost.data.toOpenArray(0, e - 1),
    topk, moeSoftmax, 1.0'f32)
  judgeMoeRouter(idsBound, wBound, refIdsBd, refWBd, "tie at the topk boundary")
  echo "router tie-break: all-equal and boundary ties select the lower expert id"

# ═══════════════════════════════════════════════════════════════════════
#  Dispatch/pack: float64 host reference, launch and judgment
# ═══════════════════════════════════════════════════════════════════════

proc moeDispatchRef(ids: openArray[int], t, topk, e: int):
    tuple[packedIds, sortedPairs, slotOfPair: seq[int], packedOffsets: seq[int]] =
  ## Integer-exact reference of the dispatch metadata over the same ids table.
  ##
  ## (token, slot) pairs in table order, sort key (expert, enumeration index),
  ## blocks padded to MoEDispatchPadM, -1 padding.
  result.packedOffsets = newSeq[int](e)
  result.sortedPairs = newSeq[int](t * topk + (32 - 1) * e)
  result.slotOfPair = newSeq[int](t * topk)
  result.packedIds = newSeq[int](result.sortedPairs.len)
  for s in 0 ..< result.sortedPairs.len:
    result.sortedPairs[s] = -1
  for i in 0 ..< result.slotOfPair.len:
    result.slotOfPair[i] = -1
  var counts = newSeq[int](e)
  var ranks = newSeq[int](t * topk)
  for tp in 0 ..< t:
    for ip in 0 ..< topk:
      let g = ids[tp * topk + ip]
      if g >= 0:
        ranks[tp * topk + ip] = counts[g]
        counts[g] += 1
  var acc = 0
  for g in 0 ..< e:
    acc += (counts[g] + 31) div 32 * 32
    result.packedOffsets[g] = acc
  for tp in 0 ..< t:
    for ip in 0 ..< topk:
      let g = ids[tp * topk + ip]
      if g >= 0:
        let start = result.packedOffsets[g] - (counts[g] + 31) div 32 * 32
        let slot = start + ranks[tp * topk + ip]
        result.sortedPairs[slot] = tp
        result.slotOfPair[tp * topk + ip] = slot
  for g in 0 ..< e:
    let start = result.packedOffsets[g] - (counts[g] + 31) div 32 * 32
    let c = counts[g]
    for s in start + c ..< result.packedOffsets[g]:
      result.sortedPairs[s] = -1
  for slot in 0 ..< result.packedIds.len:
    var expert = -1
    if result.sortedPairs[slot] >= 0:
      for g in 0 ..< e:
        if result.packedOffsets[g] > slot:
          expert = g
          break
    result.packedIds[slot] = expert

proc launchMoeDispatch[LI, LX](engine: MetalEngine, idsHost: HostTensor[int32, LI], xHost: HostTensor[float32, LX], topk, e: int): auto =
  ## Bridges the fixtures and launches the metadata and gather kernels,
  ## returning every output as host tensors over their shape-derived static grids.
  ##
  ## sortedPairs is staged pre-filled with -1, packedIds and slotOfPair with -2,
  ## packedX with the dispatch sentinel, so an unwritten slot or entry fails
  ## the exact judgment. The static grids come from the shapes, and both launches
  ## reuse them (graph-capture discipline, no host read of device data).
  let
    tok = idsHost.layout.shape[0].toInt()
    k = xHost.layout.shape[1].toInt()
    cap = moeCapacityBound(tok, topk, e)
  var
    offsetsHost = makeHostTensor(int32, (e,), (1,))
    sortedPairsHost = makeHostTensor(int32, (cap,), (1,))
    slotOfPairHost = makeHostTensor(int32, (tok, topk), (topk, 1))
    packedIdsHost = makeHostTensor(int32, (cap,), (1,))
    packedXHost = makeHostTensor(float32, (cap, k), (k, 1))
  for i in 0 ..< cap:
    sortedPairsHost.data[i] = -1
    packedIdsHost.data[i] = -2
  for i in 0 ..< tok * topk:
    slotOfPairHost.data[i] = -2
  for i in 0 ..< packedXHost.len():
    packedXHost.data[i] = MoERouterSentinel
  let
    idsDev = engine.bridge(idsHost)
    xDev = engine.bridge(xHost)
    offsetsDev = engine.bridge(offsetsHost)
    sortedPairsDev = engine.bridge(sortedPairsHost)
    slotOfPairDev = engine.bridge(slotOfPairHost)
    packedIdsDev = engine.bridge(packedIdsHost)
    packedXDev = engine.bridge(packedXHost)
  engine.launchDeviceFn moe_dispatch_metadata(offsetsDev, sortedPairsDev, slotOfPairDev, idsDev, topk) do:
    grid = (tok.ceil_div(32), 1)
    blk = (32, 1)
  engine.launchDeviceFn moe_dispatch_gather(packedIdsDev, packedXDev, xDev, sortedPairsDev, offsetsDev) do:
    grid = ((cap * k).ceil_div(32), 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(offsetsHost.data.toOpenArray(0, offsetsHost.len() - 1), offsetsDev.data)
  engine.copyToHostFromDevice(sortedPairsHost.data.toOpenArray(0, sortedPairsHost.len() - 1), sortedPairsDev.data)
  engine.copyToHostFromDevice(slotOfPairHost.data.toOpenArray(0, slotOfPairHost.len() - 1), slotOfPairDev.data)
  engine.copyToHostFromDevice(packedIdsHost.data.toOpenArray(0, packedIdsHost.len() - 1), packedIdsDev.data)
  engine.copyToHostFromDevice(packedXHost.data.toOpenArray(0, packedXHost.len() - 1), packedXDev.data)
  (offsetsHost, sortedPairsHost, slotOfPairHost, packedIdsHost, packedXHost)

proc judgeMoeDispatch[LI, LX](engine: MetalEngine, idsHost: HostTensor[int32, LI],
    xHost: HostTensor[float32, LX], topk, e: int, msg: string) =
  ## One dispatch case: exact integer judgment of the metadata against the reference,
  ## exact packedX judgment at the real slots, sentinel judgment of the padding entries,
  ## and a relaunch judged byte-identical.
  ##
  ## The relaunch runs on the SAME staged device buffers: only the metadata launch
  ## rewrites its outputs, so byte-identity across launches is the determinism
  ## contract and applies unconditionally.
  let
    tok = idsHost.layout.shape[0].toInt()
    k = xHost.layout.shape[1].toInt()
  var idsSeq: seq[int]
  for i in 0 ..< tok * topk:
    idsSeq.add(int(idsHost.data[i]))
  let refMeta = moeDispatchRef(idsSeq, tok, topk, e)
  let (offsetsHost, sortedPairsHost, slotOfPairHost, packedIdsHost, packedXHost) =
    launchMoeDispatch(engine, idsHost, xHost, topk, e)
  proc captureBits(): seq[uint32] =
    result = newSeq[uint32](offsetsHost.len() + sortedPairsHost.len() +
      slotOfPairHost.len() + packedIdsHost.len())
    var o = 0
    for i in 0 ..< offsetsHost.len():
      result[o] = cast[uint32](offsetsHost.data[i]); o += 1
    for i in 0 ..< sortedPairsHost.len():
      result[o] = cast[uint32](sortedPairsHost.data[i]); o += 1
    for i in 0 ..< slotOfPairHost.len():
      result[o] = cast[uint32](slotOfPairHost.data[i]); o += 1
    for i in 0 ..< packedIdsHost.len():
      result[o] = cast[uint32](packedIdsHost.data[i]); o += 1
  let firstBits = captureBits()
  discard launchMoeDispatch(engine, idsHost, xHost, topk, e)
  let secondBits = captureBits()
  doAssert secondBits == firstBits, msg & ": metadata differs run to run"

  let cap = moeCapacityBound(tok, topk, e)
  for g in 0 ..< e:
    doAssert int(offsetsHost[g]) == refMeta.packedOffsets[g],
      msg & ": packedOffsets[" & $g & "] = " & $int(offsetsHost[g]) &
      " want " & $refMeta.packedOffsets[g]
  doAssert int(offsetsHost[e - 1]) <= cap,
    msg & ": padded total " & $int(offsetsHost[e - 1]) & " over the capacity bound " & $cap
  for i in 0 ..< tok * topk:
    doAssert int(slotOfPairHost.data[i]) == refMeta.slotOfPair[i],
      msg & ": slotOfPair[" & $i & "] = " & $int(slotOfPairHost.data[i]) &
      " want " & $refMeta.slotOfPair[i]
  for s in 0 ..< cap:
    doAssert int(sortedPairsHost[s]) == refMeta.sortedPairs[s],
      msg & ": sortedPairs[" & $s & "] = " & $int(sortedPairsHost[s]) &
      " want " & $refMeta.sortedPairs[s]
    doAssert int(packedIdsHost[s]) == refMeta.packedIds[s],
      msg & ": packedIds[" & $s & "] = " & $int(packedIdsHost[s]) &
      " want " & $refMeta.packedIds[s]
    if refMeta.sortedPairs[s] >= 0:
      let src = refMeta.sortedPairs[s]
      for kk in 0 ..< k:
        doAssert to(packedXHost[s, kk], float64) == to(xHost[src, kk], float64),
          msg & ": packedX[" & $s & ", " & $kk & "] mismatch, src token " & $src
    else:
      for kk in 0 ..< k:
        doAssert to(packedXHost[s, kk], float64) == to(MoERouterSentinel, float64),
          msg & ": padding row " & $s & " was written"
  echo msg & ": T=" & $tok & " topk=" & $topk & " E=" & $e & " K=" & $k &
    " cap=" & $cap & " paddedTotal=" & $int(offsetsHost[e - 1]) &
    " relaunch byte-identical"

proc moeDispatchCase*(engine: MetalEngine, ids: seq[seq[int]], e, k: int, rng: var Rand, msg: string) =
  ## One dispatch case over an ids table from the caller, x from `rng`.
  doAssert e <= MoEMaxExperts and e >= 1
  let tok = ids.len
  doAssert tok >= 1
  let topk = ids[0].len
  var idsHost = makeHostTensor(int32, (tok, topk), (topk, 1))
  for tp in 0 ..< tok:
    doAssert ids[tp].len == topk
    for ip in 0 ..< topk:
      doAssert ids[tp][ip] >= -1 and ids[tp][ip] < e
      idsHost[tp, ip] = int32(ids[tp][ip])
  let xHost = rng.randomHostTensor(float32, (tok, k), (k, 1),
    bounds = Slice[float32](a: -2'f32, b: 2'f32))
  judgeMoeDispatch(engine, idsHost, xHost, topk, e, msg)

# ═══════════════════════════════════════════════════════════════════════
#  Combine: float64 host reference, launch and judgment
# ═══════════════════════════════════════════════════════════════════════

proc moeCombineRefF64[LD](dPacked: HostTensor[float32, LD],
    slotOfPair: openArray[int], weights: openArray[float32], t, topk, n: int): auto =
  ## float64 reference of the combine over the same rounded inputs.
  ##
  ## - d[t, n] = sum over ascending i of weights(t, i)·dPacked(slotOfPair(t, i), n)
  ## - a slot with map -1 contributes 0
  result = makeHostTensor(float64, (t, n), (n, 1))
  for i in 0 ..< result.len():
    result.data[i] = 0.0'f64
  for ti in 0 ..< t:
    for i in 0 ..< topk:
      let s = slotOfPair[ti * topk + i]
      if s >= 0:
        let w = weights[ti * topk + i].float64
        for j in 0 ..< n:
          result[ti, j] += w * to(dPacked[s, j], float64)

proc launchMoeCombine[LP, LS, LW](engine: MetalEngine, t, topk, n: int,
    dPackedHost: HostTensor[float32, LP], slotOfPairHost: HostTensor[int32, LS],
    weightsHost: HostTensor[float32, LW]): auto =
  ## Bridges the fixtures, launches `moe_combine` over the (n-block, token) grid
  ## and returns d as a host tensor, sentinel-filled before the launch.
  var dHost = makeHostTensor(float32, (t, n), (n, 1))
  for i in 0 ..< dHost.len():
    dHost.data[i] = MoERouterSentinel
  let
    dPackedDev = engine.bridge(dPackedHost)
    slotOfPairDev = engine.bridge(slotOfPairHost)
    weightsDev = engine.bridge(weightsHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn moe_combine(dDev, dPackedDev, slotOfPairDev, weightsDev, topk) do:
    grid = (n.ceil_div(32 * MoECombineOutputs), t)
    blk = (32, 1)
  engine.copyToHostFromDevice(dHost.data.toOpenArray(0, dHost.len() - 1), dDev.data)
  dHost

proc moeCombineCase*(engine: MetalEngine, t, topk, e, n: int, rng: var Rand, msg: string) =
  ## One combine case: a random ids table over `e` experts and the dispatch
  ## reference's inverse map, random dPacked entries and weights.
  ##
  ## - dPacked entries at the mapped slots judged against the float64 reference
  ##   within the fp32 band, unmapped (padding) entries never read
  ## - a token whose slots are all -1 must be exactly 0.0 in every element
  var
    dPackedHost = rng.randomHostTensor(float32,
      (moeCapacityBound(t, topk, e), n), (n, 1),
      bounds = Slice[float32](a: -2'f32, b: 2'f32))
    weightsHost = makeHostTensor(float32, (t, topk), (topk, 1))
    idsHost = makeHostTensor(int32, (t, topk), (topk, 1))
  var
    idsSeq: seq[int]
    weightsSeq: seq[float32]
  for ti in 0 ..< t:
    for i in 0 ..< topk:
      var gid = -1
      if rng.rand(1.0) >= 0.1:
        gid = rng.rand(0 .. e - 1)
      idsHost[ti, i] = int32(gid)
      weightsHost[ti, i] = to(rng.rand(0.1 .. 1.0), float32)
      idsSeq.add(gid)
      weightsSeq.add(weightsHost[ti, i])
  let refMeta = moeDispatchRef(idsSeq, t, topk, e)
  var slotSeq: seq[int]
  for i in 0 ..< t * topk:
    slotSeq.add(refMeta.slotOfPair[i])
  var slotOfPairStage = makeHostTensor(int32, (t, topk), (topk, 1))
  for i in 0 ..< t * topk:
    slotOfPairStage.data[i] = int32(slotSeq[i])
  let got = launchMoeCombine(engine, t, topk, n, dPackedHost, slotOfPairStage, weightsHost)
  let want = moeCombineRefF64(dPackedHost, slotSeq, weightsSeq, t, topk, n)
  let report = got.to(float64).view().allClose(want.view(),
    rtol = Fp32Rtol, abstol = Fp32Abstol)
  doAssert report.bad == 0,
    msg & ": " & $report.bad & " mismatched elements, worst |d| = " & $report.worst &
    ", first at " & report.firstCrd & ": got " & $report.firstGot &
    " want " & $report.firstWant
  for ti in 0 ..< t:
    var allInvalid = true
    for i in 0 ..< topk:
      if idsSeq[ti * topk + i] >= 0:
        allInvalid = false
    if allInvalid:
      for j in 0 ..< n:
        doAssert to(got[ti, j], float64) == 0.0'f64,
          msg & ": token " & $ti & " has all slots unused, d must store exactly 0"
  echo msg & ": T=" & $t & " topk=" & $topk & " E=" & $e & " N=" & $n

# ═══════════════════════════════════════════════════════════════════════
#  moeForward end to end
# ═══════════════════════════════════════════════════════════════════════

proc moeForwardRefF64[T, LX, LB, LS](x: HostTensor[T, LX], b: HostTensor[T, LB],
    scores: HostTensor[float32, LS], bias: openArray[float32],
    topk: int, variant: MoERoute, scalingFactor: float32): auto =
  ## float64 reference of the WHOLE layer over the same rounded inputs.
  ##
  ## Both tier paths compute the same value in float64. The packed path's
  ## dispatch is an exact integer permutation, so the layer computes
  ## d(t) = sum over ascending i of w(t, i)·(x(t)·b(ids(t, i))),
  ## the weights from the router reference's frozen contract.
  let (ids, weights) = moeRouterRefF64(scores, bias, topk, variant, scalingFactor)
  let
    t = x.layout.shape[0].toInt()
    k = x.layout.shape[1].toInt()
    n = b.layout.shape[2].toInt()
  result = makeHostTensor(float64, (t, n), (n, 1))
  for i in 0 ..< result.len():
    result.data[i] = 0.0'f64
  for ti in 0 ..< t:
    for i in 0 ..< topk:
      let g = ids[ti * topk + i]
      if g >= 0:
        let w = weights[ti * topk + i]
        for j in 0 ..< n:
          var acc = 0.0'f64
          for e in 0 ..< k:
            acc += to(x[ti, e], float64) * to(b[g, e, j], float64)
          result[ti, j] += w * acc

proc runMoeForward[Elem, LD, LX, LS, LBI, LB](engine: MetalEngine,
    dHost: var HostTensor[Elem, LD], xHost: HostTensor[Elem, LX],
    scoresHost: HostTensor[float32, LS], biasHost: HostTensor[float32, LBI],
    bHost: HostTensor[Elem, LB], topk: int, variant: MoERoute, scalingFactor: float32) =
  ## Sentinel-fills d and runs moeForward over the caller's staged inputs.
  for i in 0 ..< dHost.len():
    dHost.data[i] = to(MoERouterSentinel, Elem)
  moeForward(engine, dHost, xHost, scoresHost, biasHost, bHost, topk, variant, scalingFactor)

proc judgeMoeForward[T, LD](got: HostTensor[T, LD], want: HostTensor[float64, LD], msg: string) =
  ## Band judgment of the whole layer's output, every element written.
  const band = when T is float16: (rtol: Fp16Rtol, abstol: Fp16Abstol)
               else: (rtol: Fp32Rtol, abstol: Fp32Abstol)
  let report = got.to(float64).view().allClose(want.view(),
    rtol = band.rtol, abstol = band.abstol)
  doAssert report.bad == 0,
    msg & ": " & $report.bad & " mismatched elements, worst |d| = " & $report.worst &
    ", first at " & report.firstCrd & ": got " & $report.firstGot &
    " want " & $report.firstWant

proc moeForwardCase*[Elem](engine: MetalEngine, _: typedesc[Elem],
    t, topk, e, k, n: int, variant: MoERoute, biasScale, scalingFactor: float32,
    rng: var Rand, msg: string) =
  ## One end-to-end moeForward case, both tier paths exercised by the shapes:
  ## the caller picks shapes whose routed entries per expert fall on either side
  ## of the tier boundary.
  ##
  ## - x and b from `rng` in [-2, 2], scores in [-2, 2], bias from biasScale
  ## - d judged against the float64 whole-layer reference within the element band
  var
    xHost = rng.randomHostTensor(Elem, (t, k), (k, 1),
      bounds = Slice[float32](a: -2'f32, b: 2'f32))
    bHost = rng.randomHostTensor(Elem, (e, k, n),
      (k * n, n, 1), bounds = Slice[float32](a: -2'f32, b: 2'f32))
    scoresHost = rng.randomHostTensor(float32, (t, e), (e, 1),
      bounds = Slice[float32](a: -2'f32, b: 2'f32))
    biasHost = makeHostTensor(float32, (e,), (1,))
    dHost = makeHostTensor(Elem, (t, n), (n, 1))
  for i in 0 ..< e:
    biasHost[i] = to(rng.rand(-biasScale .. biasScale), float32)
  runMoeForward(engine, dHost, xHost, scoresHost, biasHost, bHost, topk, variant, scalingFactor)
  let want = moeForwardRefF64(xHost, bHost, scoresHost,
    biasHost.data.toOpenArray(0, e - 1), topk, variant, scalingFactor)
  judgeMoeForward(dHost, want, msg)
  echo msg & ": T=" & $t & " topk=" & $topk & " E=" & $e & " K=" & $k & " N=" & $n &
    " tier=" & $(if pickGroupedTier((t * topk + e - 1) div e) == gtGemv: "gemv" else: "gemm") &
    " variant=" & $variant

proc moeForwardGraphCaptureCase*[Elem](engine: MetalEngine, _: typedesc[Elem]) =
  ## Graph-capture compatibility over the whole layer: two moeForward runs
  ## over one staged set of shapes with mutated scores, each run judged
  ## against its own float64 reference.
  ##
  ## - the grids derive from the shapes only, both runs launch identical grids
  var rng = initRand(41)
  const
    t = 12
    topk = 4
    e = 6
    k = 24
    n = 32
  var
    xHost = rng.randomHostTensor(Elem, (t, k), (k, 1),
      bounds = Slice[float32](a: -2'f32, b: 2'f32))
    bHost = rng.randomHostTensor(Elem, (e, k, n),
      (k * n, n, 1), bounds = Slice[float32](a: -2'f32, b: 2'f32))
    scoresA = rng.randomHostTensor(float32, (t, e), (e, 1),
      bounds = Slice[float32](a: -2'f32, b: 2'f32))
    scoresB = rng.randomHostTensor(float32, (t, e), (e, 1),
      bounds = Slice[float32](a: -2'f32, b: 2'f32))
    biasHost = makeHostTensor(float32, (e,), (1,))
    dHost = makeHostTensor(Elem, (t, n), (n, 1))
  for i in 0 ..< e:
    biasHost[i] = 0'f32
  runMoeForward(engine, dHost, xHost, scoresA, biasHost, bHost, topk, moeSoftmax, 1.0'f32)
  let wantA = moeForwardRefF64(xHost, bHost, scoresA, biasHost.data.toOpenArray(0, e - 1),
    topk, moeSoftmax, 1.0'f32)
  judgeMoeForward(dHost, wantA, "graph capture, state A")
  runMoeForward(engine, dHost, xHost, scoresB, biasHost, bHost, topk, moeSoftmax, 1.0'f32)
  let wantB = moeForwardRefF64(xHost, bHost, scoresB, biasHost.data.toOpenArray(0, e - 1),
    topk, moeSoftmax, 1.0'f32)
  judgeMoeForward(dHost, wantB, "graph capture, state B")
  echo "graph-capture moeForward: both score states judged over identical shape-derived grids"

# ═══════════════════════════════════════════════════════════════════════
#  Fixture replay
# ═══════════════════════════════════════════════════════════════════════

proc moeRouterReplayCase*(engine: MetalEngine, name: string) =
  ## Replays one router fixture, with relaunch:
  ##
  ## - one launch over the (1, T) grid, ids judged exactly and weights within
  ##   the fp32 band against the recorded reference
  ## - a relaunch judged bit-identical on the raw output bits
  ##
  ## Preconditions: the recorded ids/weights have the frozen numerics contract,
  ## the id stage is pre-filled -2 and the weight stage sentinel-filled
  let dir = MoEFixturesDir / name
  let meta = parseMoEFixtureMetadata(dir)
  doAssert meta.family == "router", meta.name & ": family " & meta.family
  let
    scoresHost = loadFixtureTensor(dir, "scores.bin", (meta.t, meta.e), float32)
    biasHost = loadFixtureTensor(dir, "bias.bin", (meta.e,), float32)
    expIds = loadFixtureTensor(dir, "ids.bin", (meta.t, meta.topk), int32)
    expWeights = loadFixtureTensor(dir, "weights.bin", (meta.t, meta.topk), float32)
  doAssert xorHash(scoresHost) == meta.xorhashChecksum("scores.bin"), meta.name & ": scores checksum"
  doAssert xorHash(biasHost) == meta.xorhashChecksum("bias.bin"), meta.name & ": bias checksum"
  doAssert xorHash(expIds) == meta.xorhashChecksum("ids.bin"), meta.name & ": ids checksum"
  doAssert xorHash(expWeights) == meta.xorhashChecksum("weights.bin"), meta.name & ": weights checksum"
  let (gotIds, gotWeights) = launchMoeRouter(engine, scoresHost, biasHost,
    meta.topk, MoERoute(meta.variant), float32(meta.scaling))
  proc captureBits(): seq[uint32] =
    result = newSeq[uint32](gotIds.len() + gotWeights.len())
    var o = 0
    for i in 0 ..< gotIds.len():
      result[o] = cast[uint32](gotIds.data[i]); o += 1
    for i in 0 ..< gotWeights.len():
      result[o] = cast[uint32](gotWeights.data[i]); o += 1
  let firstBits = captureBits()
  discard launchMoeRouter(engine, scoresHost, biasHost, meta.topk, MoERoute(meta.variant), float32(meta.scaling))
  discard launchMoeRouter(engine, scoresHost, biasHost, meta.topk, MoERoute(meta.variant), float32(meta.scaling))
  let secondBits = captureBits()
  doAssert secondBits == firstBits, meta.name & ": router outputs differ run to run"
  for i in 0 ..< meta.t * meta.topk:
    doAssert int(gotIds.data[i]) == int(expIds.data[i]),
      meta.name & ": ids[" & $i & "] = " & $int(gotIds.data[i]) &
      " want " & $int(expIds.data[i])
  const band = (rtol: Fp32Rtol, abstol: Fp32Abstol)
  let report = gotWeights.to(float64).view().allClose(expWeights.to(float64).view(),
    rtol = band.rtol, abstol = band.abstol)
  doAssert report.bad == 0,
    meta.name & ": " & $report.bad & " mismatched weights, worst |d| = " & $report.worst &
    ", first at " & report.firstCrd
  echo name & ": T=" & $meta.t & " E=" & $meta.e & " topk=" & $meta.topk &
    " variant=" & $MoERoute(meta.variant) & " relaunch bit-identical"

proc moeDispatchReplayCase*(engine: MetalEngine, name: string) =
  ## Replays one dispatch fixture, with relaunch:
  ##
  ## - metadata + gather over the shape-derived static grids, every metadata
  ##   array judged exactly against the recorded integer reference
  ## - packedX judged exactly at the real slots, sentinel-judged at the padding entries
  ## - a relaunch judged byte-identical on the metadata bits
  ##
  ## Preconditions: sortedPairs is staged -1, packedIds/slotOfPair staged -2,
  ## packedX sentinel-filled, and the recorded reference has the kernel's contracts
  let dir = MoEFixturesDir / name
  let meta = parseMoEFixtureMetadata(dir)
  doAssert meta.family == "dispatch", meta.name & ": family " & meta.family
  let
    idsHost = loadFixtureTensor(dir, "ids.bin", (meta.t, meta.topk), int32)
    xHost = loadFixtureTensor(dir, "x.bin", (meta.t, meta.k), float32)
    expIds = loadFixtureTensor(dir, "packedIds.bin", (meta.t * meta.topk + 31 * meta.e,), int32)
    expOffsets = loadFixtureTensor(dir, "packedOffsets.bin", (meta.e,), int32)
    expSorted = loadFixtureTensor(dir, "sortedPairs.bin", (meta.t * meta.topk + 31 * meta.e,), int32)
    expSlotOfPair = loadFixtureTensor(dir, "slotOfPair.bin", (meta.t, meta.topk), int32)
    expPackedX = loadFixtureTensor(dir, "packedX.bin", (meta.t * meta.topk + 31 * meta.e, meta.k), float32)
  doAssert xorHash(idsHost) == meta.xorhashChecksum("ids.bin"), meta.name & ": ids checksum"
  doAssert xorHash(xHost) == meta.xorhashChecksum("x.bin"), meta.name & ": x checksum"
  doAssert xorHash(expIds) == meta.xorhashChecksum("packedIds.bin"), meta.name & ": packedIds checksum"
  doAssert xorHash(expOffsets) == meta.xorhashChecksum("packedOffsets.bin"), meta.name & ": offsets checksum"
  doAssert xorHash(expSorted) == meta.xorhashChecksum("sortedPairs.bin"), meta.name & ": sortedPairs checksum"
  doAssert xorHash(expSlotOfPair) == meta.xorhashChecksum("slotOfPair.bin"), meta.name & ": slotOfPair checksum"
  doAssert xorHash(expPackedX) == meta.xorhashChecksum("packedX.bin"), meta.name & ": packedX checksum"
  let cap = moeCapacityBound(meta.t, meta.topk, meta.e)
  let (offsetsHost, sortedPairsHost, slotOfPairHost, packedIdsHost, packedXHost) =
    launchMoeDispatch(engine, idsHost, xHost, meta.topk, meta.e)
  proc captureBits(): seq[uint32] =
    result = newSeq[uint32](offsetsHost.len() + sortedPairsHost.len() +
      slotOfPairHost.len() + packedIdsHost.len())
    var o = 0
    for i in 0 ..< offsetsHost.len():
      result[o] = cast[uint32](offsetsHost.data[i]); o += 1
    for i in 0 ..< sortedPairsHost.len():
      result[o] = cast[uint32](sortedPairsHost.data[i]); o += 1
    for i in 0 ..< slotOfPairHost.len():
      result[o] = cast[uint32](slotOfPairHost.data[i]); o += 1
    for i in 0 ..< packedIdsHost.len():
      result[o] = cast[uint32](packedIdsHost.data[i]); o += 1
  let firstBits = captureBits()
  discard launchMoeDispatch(engine, idsHost, xHost, meta.topk, meta.e)
  let secondBits = captureBits()
  doAssert secondBits == firstBits, name & ": metadata differs run to run"
  for g in 0 ..< meta.e:
    doAssert int(offsetsHost[g]) == int(expOffsets[g]),
      name & ": packedOffsets[" & $g & "] = " & $int(offsetsHost[g]) &
      " want " & $int(expOffsets[g])
  for i in 0 ..< meta.t * meta.topk:
    doAssert int(slotOfPairHost.data[i]) == int(expSlotOfPair.data[i]),
      name & ": slotOfPair[" & $i & "] = " & $int(slotOfPairHost.data[i]) &
      " want " & $int(expSlotOfPair.data[i])
  for s in 0 ..< cap:
    doAssert int(sortedPairsHost[s]) == int(expSorted[s]),
      name & ": sortedPairs[" & $s & "] = " & $int(sortedPairsHost[s]) &
      " want " & $int(expSorted[s])
    doAssert int(packedIdsHost[s]) == int(expIds[s]),
      name & ": packedIds[" & $s & "] = " & $int(packedIdsHost[s]) &
      " want " & $int(expIds[s])
    if int(expSorted[s]) >= 0:
      for kk in 0 ..< meta.k:
        doAssert to(packedXHost[s, kk], float64) == to(expPackedX[s, kk], float64),
          name & ": packedX[" & $s & ", " & $kk & "] mismatch"
    else:
      for kk in 0 ..< meta.k:
        doAssert to(packedXHost[s, kk], float64) == to(MoERouterSentinel, float64),
          name & ": padding row " & $s & " was written"
  echo name & ": T=" & $meta.t & " topk=" & $meta.topk & " E=" & $meta.e &
    " K=" & $meta.k & " cap=" & $cap & " relaunch byte-identical"

proc moeForwardReplayCase*(engine: MetalEngine, Elem: typedesc, name: string) =
  ## Replays one end-to-end fixture through moeForward, with relaunch:
  ##
  ## - d judged against the recorded float32 expected within the element band
  ## - a relaunch judged bit-identical on the raw output bits
  ##
  ## Preconditions: the recorded d has the whole-layer float64 reference contract,
  ## the destination staging is sentinel-filled
  const wantDtype = when Elem is float16: "float16" else: "float32"
  let dir = MoEFixturesDir / name
  let meta = parseMoEFixtureMetadata(dir)
  doAssert meta.family == "e2e", meta.name & ": family " & meta.family
  doAssert meta.dtype == wantDtype, meta.name & ": dtype " & meta.dtype
  let
    xBits = loadFixtureTensor(dir, "x.bin", (meta.t, meta.k),
      when Elem is float16: uint16 else: uint32)
    bBits = loadFixtureTensor(dir, "b.bin", (meta.e, meta.k, meta.n),
      when Elem is float16: uint16 else: uint32)
    scoresHost = loadFixtureTensor(dir, "scores.bin", (meta.t, meta.e), float32)
    biasHost = loadFixtureTensor(dir, "bias.bin", (meta.e,), float32)
    expected = loadFixtureTensor(dir, "d.bin", (meta.t, meta.n), float32)
  doAssert xorHash(xBits) == meta.xorhashChecksum("x.bin"), meta.name & ": x checksum"
  doAssert xorHash(bBits) == meta.xorhashChecksum("b.bin"), meta.name & ": b checksum"
  doAssert xorHash(scoresHost) == meta.xorhashChecksum("scores.bin"), meta.name & ": scores checksum"
  doAssert xorHash(biasHost) == meta.xorhashChecksum("bias.bin"), meta.name & ": bias checksum"
  doAssert xorHash(expected) == meta.xorhashChecksum("d.bin"), meta.name & ": d checksum"
  var
    xStage = makeHostTensor(Elem, (meta.t, meta.k), LayoutRight)
    bStage = makeHostTensor(Elem, (meta.e, meta.k, meta.n), LayoutRight)
    dStage = makeHostTensor(Elem, (meta.t, meta.n), LayoutRight)
  for i in 0 ..< xStage.len():
    xStage.data[i] = cast[Elem](xBits.data[i])
  for i in 0 ..< bStage.len():
    bStage.data[i] = cast[Elem](bBits.data[i])
  proc captureBits(): seq[uint32] =
    result = newSeq[uint32](dStage.len())
    for i in 0 ..< dStage.len():
      when Elem is float16:
        result[i] = uint32(cast[uint16](dStage.data[i]))
      else:
        result[i] = cast[uint32](dStage.data[i])
  runMoeForward(engine, dStage, xStage, scoresHost, biasHost, bStage,
    meta.topk, MoERoute(meta.variant), float32(meta.scaling))
  let firstBits = captureBits()
  runMoeForward(engine, dStage, xStage, scoresHost, biasHost, bStage,
    meta.topk, MoERoute(meta.variant), float32(meta.scaling))
  let secondBits = captureBits()
  for i in 0 ..< dStage.len():
    doAssert secondBits[i] == firstBits[i],
      name & ": d differs run to run at element " & $i
  const band = when Elem is float16: (rtol: Fp16Rtol, abstol: Fp16Abstol)
               else: (rtol: Fp32Rtol, abstol: Fp32Abstol)
  let report = dStage.to(float64).view().allClose(expected.to(float64).view(),
    rtol = band.rtol, abstol = band.abstol)
  doAssert report.bad == 0,
    name & ": " & $report.bad & " mismatched elements, worst |d| = " & $report.worst &
    ", first at " & report.firstCrd & ": got " & $report.firstGot &
    " want " & $report.firstWant
  echo name & ": T=" & $meta.t & " topk=" & $meta.topk & " E=" & $meta.e &
    " K=" & $meta.k & " N=" & $meta.n & " relaunch bit-identical"

# ═══════════════════════════════════════════════════════════════════════
#  Analytic fuzz
# ═══════════════════════════════════════════════════════════════════════

proc moeRouterFuzz*(engine: MetalEngine) =
  ## Seeded router fuzz, float64 reference, seed printed.
  ##
  ## - 8 random cases, T in [1, 8], E in [1, 32], topk in [1, 6]
  ## - the variant alternates softmax and sigmoid, biasScale alternates 0 and 0.5,
  ##   scalingFactor alternates 1.0 and 2.5
  var rng = initRand(FuzzSeed)
  echo "fuzz seed: ", FuzzSeed
  for i in 0 ..< 8:
    let
      tok = rng.rand(1 .. 8)
      e = rng.rand(1 .. 32)
      topk = rng.rand(1 .. 6)
      variant = if i mod 2 == 0: moeSoftmax else: moeSigmoid
      biasScale = if i mod 3 == 0: 0.0'f32 else: 0.5'f32
      scaling = if i mod 4 == 0: 1.0'f32 else: 2.5'f32
    moeRouterCase(engine, tok, e, topk, variant, biasScale, scaling, rng,
      "router fuzz case " & $i)

proc moeDispatchFuzz*(engine: MetalEngine) =
  ## Seeded dispatch fuzz, integer-exact reference, seed printed.
  ##
  ## - 8 random cases, T in [1, 8], topk in [1, 6], E in [1, 16], K in [8, 70]
  ## - about one slot id in ten forced to -1, expert 0 forced empty when E >= 2,
  ##   one token forced to repeat one expert in all its slots
  var rng = initRand(FuzzSeed + 1)
  for i in 0 ..< 8:
    let
      tok = rng.rand(1 .. 8)
      topk = rng.rand(1 .. 6)
      e = rng.rand(1 .. 16)
      k = rng.rand(8 .. 70)
    var ids: seq[seq[int]]
    for ti in 0 ..< tok:
      var row: seq[int]
      for ip in 0 ..< topk:
        var gid = -1
        if rng.rand(1.0) >= 0.1:
          gid = rng.rand(0 .. e - 1)
        row.add(gid)
      ids.add(row)
    if e >= 2:
      for ip in 0 ..< topk:
        ids[0][ip] = rng.rand(1 .. e - 1)
    let rep = rng.rand(0 .. e - 1)
    if tok >= 2:
      for ip in 0 ..< topk:
        ids[1][ip] = rep
    moeDispatchCase(engine, ids, e, k, rng, "dispatch fuzz case " & $i)

proc moeForwardFuzz*(engine: MetalEngine) =
  ## Seeded end-to-end fuzz, float64 whole-layer reference, seed printed.
  ##
  ## - 8 random fp32 cases, T in [1, 64], E in [1, 64], topk in [1, 6],
  ##   K and N in [8, 70], tier chosen by the shapes' routed entries per expert
  var rng = initRand(FuzzSeed + 2)
  for i in 0 ..< 8:
    let
      tok = rng.rand(1 .. 64)
      topk = rng.rand(1 .. 6)
      e = rng.rand(1 .. 64)
      k = rng.rand(8 .. 70)
      n = rng.rand(8 .. 70)
      variant = if i mod 2 == 0: moeSoftmax else: moeSigmoid
      biasScale = if i mod 3 == 0: 0.0'f32 else: 0.5'f32
      scaling = if i mod 4 == 0: 1.0'f32 else: 2.5'f32
    moeForwardCase(engine, float32, tok, topk, e, k, n, variant, biasScale, scaling,
      rng, "e2e fuzz case " & $i)
