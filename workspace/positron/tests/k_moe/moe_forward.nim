## moeForward: the MoE layer driver, routing -> tier choice -> kernels -> d,
## one call per layer per step, the caller's engine drives the launches,
## kernel modules never see one.
##
## - x: (T, K), expertB: (E, K, N), d: (T, N), one element dtype per call,
##   fp32 router logits (T, E), bias: (E,) fp32 zero-filled for biasless models
## - topk <= MoERouterMaxTopk, E <= MoEMaxExperts, host-asserted
## - tier choice: pickGroupedTier over the routed entries per expert, forceTier
##   overrides it for measurement (GroupedTier ordinals, -1 auto), the gemv tier
##   reads the router's ids and weights directly, the gemm tier packs over
##   moeCapacityBound(T, topk, E), runs grouped_gemm_via_expertIds and un-permutes with moe_combine
## - every launch is static-grid from the shapes, no device-to-host read anywhere

import std/[math]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/positron/src/kernels/k_grouped_gemv_gemm {.all.}
import workspace/positron/src/kernels/k_moe_forward {.all.}
import workspace/positron/src/kernels/k_moe_router {.all.}
import ../positron_testutils

export int_tuples, layout_algebra, tensors, tile_algebra
export k_grouped_gemv_gemm, k_moe_forward, k_moe_router

proc moeForward*[Elem, LD, LX, LS, LBI, LB](engine: MetalEngine,
    dHost: var HostTensor[Elem, LD], xHost: HostTensor[Elem, LX],
    scoresHost: HostTensor[float32, LS], biasHost: HostTensor[float32, LBI],
    bHost: HostTensor[Elem, LB], topk: int, variant: MoERoute, scalingFactor: float32,
    forceTier: int = -1) =
  ## Runs the whole MoE layer over host-staged inputs, writing d in place.
  ##
  ## Usage:
  ##   var d = makeHostTensor(Elem, (t, n), LayoutRight)
  ##   moeForward(engine, d, x, scores, bias, b, 6, moeSoftmax, 1.0'f32)
  ##
  ## d is sentinel-filled by the caller: every element is written on every path,
  ## an unwritten element fails the caller's judgment.
  template makeDevView(dev, sh, st: untyped): untyped =
    let lay = make_layout(sh, st)
    BridgeTensor[typeof(dev), typeof(lay)](data: dev, layout: lay)
  let
    t = xHost.layout.shape[0].toInt()
    k = xHost.layout.shape[1].toInt()
    e = bHost.layout.shape[0].toInt()
    n = dHost.layout.shape[1].toInt()
  doAssert topk <= MoERouterMaxTopk, "topk above the router's insertion buffer"
  doAssert e <= MoEMaxExperts, "E above the dispatch histogram"
  doAssert e >= 1 and t >= 1
  let cap = moeCapacityBound(t, topk, e)
  # Bridged inputs are already TensorView-shaped bridge values.
  # Each workspace is a DeviceMem slab the template wraps in a runtime-layout view
  let
    xView = engine.bridge(xHost)
    scoresView = engine.bridge(scoresHost)
    biasView = engine.bridge(biasHost)
    bView = engine.bridge(bHost)
    dView = engine.bridge(dHost)
    idsDev = engine.allocDevice(int32, t * topk)
    weightsDev = engine.allocDevice(float32, t * topk)
  let
    idsView = makeDevView(idsDev, (t, topk), (topk, 1))
    weightsView = makeDevView(weightsDev, (t, topk), (topk, 1))
  engine.launchDeviceFn moe_router_topk(idsView, weightsView, scoresView, biasView,
      topk, ord(variant), scalingFactor) do:
    grid = (1, t)
    blk = (32, 1)
  # Each bench cell forces its tier, -1 picks by the routed entries per expert
  let rowsPerExpert = (t * topk + e - 1) div e
  let tier = if forceTier < 0: pickGroupedTier(rowsPerExpert) else: GroupedTier(forceTier)
  if tier == gtGemv:
    # Gemv tier: reads the router's (T, topk) form directly, no pack
    let (form, kSplits) = groupedGemvForm(t, n)
    if form == ggfSplitK:
      let
        pDev = engine.allocDevice(float32, t * kSplits * n)
        pView = makeDevView(pDev, (t, kSplits, n),
                            (kSplits * n, n, 1))
      engine.launchDeviceFn grouped_gemv_splitk(pView, xView, bView, idsView,
          weightsView, topk, kSplits) do:
        grid = (n.ceil_div(32 * GroupedGemvOutputs) * kSplits, t)
        blk = (32, 1)
      engine.launchDeviceFn grouped_gemv_reduce(dView, pView, kSplits) do:
        grid = (n.ceil_div(32 * GroupedGemvOutputs), t)
        blk = (32, 1)
    else:
      engine.launchDeviceFn grouped_gemv_via_expertIds(dView, xView, bView, idsView, weightsView, topk) do:
        grid = (n.ceil_div(32 * GroupedGemvOutputs), t)
        blk = (32, 1)
  else:
    # Gemm tier: packs the routed entries over the capacity bound, and the grid never reads device data back
    let
      offsetsDev = engine.allocDevice(int32, e)
      sortedPairsDev = engine.allocDevice(int32, cap)
      slotOfPairDev = engine.allocDevice(int32, t * topk)
      packedIdsDev = engine.allocDevice(int32, cap)
      packedXDev = engine.allocDevice(Elem, cap * k)
      dPackedDev = engine.allocDevice(Elem, cap * n)
    var sortedPairsStage = newSeq[int32](cap)
    for i in 0 ..< cap:
      sortedPairsStage[i] = -1
    engine.copyToDeviceFromHost(sortedPairsDev, sortedPairsStage)
    let
      offsetsView = makeDevView(offsetsDev, (e,), (1,))
      sortedPairsView = makeDevView(sortedPairsDev, (cap,), (1,))
      slotOfPairView = makeDevView(slotOfPairDev, (t, topk), (topk, 1))
      packedIdsView = makeDevView(packedIdsDev, (cap,), (1,))
      packedXView = makeDevView(packedXDev, (cap, k), (k, 1))
      dPackedView = makeDevView(dPackedDev, (cap, n), (n, 1))
    engine.launchDeviceFn moe_dispatch_metadata(offsetsView, sortedPairsView,
        slotOfPairView, idsView, topk) do:
      grid = (t.ceil_div(32), 1)
      blk = (32, 1)
    engine.launchDeviceFn moe_dispatch_gather(packedIdsView, packedXView, xView,
        sortedPairsView, offsetsView) do:
      grid = ((cap * k).ceil_div(32), 1)
      blk = (32, 1)
    engine.launchDeviceFn grouped_gemm_via_expertIds(dPackedView, packedXView, bView, packedIdsView) do:
      grid = (countGroupedGemmTiles(Elem, cap, n), 1)
      blk = (32, 1)
    engine.launchDeviceFn moe_combine(dView, dPackedView, slotOfPairView, weightsView, topk) do:
      grid = (n.ceil_div(32 * MoECombineOutputs), t)
      blk = (32, 1)
  engine.copyToHostFromDevice(dHost.data.toOpenArray(0, dHost.len() - 1), dView.data)
