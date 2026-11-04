# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal MoE router suite, the tile-stack kernels judged against f64 host references
##
## Test that the tile-stack MoE router matches its f64 references end to end:
## the Qwen softmax routing chain (ids lowest-index on ties, weights per bar),
## the GLM sigmoid chain (ragged runtime eCount, RouteSumEps normalization),
## the shared-gate weight logit and the bit-exact fp32-partial merge.
##
## Run:
##   nim c -r --outdir:build/tests --nimcache:nimcache/moe_router --hints:off workspace/positron/tests/ceramic/t_moe_router_metal.nim

import std/[math, random, strformat, strutils]
import workspace/crucible
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/tests/ceramic_testutils
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_router
import moe_ref_utils

# ─── Device entries, one per (element dtype, shape) binding ────────────

proc routeBf16Mega(ids, rout_w, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 2048, 256, 8, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc routeF16Mega(ids, rout_w, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[float16, 2048, 256, 8, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc routeBf16Small(ids, rout_w, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 256, 64, 4, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc routeBf16MegaS2(ids, rout_w, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 2048, 256, 8, 2.0'f32](ids, rout_w, x, router_w, scratch)

proc routeBf16Probe(ids, rout_w, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 16, 64, 4, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc sharedGateBf16(outp, x, sgw: distinct TensorView) =
  let t = int(threadgroup_position_in_grid.x)
  let v = sharedGateLogit[bfloat16, 2048](x, sgw, int32(t))
  if thread_index_in_threadgroup == 0:
    outp[t, 0] = v

proc sharedGateF16(outp, x, sgw: distinct TensorView) =
  let t = int(threadgroup_position_in_grid.x)
  let v = sharedGateLogit[float16, 2048](x, sgw, int32(t))
  if thread_index_in_threadgroup == 0:
    outp[t, 0] = v

proc sigmoidBf16(ids, wOut, x, router_w, scratch: distinct TensorView, eCount: int32, scale: float32) =
  let t = int(threadgroup_position_in_grid.x)
  var idsReg: array[8, int32]
  var wReg: array[8, float32]
  let sRow = scratch(t, _)
  moeRouteSigmoid[bfloat16, 2048, 8](x, router_w, int32(t), sRow, eCount,
    scale, idsReg, wReg)
  for slot in 0 ..< 8:
    ids[t, slot] = idsReg[slot]
    wOut[t, slot] = wReg[slot]

proc mergeBf16Small(out_r, partial: distinct TensorView) =
  moe_decode_merge_at[bfloat16, 256, 8](out_r, partial,
    int32(threadgroup_position_in_grid.x),
    int32(threadgroup_position_in_grid.y))

proc mergeF16Small(out_r, partial: distinct TensorView) =
  moe_decode_merge_at[float16, 256, 8](out_r, partial,
    int32(threadgroup_position_in_grid.x),
    int32(threadgroup_position_in_grid.y))

proc mergeBf16Mega(out_r, partial: distinct TensorView) =
  moe_decode_merge_at[bfloat16, 2048, 8](out_r, partial,
    int32(threadgroup_position_in_grid.x),
    int32(threadgroup_position_in_grid.y))

# Weights-first twins: the launch machinery consumes the non-output
# bridge arguments (move-only DeviceMem), so a tensor the host reads back
# must be its launch's first argument.

proc routeBf16MegaW(rout_w, ids, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 2048, 256, 8, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc routeF16MegaW(rout_w, ids, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[float16, 2048, 256, 8, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc routeBf16SmallW(rout_w, ids, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 256, 64, 4, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc routeBf16MegaS2W(rout_w, ids, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 2048, 256, 8, 2.0'f32](ids, rout_w, x, router_w, scratch)

proc routeBf16ProbeW(rout_w, ids, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 16, 64, 4, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc sigmoidBf16W(wOut, ids, x, router_w, scratch: distinct TensorView, eCount: int32, scale: float32) =
  let t = int(threadgroup_position_in_grid.x)
  var idsReg: array[8, int32]
  var wReg: array[8, float32]
  let sRow = scratch(t, _)
  moeRouteSigmoid[bfloat16, 2048, 8](x, router_w, int32(t), sRow, eCount,
    scale, idsReg, wReg)
  for slot in 0 ..< 8:
    ids[t, slot] = idsReg[slot]
    wOut[t, slot] = wReg[slot]

template routeRound(engine, LauncherIds, LauncherW, idsH, wH, xH, rwH, sH: untyped, gridX: int) =
  ## One judgment round over the bridged router tensors, two launches.
  ## The ids launch runs first, then the weights launch. The launch machinery
  ## consumes the non-output bridge arguments (move-only DeviceMem),
  ## so a tensor the host reads back must be its launch's first argument.
  ##
  ## The weights launch rewrites the already-copied ids with the same deterministic values.
  block:
    let idsDev = engine.bridge(idsH)
    let wDev = engine.bridge(wH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let sDev = engine.bridge(sH)
    engine.launchDeviceFn LauncherIds(idsDev, wDev, xDev, rwDev, sDev) do:
      grid = (gridX, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr idsH.data[0], idsDev.data, idsH.len())
  block:
    let idsDev = engine.bridge(idsH)
    let wDev = engine.bridge(wH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let sDev = engine.bridge(sH)
    engine.launchDeviceFn LauncherW(wDev, idsDev, xDev, rwDev, sDev) do:
      grid = (gridX, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr wH.data[0], wDev.data, wH.len())

template sigmoidRound(engine, LauncherIds, LauncherW, idsH, wH, xH, rwH, sH: untyped, gridX: int, eCount: int32, scale: float32) =
  ## One sigmoid judgment round, the same two-launch shape as `routeRound`
  ## with the sigmoid launcher's runtime expert count and scale.
  block:
    let idsDev = engine.bridge(idsH)
    let wDev = engine.bridge(wH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let sDev = engine.bridge(sH)
    engine.launchDeviceFn LauncherIds(idsDev, wDev, xDev, rwDev, sDev,
        eCount, scale) do:
      grid = (gridX, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr idsH.data[0], idsDev.data, idsH.len())
  block:
    let idsDev = engine.bridge(idsH)
    let wDev = engine.bridge(wH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let sDev = engine.bridge(sH)
    engine.launchDeviceFn LauncherW(wDev, idsDev, xDev, rwDev, sDev,
        eCount, scale) do:
      grid = (gridX, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr wH.data[0], wDev.data, wH.len())

# ─── The Qwen softmax form ────────────────────────────────────────────

proc dotAbsSum[El, TSh, TSt, WSh, WSt](
    xH: HostTensor[El, TSh, TSt], wH: HostTensor[El, WSh, WSt],
    H: int, t, e: int): float64 =
  sumAbsF64[El](xH.data, wH.data, H, t * H, e * H)

template runRouteCombo(engine, Launcher, LauncherW: untyped, El: typedesc, T, H, E, K: static int,
    Scale: static float32, cases: int, seed: uint64, label: string) =
  ## One (element dtype, shape, Scale) combination over `cases` independent
  ## seeded runs. The judgment has two parts:
  ##
  ## - ids against the f64 top-K under the tie region: a swapped pair sits
  ##   inside the pair's reassociation and El grid slack bound.
  ## - weights per element under the fp32 bar.
  ##
  ## Case 0 relaunches bit-identical.
  let uStep = uStepF64(El)
  var idsH = makeHostTensor(int32, (T, K), (K, 1))
  var wH = makeHostTensor(El, (T, K), (K, 1))
  var xH = makeHostTensor(El, (T, H), (H, 1))
  var rwH = makeHostTensor(El, (E, H), (H, 1))
  var sH = makeHostTensor(float32, (T, E), (E, 1))
  var rng = initRand(int64(seed))
  var worstUse = 0.0'f64
  var exactW = 0
  var total = 0
  var totalSwaps = 0
  var case0ids = newSeq[int32](T * K)
  var case0w = newSeq[El](T * K)
  var case0x = newSeq[El](T * H)
  var case0rw = newSeq[El](E * H)
  for caseId in 0 ..< cases:
    for i in 0 ..< T * H:
      xH.data[i] = rng.rand(-1.0'f32 .. 1.0'f32).to(El)
    for i in 0 ..< E * H:
      rwH.data[i] = rng.rand(-1.0'f32 .. 1.0'f32).to(El)
    for i in 0 ..< T * K:
      idsH.data[i] = -1
      wH.data[i] = zero(El)
    var xBits = newSeq[El](T * H)
    var rwBits = newSeq[El](E * H)
    for i in 0 ..< T * H:
      xBits[i] = xH.data[i]
    for i in 0 ..< E * H:
      rwBits[i] = rwH.data[i]
    let rawLogits = rawLogitsF64[El](xH.data, rwH.data, T, H, E)
    var wantLogits = newSeq[float64](T * E)
    for i in 0 ..< T * E:
      wantLogits[i] = elRoundF64[El](rawLogits[i])
    var wantIds = newSeq[int32](T * K)
    var wantW = newSeq[El](T * K)
    for tok in 0 ..< T:
      let p = softmaxF64(wantLogits[tok * E ..< (tok + 1) * E], E)
      let sel = topKF64(p, E, K)
      let w64 = renormWeightsF64(p, sel, K, Scale)
      for slot in 0 ..< K:
        wantIds[tok * K + slot] = sel[slot]
        wantW[tok * K + slot] = narrowEl[El](w64[slot])
    routeRound(engine, Launcher, LauncherW, idsH, wH, xH, rwH, sH, T)
    # Inputs must come back unchanged. The kernel reads them only.
    for i in 0 ..< T * H:
      doAssert widenF64(xH.data[i]) == widenF64(xBits[i]), "input x modified"
    for i in 0 ..< E * H:
      doAssert widenF64(rwH.data[i]) == widenF64(rwBits[i]), "router weight modified"
    for tok in 0 ..< T:
      var prev = -float64(finiteFloor(float32))
      for slot in 0 ..< K:
        let idx = tok * K + slot
        doAssert idsH.data[idx] >= 0 and idsH.data[idx] < int32(E),
          "expert id out of range at (tok " & $tok & ", slot " & $slot & ", case " & $caseId & ")"
        doAssert slot == 0 or idsH.data[idx] != idsH.data[idx - 1],
          "duplicate expert id at (tok " & $tok & ", slot " & $slot & ", case " & $caseId & ")"
        let gE = idsH.data[idx].int
        let wE = wantIds[idx].int
        let gotW = widenF64(wH.data[idx])
        doAssert gotW <= prev, "routing weights not descending"
        prev = gotW
        if gE != wE:
          inc totalSwaps
          let lgA = wantLogits[tok * E + gE]
          let lgB = wantLogits[tok * E + wE]
          doAssert abs(lgA - lgB) <= pairTieBound(uStep, H, dotAbsSum(xH, rwH, H, tok, gE),
              dotAbsSum(xH, rwH, H, tok, wE), lgA, lgB),
            "top-K swap at (tok " & $tok & ", slot " & $slot &
            ", case " & $caseId & ") outside the tie region: experts " &
            $gE & " and " & $wE
        let wantWv = widenF64(wantW[idx])
        let bar = weightBar(uStep, wantWv, E, K)
        let diff = abs(gotW - wantWv)
        doAssert diff <= bar,
          "routing weight outside the bar at (tok " & $tok & ", slot " &
          $slot & ", case " & $caseId & "): " & $diff & " > " & $bar
        worstUse = max(worstUse, diff / bar)
        if widenF64(wH.data[idx]) == widenF64(wantW[idx]):
          inc exactW
        inc total
    if caseId == 0:
      for i in 0 ..< T * K:
        case0ids[i] = idsH.data[i]
        case0w[i] = wH.data[i]
      for i in 0 ..< T * H:
        case0x[i] = xH.data[i]
      for i in 0 ..< E * H:
        case0rw[i] = rwH.data[i]
  block determinism:
    for i in 0 ..< T * H:
      xH.data[i] = case0x[i]
    for i in 0 ..< E * H:
      rwH.data[i] = case0rw[i]
    for i in 0 ..< T * K:
      idsH.data[i] = -1
      wH.data[i] = zero(El)
    routeRound(engine, Launcher, LauncherW, idsH, wH, xH, rwH, sH, T)
    for i in 0 ..< T * K:
      doAssert idsH.data[i] == case0ids[i], "ids differ run to run"
      doAssert widenF64(wH.data[i]) == widenF64(case0w[i]),
        "routing weights differ run to run"
  echo "[" & label & "] cases=" & $cases &
    " worst bar usage " & formatFloat(worstUse, ffScientific, 3) &
    ", bit-exact " & $exactW & "/" & $total &
    ", tie-region swaps " & $totalSwaps

# ─── The expert-index probe ───────────────────────────────────────────

proc runProbe[E: HwEngine](engine: E) =
  ## Router GEMV fragment-map test: x all ones, expert e's router weight
  ## vector all e/16, so expert e's logit is e exactly in fp32 mma arithmetic.
  ## The softmax ordering then forces ids [63, 62, 61, 60]:
  ## any mis-mapped accumulator cell shows up as a wrong expert id.
  const
    T = 1
    H = 16
    E = 64
    K = 4
  var idsH = makeHostTensor(int32, (T, K), (K, 1))
  var wH = makeHostTensor(bfloat16, (T, K), (K, 1))
  var xH = makeHostTensor(bfloat16, (T, H), (H, 1))
  var rwH = makeHostTensor(bfloat16, (E, H), (H, 1))
  var sH = makeHostTensor(float32, (T, E), (E, 1))
  for k in 0 ..< H:
    xH.data[k] = 1.0'f32.to(bfloat16)
  for e in 0 ..< E:
    for k in 0 ..< H:
      rwH.data[e * H + k] = (e.float32 / 16.0'f32).to(bfloat16)
  for i in 0 ..< T * K:
    idsH.data[i] = -1
    wH.data[i] = zero(bfloat16)
  let idsDev = engine.bridge(idsH)
  let wDev = engine.bridge(wH)
  let xDev = engine.bridge(xH)
  let rwDev = engine.bridge(rwH)
  let sDev = engine.bridge(sH)
  routeRound(engine, routeBf16Probe, routeBf16ProbeW, idsH, wH, xH, rwH, sH, T)
  let wantIds = [63, 62, 61, 60]
  for slot in 0 ..< K:
    doAssert idsH.data[slot] == int32(wantIds[slot]),
      &"fragment-map probe: slot {slot} routed to expert {idsH.data[slot]}, " &
      &"want {wantIds[slot]}"
  # Routing weights against the same f64 chain the combos use.
  let rawLogits = rawLogitsF64[bfloat16](xH.data, rwH.data, T, H, E)
  var wantLogits = newSeq[float64](E)
  for e in 0 ..< E:
    wantLogits[e] = elRoundF64[bfloat16](rawLogits[e])
  let p = softmaxF64(wantLogits, E)
  let sel = topKF64(p, E, K)
  let w64 = renormWeightsF64(p, sel, K, 1.0'f32)
  for slot in 0 ..< K:
    let gotW = widenF64(wH.data[slot])
    let wantWv = widenF64(narrowEl[bfloat16](w64[slot]))
    let bar = weightBar(uStepF64(bfloat16), wantWv, E, K)
    doAssert abs(gotW - wantWv) <= bar,
      &"fragment-map probe weight outside the bar at slot {slot}: " &
      &"{abs(gotW - wantWv):.3e} > {bar:.3e}"
  echo "[fragment-map probe bf16] ids [63, 62, 61, 60], weights inside the bar"

# ─── The GLM sigmoid form ─────────────────────────────────────────────

template runSigmoidCombo(engine, Launcher, LauncherW: untyped, El: typedesc, T, H, E, K: static int,
    eCount: int, scale: float32, cases: int, seed: uint64, label: string) =
  ## One sigmoid combination over `cases` seeded runs. The judgment has two parts:
  ##
  ## - ids against the f64 top-K over the real experts. A swap sits inside
  ##   the fp32 sigmoid scores' tie bound, whose wide end is the saturated
  ##   scores collapsing onto 1.0.
  ## - weights judged in fp32 against the f64 reference recomputed over
  ##   the kernel's own id set.
  ##
  ## Case 0 relaunches bit-identical.
  var idsH = makeHostTensor(int32, (T, K), (K, 1))
  var wH = makeHostTensor(float32, (T, K), (K, 1))
  var xH = makeHostTensor(El, (T, H), (H, 1))
  var rwH = makeHostTensor(El, (E, H), (H, 1))
  var sH = makeHostTensor(float32, (T, E), (E, 1))
  var rng = initRand(int64(seed))
  var worstUse = 0.0'f64
  var case0ids = newSeq[int32](T * K)
  var case0w = newSeq[float32](T * K)
  var case0x = newSeq[El](T * H)
  var case0rw = newSeq[El](E * H)
  for caseId in 0 ..< cases:
    for i in 0 ..< T * H:
      xH.data[i] = rng.rand(-1.0'f32 .. 1.0'f32).to(El)
    for i in 0 ..< E * H:
      rwH.data[i] = rng.rand(-0.5'f32 .. 0.5'f32).to(El)
    for i in 0 ..< T * K:
      idsH.data[i] = -1
      wH.data[i] = -1.0'f32
    var xBits = newSeq[El](T * H)
    var rwBits = newSeq[El](E * H)
    for i in 0 ..< T * H:
      xBits[i] = xH.data[i]
    for i in 0 ..< E * H:
      rwBits[i] = rwH.data[i]
    let rawLogits = rawLogitsF64[El](xH.data, rwH.data, T, H, E)
    var wantIds = newSeq[int32](T * K)
    var wantW = newSeq[float64](T * K)
    for tok in 0 ..< T:
      let s = sigmoidScoresF64(rawLogits[tok * E ..< (tok + 1) * E], eCount)
      let sel = topKF64(s[0 ..< eCount], eCount, K)
      let w64 = sigmoidWeightsF64(s, sel, K, scale)
      for slot in 0 ..< K:
        wantIds[tok * K + slot] = sel[slot]
        wantW[tok * K + slot] = w64[slot]
    sigmoidRound(engine, Launcher, LauncherW, idsH, wH, xH, rwH, sH, T,
      int32(eCount), scale)
    for i in 0 ..< T * H:
      doAssert widenF64(xH.data[i]) == widenF64(xBits[i]), "input x modified"
    for i in 0 ..< E * H:
      doAssert widenF64(rwH.data[i]) == widenF64(rwBits[i]), "router weight modified"
    for tok in 0 ..< T:
      let sRef = sigmoidScoresF64(rawLogits[tok * E ..< (tok + 1) * E], eCount)
      var gotRow = newSeq[int32](K)
      for slot in 0 ..< K:
        gotRow[slot] = idsH.data[tok * K + slot]
      var slopeDeltaSum = 0.0'f64
      for slot in 0 ..< K:
        slopeDeltaSum += reassocDelta(H, dotAbsSum(xH, rwH, H, tok, gotRow[slot].int))
      for slot in 0 ..< K:
        let idx = tok * K + slot
        let gE = idsH.data[idx].int
        doAssert gE >= 0 and gE < eCount,
          "sigmoid id " & $gE & " outside [0, " & $eCount & ") at (tok " &
          $tok & ", slot " & $slot & ", case " & $caseId & ")"
        doAssert slot == 0 or idsH.data[idx] != idsH.data[idx - 1],
          "duplicate sigmoid id at (tok " & $tok & ", slot " & $slot & ", case " & $caseId & ")"
        if gE != wantIds[idx].int:
          let sA = sRef[gE]
          let sB = sRef[wantIds[idx].int]
          doAssert abs(sA - sB) <= sigmoidScoreTieBound(
              reassocDelta(H, dotAbsSum(xH, rwH, H, tok, gE)),
              reassocDelta(H, dotAbsSum(xH, rwH, H, tok, wantIds[idx].int))),
            "sigmoid top-K swap at (tok " & $tok & ", slot " & $slot &
            ", case " & $caseId & ") outside the tie region: experts " &
            $gE & " and " & $wantIds[idx]
        let gotW = wH.data[idx].float64
        let wRef = sigmoidWeightsF64(sRef, gotRow, K, scale)[slot]
        let bar = sigmoidWeightBar(gotW, K, slopeDeltaSum,
          reassocDelta(H, dotAbsSum(xH, rwH, H, tok, gE)))
        let diff = abs(gotW - wRef)
        doAssert diff <= bar,
          "sigmoid weight outside the bar at (tok " & $tok & ", slot " &
          $slot & ", case " & $caseId & "): " & $diff & " > " & $bar
        worstUse = max(worstUse, diff / bar)
    if caseId == 0:
      for i in 0 ..< T * K:
        case0ids[i] = idsH.data[i]
        case0w[i] = wH.data[i]
      for i in 0 ..< T * H:
        case0x[i] = xH.data[i]
      for i in 0 ..< E * H:
        case0rw[i] = rwH.data[i]
  block determinism:
    for i in 0 ..< T * H:
      xH.data[i] = case0x[i]
    for i in 0 ..< E * H:
      rwH.data[i] = case0rw[i]
    for i in 0 ..< T * K:
      idsH.data[i] = -1
      wH.data[i] = -1.0'f32
    sigmoidRound(engine, Launcher, LauncherW, idsH, wH, xH, rwH, sH, T,
      int32(eCount), scale)
    for i in 0 ..< T * K:
      doAssert idsH.data[i] == case0ids[i], "sigmoid ids differ run to run"
      doAssert wH.data[i] == case0w[i], "sigmoid weights differ run to run"
  echo "[" & label & "] cases=" & $cases &
    " worst bar usage " & formatFloat(worstUse, ffScientific, 3)

# ─── The ragged sigmoid boundary ──────────────────────────────────────

proc runRaggedSigmoid[E: HwEngine](engine: E) =
  ## Ragged sigmoid boundary: the router weight view's expert count
  ## E = 70 is not a multiple of the 64-expert staging chunk.
  ## It is not a multiple of the 32-lane selection slice either.
  ##
  ## Expert 65's router weight vector dominates, its logit +1024
  ## against -1024 for every other expert. Expert 65 must win top-1,
  ## and the remaining slots take the lowest real indices at weight 1
  ## for the winner, 0 for the zero-score fillers, both exact in fp32.
  const
    T = 1
    H = 2048
    E = 70
    K = 8
  var idsH = makeHostTensor(int32, (T, K), (K, 1))
  var wH = makeHostTensor(float32, (T, K), (K, 1))
  var xH = makeHostTensor(bfloat16, (T, H), (H, 1))
  var rwH = makeHostTensor(bfloat16, (E, H), (H, 1))
  var sH = makeHostTensor(float32, (T, E), (E, 1))
  for i in 0 ..< T * H:
    xH.data[i] = 1.0'f32.to(bfloat16)
  for e in 0 ..< E:
    let v = (if e == 65: 0.5'f32 else: -0.5'f32).to(bfloat16)
    for k in 0 ..< H:
      rwH.data[e * H + k] = v
  for i in 0 ..< T * E:
    sH.data[i] = 0.0'f32
  for i in 0 ..< T * K:
    idsH.data[i] = -1
    wH.data[i] = -1.0'f32
  sigmoidRound(engine, sigmoidBf16, sigmoidBf16W, idsH, wH, xH, rwH, sH, T,
    int32(E), 1.0'f32)
  doAssert idsH.data[0] == 65,
    &"ragged sigmoid: expert 65 must win top-1, got expert {idsH.data[0]}"
  for slot in 1 ..< K:
    doAssert idsH.data[slot] == int32(slot - 1),
      &"ragged sigmoid: slot {slot} routed to expert {idsH.data[slot]}, " &
      &"want the lowest zero-score index {slot - 1}"
  doAssert wH.data[0] == 1.0'f32,
    &"ragged sigmoid: winning weight {wH.data[0]}, want 1.0 in exact fp32"
  for slot in 1 ..< K:
    doAssert wH.data[slot] == 0.0'f32,
      &"ragged sigmoid: filler weight at slot {slot} is {wH.data[slot]}, want 0"
  echo "[ragged sigmoid bf16 E=70] expert 65 wins top-1, fillers 0..6 at weight 0"

# ─── The tie-break case ───────────────────────────────────────────────

proc runTieCase[E: HwEngine](engine: E) =
  ## Lowest-index tiebreak on exactly equal El-rounded logits. Router weight vectors
  ## 5 and 9 are bit-identical and dominant, so both experts' logits agree
  ## bit for bit on both sides and the ids must read [5, 9, ...],
  ## the lower id first, weights of the pair equal.
  const
    T = 1
    H = 2048
    E = 256
    K = 8
  var idsH = makeHostTensor(int32, (T, K), (K, 1))
  var wH = makeHostTensor(bfloat16, (T, K), (K, 1))
  var xH = makeHostTensor(bfloat16, (T, H), (H, 1))
  var rwH = makeHostTensor(bfloat16, (E, H), (H, 1))
  var sH = makeHostTensor(float32, (T, E), (E, 1))
  var rng = initRand(0xC04D0535'i64)
  for i in 0 ..< T * H:
    xH.data[i] = rng.rand(-1.0'f32 .. 1.0'f32).to(bfloat16)
  for i in 0 ..< E * H:
    rwH.data[i] = rng.rand(-0.02'f32 .. 0.02'f32).to(bfloat16)
  for k in 0 ..< H:
    let dom = 0.5'f32.to(bfloat16)
    rwH.data[5 * H + k] = dom
    rwH.data[9 * H + k] = dom
  for i in 0 ..< T * K:
    idsH.data[i] = -1
    wH.data[i] = zero(bfloat16)
  let idsDev = engine.bridge(idsH)
  let wDev = engine.bridge(wH)
  let xDev = engine.bridge(xH)
  let rwDev = engine.bridge(rwH)
  let sDev = engine.bridge(sH)
  routeRound(engine, routeBf16Mega, routeBf16MegaW, idsH, wH, xH, rwH, sH, T)
  doAssert idsH.data[0] == 5, &"tie-break: first slot routed to {idsH.data[0]}, want 5"
  doAssert idsH.data[1] == 9, &"tie-break: second slot routed to {idsH.data[1]}, want 9"
  doAssert cast[uint16](wH.data[0]) == cast[uint16](wH.data[1]),
    "tie-break: tied experts carry different weights"
  let rawLogits = rawLogitsF64[bfloat16](xH.data, rwH.data, T, H, E)
  var wantLogits = newSeq[float64](E)
  for e in 0 ..< E:
    wantLogits[e] = elRoundF64[bfloat16](rawLogits[e])
  doAssert wantLogits[5] == wantLogits[9], "tie-break: reference logits of the duplicated rows differ"
  let p = softmaxF64(wantLogits, E)
  let sel = topKF64(p, E, K)
  doAssert sel[0] == 5 and sel[1] == 9, "tie-break: reference top-K disagrees"
  for slot in 0 ..< K:
    doAssert idsH.data[slot] == sel[slot],
      &"tie-break: slot {slot} routed to {idsH.data[slot]}, want {sel[slot]}"
  echo "[tie-break bf16] ids [5, 9, ...], lowest index wins the exact tie"

# ─── The poisoned pass ────────────────────────────────────────────────

proc runPoisoned[E: HwEngine](engine: E) =
  ## Poisoned score pass: the router weight vectors alternate NaN and +Inf
  ## bit patterns. Every logit is non-finite, so every staged score NaNs
  ## through the softmax denominator and no staged score ever matches
  ## the group max.
  ##
  ## All K slots must take the unmatched branch, with id E−1,
  ## weight set to the zero pattern, and the chain relaunches bit-identical.
  const
    T = 1
    H = 2048
    E = 256
    K = 8
  var idsH = makeHostTensor(int32, (T, K), (K, 1))
  var wH = makeHostTensor(bfloat16, (T, K), (K, 1))
  var xH = makeHostTensor(bfloat16, (T, H), (H, 1))
  var rwH = makeHostTensor(bfloat16, (E, H), (H, 1))
  var sH = makeHostTensor(float32, (T, E), (E, 1))
  var rng = initRand(0xC04D0536'i64)
  for i in 0 ..< T * H:
    xH.data[i] = rng.rand(-1.0'f32 .. 1.0'f32).to(bfloat16)
  for e in 0 ..< E:
    let bits = if (e and 1) == 0: 0x7FC0'u16 else: 0x7F80'u16
    for k in 0 ..< H:
      rwH.data[e * H + k] = cast[bfloat16](bits)
  for i in 0 ..< T * K:
    idsH.data[i] = -777
    wH.data[i] = cast[bfloat16](0xBEEF'u16)
  let idsDev = engine.bridge(idsH)
  let wDev = engine.bridge(wH)
  let xDev = engine.bridge(xH)
  let rwDev = engine.bridge(rwH)
  let sDev = engine.bridge(sH)
  routeRound(engine, routeBf16Mega, routeBf16MegaW, idsH, wH, xH, rwH, sH, T)
  for slot in 0 ..< K:
    doAssert idsH.data[slot] == int32(E - 1),
      &"poisoned pass slot {slot} not on the unmatched branch: {idsH.data[slot]}"
    doAssert cast[uint16](wH.data[slot]) == 0x0000'u16,
      &"poisoned pass weight not the zero pattern at slot {slot}: " &
      &"0x{cast[uint16](wH.data[slot]):04x}"
  # Relaunch: bit-identical ids and weights.
  for i in 0 ..< T * K:
    idsH.data[i] = -777
    wH.data[i] = cast[bfloat16](0xBEEF'u16)
  let idsDev2 = engine.bridge(idsH)
  let wDev2 = engine.bridge(wH)
  let xDev2 = engine.bridge(xH)
  let rwDev2 = engine.bridge(rwH)
  let sDev2 = engine.bridge(sH)
  routeRound(engine, routeBf16Mega, routeBf16MegaW, idsH, wH, xH, rwH, sH, T)
  for slot in 0 ..< K:
    doAssert idsH.data[slot] == int32(E - 1), "poisoned pass ids differ run to run"
    doAssert cast[uint16](wH.data[slot]) == 0x0000'u16,
      "poisoned pass weights differ run to run"
  echo "[poisoned router bf16 mega] all K slots on the unmatched branch, " &
    &"weights the zero pattern, relaunch bit-identical"

# ─── The shared-expert gate logit ─────────────────────────────────────

template runSharedGateCombo(engine, Launcher: untyped, El: typedesc, T: static int, cases: int, seed: uint64, label: string) =
  ## Shared-gate weight logit at the mega binding (H = 2048). The raw fp32
  ## GEMV output judged against the exact f64 dot under the chunk-walk
  ## reassociation bar. Case 0 relaunched bit-identical.
  const H = 2048
  var outH = makeHostTensor(float32, (T, 1), (1, 1))
  var xH = makeHostTensor(El, (T, H), (H, 1))
  var sgwH = makeHostTensor(El, (1, H), (H, 1))
  var rng = initRand(int64(seed))
  var worstUse = 0.0'f64
  var case0x = newSeq[El](T * H)
  var case0g = newSeq[El](H)
  var case0out = newSeq[float32](T)
  for caseId in 0 ..< cases:
    for i in 0 ..< T * H:
      xH.data[i] = rng.rand(-1.0'f32 .. 1.0'f32).to(El)
    for k in 0 ..< H:
      sgwH.data[k] = rng.rand(-1.0'f32 .. 1.0'f32).to(El)
    for tok in 0 ..< T:
      outH.data[tok] = 4.203895392974451e-45'f32
    let outDev = engine.bridge(outH)
    let xDev = engine.bridge(xH)
    let gDev = engine.bridge(sgwH)
    engine.launchDeviceFn Launcher(outDev, xDev, gDev) do:
      grid = (T, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr outH.data[0], outDev.data, outH.len())
    for tok in 0 ..< T:
      doAssert outH.data[tok] != 4.203895392974451e-45'f32,
        "shared gate logit never stored at token " & $tok
      let want = gateDotF64[El](xH.data, sgwH.data, H, tok)
      var sumAbs = 0.0'f64
      for k in 0 ..< H:
        sumAbs += abs(xH.data[tok * H + k].widenF64() * sgwH.data[k].widenF64())
      let bar = sharedGateBar(H, sumAbs)
      let diff = abs(outH.data[tok].float64 - want)
      doAssert diff <= bar,
        "shared gate logit outside the bar at (tok " & $tok & ", case " &
        $caseId & "): " & $diff & " > " & $bar
      worstUse = max(worstUse, diff / bar)
    if caseId == 0:
      for i in 0 ..< T * H:
        case0x[i] = xH.data[i]
      for k in 0 ..< H:
        case0g[k] = sgwH.data[k]
      for tok in 0 ..< T:
        case0out[tok] = outH.data[tok]
  block determinism:
    for i in 0 ..< T * H:
      xH.data[i] = case0x[i]
    for k in 0 ..< H:
      sgwH.data[k] = case0g[k]
    for tok in 0 ..< T:
      outH.data[tok] = 4.203895392974451e-45'f32
    let outDev = engine.bridge(outH)
    let xDev = engine.bridge(xH)
    let gDev = engine.bridge(sgwH)
    engine.launchDeviceFn Launcher(outDev, xDev, gDev) do:
      grid = (T, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr outH.data[0], outDev.data, outH.len())
    for tok in 0 ..< T:
      doAssert outH.data[tok] == case0out[tok], "shared gate logit differs run to run"
  echo "[" & label & "] cases=" & $cases &
    " worst bar usage " & formatFloat(worstUse, ffScientific, 3)

# ─── The fp32-partial merge ───────────────────────────────────────────

template runMergeCombo(engine, Launcher: untyped, El: typedesc, T, H, K: static int, cases: int, seed: uint64, label: string) =
  ## fp32-partial merge, judged bit-exact against the same sequential fp32
  ## slot-order sum with one El RNE round at the store, both sides narrow
  ## the same fp32 accumulator.
  var outH = makeHostTensor(El, (T, H), (H, 1))
  var pH = makeHostTensor(float32, (T, K + 1, H), ((K + 1) * H, H, 1))
  var rng = initRand(int64(seed))
  var exact = 0
  var total = 0
  for caseId in 0 ..< cases:
    for i in 0 ..< T * (K + 1) * H:
      pH.data[i] = rng.rand(-1.0'f32 .. 1.0'f32)
    for i in 0 ..< T * H:
      outH.data[i] = cast[El](0xFFFF'u16)
    let outDev = engine.bridge(outH)
    let pDev = engine.bridge(pH)
    engine.launchDeviceFn Launcher(outDev, pDev) do:
      grid = (T, H div 32)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr outH.data[0], outDev.data, outH.len())
    for tok in 0 ..< T:
      for col in 0 ..< H:
        var acc = 0.0'f32
        for y in 0 ..< K + 1:
          acc += pH.data[tok * (K + 1) * H + y * H + col]
        let want = acc.to(El)
        doAssert cast[uint16](outH.data[tok * H + col]) == cast[uint16](want),
          "merge output not bit-identical at (tok " & $tok & ", col " & $col &
          ", case " & $caseId & ")"
        if cast[uint16](outH.data[tok * H + col]) == cast[uint16](want):
          inc exact
        inc total
  echo "[" & label & "] cases=" & $cases & " bit-exact " & $exact & "/" & $total

# ─── The suite ────────────────────────────────────────────────────────

proc main() =
  var engine = newEngine(runMetal)
  echo "device: ", engine.deviceName()
  runProbe(engine)
  runRouteCombo(engine, routeBf16Mega, routeBf16MegaW, bfloat16, 8, 2048, 256, 8, 1.0'f32, 4,
    0xC04D0521'u64, "router mega bf16")
  runRouteCombo(engine, routeF16Mega, routeF16MegaW, float16, 8, 2048, 256, 8, 1.0'f32, 4,
    0xC04D0522'u64, "router mega f16")
  runRouteCombo(engine, routeBf16Small, routeBf16SmallW, bfloat16, 4, 256, 64, 4, 1.0'f32, 4,
    0xC04D0523'u64, "router small bf16")
  runRouteCombo(engine, routeBf16MegaS2, routeBf16MegaS2W, bfloat16, 8, 2048, 256, 8, 2.0'f32, 4,
    0xC04D0528'u64, "router mega bf16 scale 2")
  runSigmoidCombo(engine, sigmoidBf16, sigmoidBf16W, bfloat16, 2, 2048, 128, 8, 128, 1.0'f32,
    2, 0xC04D0533'u64, "sigmoid bf16 exact eCount")
  runSigmoidCombo(engine, sigmoidBf16, sigmoidBf16W, bfloat16, 2, 2048, 128, 8, 70, 1.8'f32,
    2, 0xC04D0534'u64, "sigmoid bf16 ragged eCount")
  runRaggedSigmoid(engine)
  runTieCase(engine)
  runPoisoned(engine)
  runSharedGateCombo(engine, sharedGateBf16, bfloat16, 8, 2, 0xC04D0527'u64,
    "shared gate bf16")
  runSharedGateCombo(engine, sharedGateF16, float16, 8, 2, 0xC04D0531'u64,
    "shared gate f16")
  runMergeCombo(engine, mergeBf16Small, bfloat16, 8, 256, 8, 2, 0xC04D0524'u64,
    "merge bf16 small")
  runMergeCombo(engine, mergeF16Small, float16, 8, 256, 8, 2, 0xC04D0532'u64,
    "merge f16 small")
  runMergeCombo(engine, mergeBf16Mega, bfloat16, 8, 2048, 8, 1, 0xC04D0529'u64,
    "merge bf16 mega")

main()
