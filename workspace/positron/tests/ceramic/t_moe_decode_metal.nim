# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal MoE decode suite, tile-stack kernel judged against f64 host references
##
## Test that the tile-stack MoE decode walk matches its f64 references end to end:
##
## - the routed h_scratch slices and fp32 partials
## - the shared slot's sigmoid-gated value, hs_scratch slice and partial
## - the bit-exact fp32 slot-order merge
## - the scratch bounds walk and relaunch determinism
##
## Cases prod / nogate / scale2 at the production shape
## H=2048, E=256, K=8, I=512, bfloat16.
##
## Run:
##   nim c -r --outdir:build/tests --nimcache:nimcache/moe_decode --hints:off workspace/positron/tests/ceramic/t_moe_decode_metal.nim

import std/[math, random]
import workspace/crucible
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/tests/ceramic_testutils
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_router
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_decode
import moe_ref_utils

const
  H = 2048
  E = 256
  K = 8
  I = 512
  Sentinel = 4.203895392974451e-45'f32 # 3·2^-149, a denormal no real partial can equal, marks an unwritten cell

# ─── Device entries, one per (case, readback tensor) binding ──────────

proc routeProd(ids, rout_w, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 2048, 256, 8, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc routeProdW(rout_w, ids, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 2048, 256, 8, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc routeScale2(ids, rout_w, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 2048, 256, 8, 2.0'f32](ids, rout_w, x, router_w, scratch)

proc routeScale2W(rout_w, ids, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[bfloat16, 2048, 256, 8, 2.0'f32](ids, rout_w, x, router_w, scratch)

proc decodeProdP(partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores: distinct TensorView) =
  moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 1.0'f32, true](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc decodeProdH(h, partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, hs, scores: distinct TensorView) =
  moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 1.0'f32, true](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc decodeProdS(hs, partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, scores: distinct TensorView) =
  moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 1.0'f32, true](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc decodeNogateP(partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores: distinct TensorView) =
  moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 1.0'f32, false](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc decodeNogateH(h, partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, hs, scores: distinct TensorView) =
  moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 1.0'f32, false](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc decodeNogateS(hs, partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, scores: distinct TensorView) =
  moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 1.0'f32, false](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc decodeScale2P(partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores: distinct TensorView) =
  moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 2.0'f32, true](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc decodeScale2H(h, partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, hs, scores: distinct TensorView) =
  moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 2.0'f32, true](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc decodeScale2S(hs, partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, scores: distinct TensorView) =
  moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 2.0'f32, true](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc mergeBf16(out_r, partial: distinct TensorView) =
  moe_decode_merge_at[bfloat16, 2048, 8](out_r, partial, int32(threadgroup_position_in_grid.x), int32(threadgroup_position_in_grid.y))

# ─── Launch rounds ────────────────────────────────────────────────────

template routeRound(engine, LauncherIds, LauncherW, idsH, wH, xH, rwH, sH: untyped, gridX: int) =
  ## One routing judgment round over the bridged router tensors, two launches:
  ##
  ## - the ids launch first, then the weights launch
  ## - the launch machinery consumes the non-output bridge arguments
  ##   (move-only DeviceMem), a tensor the host reads back must be its launch's first argument
  ## - the weights launch rewrites the copied ids with identical values
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

template decodeRound(engine, LP, LH, LS, pH, xH, rwH, guH, dH, sgmH, supH,
    sdmH, sgvH, hH, hsH, scH: untyped, gridX: int, which: static int, dst: untyped,
    n: int) =
  ## One decode launch over the bridged fixtures:
  ##
  ## - the launch machinery consumes the non-output bridge arguments
  ##   (move-only DeviceMem), a tensor the host reads back must be its launch's first argument
  ## - the copy-back sits inside the launch block
  ## - `which` selects the readback: 0 = partial, 1 = h_scratch,
  ##   2 = hs_scratch
  block:
    let pDev = engine.bridge(pH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let guDev = engine.bridge(guH)
    let dDev = engine.bridge(dH)
    let sgmDev = engine.bridge(sgmH)
    let supDev = engine.bridge(supH)
    let sdmDev = engine.bridge(sdmH)
    let sgvDev = engine.bridge(sgvH)
    let hDev = engine.bridge(hH)
    let hsDev = engine.bridge(hsH)
    let scDev = engine.bridge(scH)
    when which == 0:
      engine.launchDeviceFn LP(pDev, xDev, rwDev, guDev, dDev, sgmDev, supDev,
          sdmDev, sgvDev, hDev, hsDev, scDev) do:
        grid = (gridX, K + 1)
        blk = (32, 1)
      engine.copyToHostFromDevice(addr dst.data[0], pDev.data, n)
    elif which == 1:
      engine.launchDeviceFn LH(hDev, pDev, xDev, rwDev, guDev, dDev, sgmDev,
          supDev, sdmDev, sgvDev, hsDev, scDev) do:
        grid = (gridX, K + 1)
        blk = (32, 1)
      engine.copyToHostFromDevice(addr dst.data[0], hDev.data, n)
    else:
      engine.launchDeviceFn LS(hsDev, pDev, xDev, rwDev, guDev, dDev, sgmDev,
          supDev, sdmDev, sgvDev, hDev, scDev) do:
        grid = (gridX, K + 1)
        blk = (32, 1)
      engine.copyToHostFromDevice(addr dst.data[0], hsDev.data, n)

template mergeRound(engine, outH, pH: untyped, gridX: int) =
  ## One merge launch over the bridged fp32 partials, then the out_r copy-back.
  block:
    let outDev = engine.bridge(outH)
    let pDev = engine.bridge(pH)
    engine.launchDeviceFn mergeBf16(outDev, pDev) do:
      grid = (gridX, H div 32)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr outH.data[0], outDev.data, outH.len())

# ─── Fixture recipes (the v1 decode suite's) ──────────────────────────

proc fillFormula(dst: ptr UncheckedArray[bfloat16], n: int) =
  ## Weights formula: magnitudes [-0.025, 0.0251] per the v1 decode recipe.
  for jj in 0 ..< n:
    dst[jj] = (float32(jj mod 501 - 250) * 1.0e-4'f32).to(bfloat16)

proc fillRngBf(dst: ptr UncheckedArray[bfloat16], rng: var Rand, n: int, lo, hi: float32) =
  for jj in 0 ..< n:
    dst[jj] = rng.rand(lo .. hi).to(bfloat16)

# ─── The f64 route judgment over the pre-pass ids and weights ─────────

template judgeRoute(idsH, wH, xH, rwH: untyped, nTok: int, uStep: float64, scaleV: float32, label: string) =
  ## Asserts the pre-pass routing against the f64 reference: ids inside
  ## [0, E) and distinct, slot pairs against the top-K under the tie region,
  ## weights per element under the fp32 bar.
  let rawLogits = rawLogitsF64[bfloat16](xH.data, rwH.data, nTok, H, E)
  for tok in 0 ..< nTok:
    var wantLogits = newSeq[float64](E)
    for jj in 0 ..< E:
      wantLogits[jj] = elRoundF64[bfloat16](rawLogits[tok * E + jj])
    let p = softmaxF64(wantLogits, E)
    let sel = topKF64(p, E, K)
    for slot in 0 ..< K:
      let kid = idsH.data[tok * K + slot]
      doAssert kid >= 0 and kid < int32(E),
        label & ": id out of range at (tok " & $tok & ", slot " & $slot & ")"
      for other in 0 ..< slot:
        doAssert kid != idsH.data[tok * K + other],
          label & ": duplicate id at (tok " & $tok & ", slot " & $slot & ")"
      if kid != sel[slot]:
        # a tie swap: the swapped pair's logit gap must sit within the pair's reassociation and El grid slack bound
        let dA = sumAbsF64[bfloat16](xH.data, rwH.data, H, tok * H, kid.int * H)
        let dB = sumAbsF64[bfloat16](xH.data, rwH.data, H, tok * H,
            sel[slot].int * H)
        let bound = pairTieBound(uStep, H, dA, dB,
            rawLogits[tok * E + kid.int], rawLogits[tok * E + sel[slot].int])
        doAssert abs(rawLogits[tok * E + kid.int] - rawLogits[tok * E + sel[slot].int]) <= bound,
          label & ": non-tie id mismatch at (tok " & $tok & ", slot " & $slot & ")"
    for slot in 0 ..< K:
      let kid = idsH.data[tok * K + slot].int
      var sumSel = 0.0'f64
      for other in 0 ..< K:
        sumSel += p[idsH.data[tok * K + other].int]
      let wantW = p[kid] / sumSel * scaleV.float64
      doAssert abs(widenF64(wH.data[tok * K + slot]) - wantW) <=
          weightBar(uStep, wantW, E, K),
        label & ": weight bar at (tok " & $tok & ", slot " & $slot & ")"

# ─── The decode judgment: bounds walk, f64 bars, bit-exact merge ──────

template judgeDecode(hH, hsH, pH, outH, idsH, wH, xH, rwH, guH, dH, sgmH, supH,
    sdmH, sgvH: untyped, nTok: int, uStep: float64, useGate: static bool,
    label: string) =
  ## Full decode judgment over the read-back scratch and output buffers:
  ##
  ## - the bounds walk: every claimed cell written and the spare
  ##   (T+1)-th scratch slices untouched
  ## - the routed h_scratch slices and the shared hs_scratch slice
  ##   against the f64 silu-mul chain under the derived bar
  ## - the fp32 partial slices against the f64 down dot scaled by w,
  ##   computed over the kernel's own El scratch slices, a pairing that isolates
  ##   the down walk's error from the already-barred h values
  ## - out_r bit-exact against the host fp32 slot-order sum of the kernel
  ##   partials with one RNE round, and the nogate case's poisoned vector
  ##   never reaches a result (the shared partial slice and out_r hold no NaN)
  ##
  ##   x, weights ──► [ f64 dots, exact El rounds ] ──► h/hs/partial refs
  ##   kernel partials ──► [ host fp32 slot-order sum ] ──► out_r (bit-exact)
  ##
  # ── bounds walk ──
  for jj in 0 ..< nTok * K * I:
    doAssert cast[uint16](hH.data[jj]) != 0xFFFF'u16,
      label & ": h_scratch cell " & $jj & " never written"
  for jj in 0 ..< nTok * I:
    doAssert cast[uint16](hsH.data[jj]) != 0xFFFF'u16,
      label & ": hs_scratch cell " & $jj & " never written"
  for jj in 0 ..< nTok * (K + 1) * H:
    doAssert pH.data[jj] != Sentinel,
      label & ": partial cell " & $jj & " never written"
  for jj in 0 ..< nTok * H:
    doAssert cast[uint16](outH.data[jj]) != 0xFFFF'u16,
      label & ": out cell " & $jj & " never written"
  for jj in nTok * K * I ..< (nTok + 1) * K * I:
    doAssert cast[uint16](hH.data[jj]) == 0xFFFF'u16,
      label & ": h_scratch bleed into the spare row at " & $jj
  for jj in nTok * I ..< (nTok + 1) * I:
    doAssert cast[uint16](hsH.data[jj]) == 0xFFFF'u16,
      label & ": hs_scratch bleed into the spare row at " & $jj
  for jj in nTok * (K + 1) * H ..< (nTok + 1) * (K + 1) * H:
    doAssert pH.data[jj] == Sentinel,
      label & ": partial bleed into the spare row at " & $jj
  # ── routed h rows ──
  for tok in 0 ..< nTok:
    for slot in 0 ..< K:
      let e = idsH.data[tok * K + slot].int
      for row in 0 ..< I:
        let gOff = (e * 2) * I * H + row * H
        let uOff = (e * 2 + 1) * I * H + row * H
        let g64 = dotOffF64[bfloat16](xH.data, guH.data, H, tok * H, gOff)
        let u64 = dotOffF64[bfloat16](xH.data, guH.data, H, tok * H, uOff)
        let hRef = hValueF64[bfloat16](g64, u64)
        let h64 = widenF64(hRef)
        let bar = hBarF64(H, uStep,
          sumAbsF64[bfloat16](xH.data, guH.data, H, tok * H, gOff),
          sumAbsF64[bfloat16](xH.data, guH.data, H, tok * H, uOff),
          siluF64(g64), u64, h64)
        let got = widenF64(hH.data[(tok * K + slot) * I + row])
        doAssert abs(got - h64) <= bar,
          label & ": h bar at (tok " & $tok & ", slot " & $slot & ", row " &
            $row & "): got " & $got & ", want " & $h64 & ", bar " & $bar
  # ── shared hs row ──
  for tok in 0 ..< nTok:
    var gateVal64 = 1.0'f64
    when useGate:
      let l64 = dotOffF64[bfloat16](xH.data, sgvH.data, H, tok * H, 0)
      let sumAbsGate = sumAbsF64[bfloat16](xH.data, sgvH.data, H, tok * H, 0)
      gateVal64 = narrowEl[bfloat16](
        1.0 / (1.0 + exp2(-l64 * Log2e64))).to(float32).float64
    for row in 0 ..< I:
      let g64 = dotOffF64[bfloat16](xH.data, sgmH.data, H, tok * H, row * H)
      let u64 = dotOffF64[bfloat16](xH.data, supH.data, H, tok * H, row * H)
      let hsRef = hValueF64[bfloat16](g64, u64)
      let hs64 = widenF64(hsRef)
      let bar = hBarF64(H, uStep,
        sumAbsF64[bfloat16](xH.data, sgmH.data, H, tok * H, row * H),
        sumAbsF64[bfloat16](xH.data, supH.data, H, tok * H, row * H),
        siluF64(g64), u64, hs64)
      let got = widenF64(hsH.data[tok * I + row])
      doAssert abs(got - hs64) <= bar,
        label & ": hs bar at (tok " & $tok & ", row " & $row & "): got " &
          $got & ", want " & $hs64 & ", bar " & $bar
    # ── shared partial row ──
    for col in 0 ..< H:
      let d64 = dotOffF64[bfloat16](hsH.data, sdmH.data, I, tok * I, col * I)
      let sumAbsDown = sumAbsF64[bfloat16](hsH.data, sdmH.data, I, tok * I,
          col * I)
      let wantP = gateVal64 * d64
      when useGate:
        let bar = sharedPartialBarF64(I, H, uStep, sumAbsDown, sumAbsGate,
            gateVal64, d64, wantP)
        doAssert abs(pH.data[tok * (K + 1) * H + K * H + col] - wantP) <= bar,
          label & ": shared partial bar at (tok " & $tok & ", col " & $col &
            ")"
      else:
        let bar = partialBarF64(I, sumAbsDown, 1.0, wantP)
        doAssert abs(pH.data[tok * (K + 1) * H + K * H + col] - wantP) <= bar,
          label & ": shared partial bar at (tok " & $tok & ", col " & $col &
            ")"
        doAssert classify(pH.data[tok * (K + 1) * H + K * H + col]) notin
            {fcNan, fcInf},
          label & ": NaN in the shared partial row at (tok " & $tok &
            ", col " & $col & ")"
  # ── routed partial rows ──
  for tok in 0 ..< nTok:
    for slot in 0 ..< K:
      let e = idsH.data[tok * K + slot].int
      let wv = widenF64(wH.data[tok * K + slot])
      for col in 0 ..< H:
        let dOff = e * H * I + col * I
        let d64 = dotOffF64[bfloat16](hH.data, dH.data, I,
            (tok * K + slot) * I, dOff)
        let sumAbsDown = sumAbsF64[bfloat16](hH.data, dH.data, I,
            (tok * K + slot) * I, dOff)
        let wantP = wv * d64
        let bar = partialBarF64(I, sumAbsDown, wv, wantP)
        doAssert abs(pH.data[tok * (K + 1) * H + slot * H + col] - wantP) <= bar,
          label & ": partial bar at (tok " & $tok & ", slot " & $slot &
            ", col " & $col & ")"
  # ── out_r bit-exact against the host fp32 slot-order sum ──
  for tok in 0 ..< nTok:
    for col in 0 ..< H:
      var acc = 0.0'f32
      for yv in 0 ..< K + 1:
        acc += pH.data[tok * (K + 1) * H + yv * H + col]
      let want = narrowEl[bfloat16](acc.float64)
      doAssert cast[uint16](outH.data[tok * H + col]) == cast[uint16](want),
        label & ": merge mismatch at (tok " & $tok & ", col " & $col & ")"

# ─── Case runner ──────────────────────────────────────────────────────

template runCase(engine, RouteIds, RouteW, CaseP, CaseH, CaseS: untyped,
    nTok: int, seed: int64, scaleV: float32, useGate: static bool,
    label: string) =
  ## One case:
  ##
  ## - fixture fill, then the routing pre-pass judged against the f64 route reference
  ## - three decode readback rounds, then the merge
  ## - the full decode judgment plus the relaunch determinism check
  ##   on the partials and out_r
  let uStep = uStepF64(bfloat16)
  var rng = initRand(seed)
  var xH = makeHostTensor(bfloat16, (nTok, H), (H, 1))
  var rwH = makeHostTensor(bfloat16, (E, H), (H, 1))
  var guH = makeHostTensor(bfloat16, (E, 2, I, H), (2 * I * H, I * H, H, 1))
  var dH = makeHostTensor(bfloat16, (E, H, I), (H * I, I, 1))
  var sgmH = makeHostTensor(bfloat16, (1, I, H), (I * H, H, 1))
  var supH = makeHostTensor(bfloat16, (1, I, H), (I * H, H, 1))
  var sdmH = makeHostTensor(bfloat16, (1, H, I), (H * I, I, 1))
  var sgvH = makeHostTensor(bfloat16, (1, H), (H, 1))
  var hH = makeHostTensor(bfloat16, (nTok + 1, K, I), (K * I, I, 1))
  var hsH = makeHostTensor(bfloat16, (nTok + 1, I), (I, 1))
  var pH = makeHostTensor(float32, (nTok + 1, K + 1, H), ((K + 1) * H, H, 1))
  var outH = makeHostTensor(bfloat16, (nTok, H), (H, 1))
  var idsH = makeHostTensor(int32, (nTok, K), (K, 1))
  var wH = makeHostTensor(bfloat16, (nTok, K), (K, 1))
  var routeScratchH = makeHostTensor(float32, (nTok, E), (E, 1))
  var scH = makeHostTensor(float32, (nTok, K + 1, E), ((K + 1) * E, E, 1))

  fillRngBf(xH.data, rng, nTok * H, -1.0'f32, 1.0'f32)
  fillRngBf(rwH.data, rng, E * H, -0.02'f32, 0.02'f32)
  fillFormula(guH.data, E * 2 * I * H)
  fillFormula(dH.data, E * H * I)
  fillFormula(sgmH.data, I * H)
  fillFormula(supH.data, I * H)
  fillFormula(sdmH.data, H * I)
  when useGate:
    fillRngBf(sgvH.data, rng, H, -1.0'f32, 1.0'f32)
  else:
    # poisoned sgv fixture for the nogate case, never read with SharedGate off. Every element contains the quiet NaN bit pattern
    for jj in 0 ..< H:
      sgvH.data[jj] = cast[bfloat16](0x7FC0'u16)
  for jj in 0 ..< (nTok + 1) * K * I:
    hH.data[jj] = cast[bfloat16](0xFFFF'u16)
  for jj in 0 ..< (nTok + 1) * I:
    hsH.data[jj] = cast[bfloat16](0xFFFF'u16)
  for jj in 0 ..< (nTok + 1) * (K + 1) * H:
    pH.data[jj] = Sentinel
  for jj in 0 ..< nTok * H:
    outH.data[jj] = cast[bfloat16](0xFFFF'u16)

  # ── routing pre-pass, judged against the f64 route reference ──
  routeRound(engine, RouteIds, RouteW, idsH, wH, xH, rwH, routeScratchH, nTok)
  judgeRoute(idsH, wH, xH, rwH, nTok, uStep, scaleV, label)

  # ── decode rounds: h_scratch, then partial, then hs_scratch ──
  decodeRound(engine, CaseP, CaseH, CaseS, pH, xH, rwH, guH, dH, sgmH, supH,
    sdmH, sgvH, hH, hsH, scH, nTok, 1, hH, (nTok + 1) * K * I)
  decodeRound(engine, CaseP, CaseH, CaseS, pH, xH, rwH, guH, dH, sgmH, supH,
    sdmH, sgvH, hH, hsH, scH, nTok, 0, pH, (nTok + 1) * (K + 1) * H)
  decodeRound(engine, CaseP, CaseH, CaseS, pH, xH, rwH, guH, dH, sgmH, supH,
    sdmH, sgvH, hH, hsH, scH, nTok, 2, hsH, (nTok + 1) * I)
  mergeRound(engine, outH, pH, nTok)

  judgeDecode(hH, hsH, pH, outH, idsH, wH, xH, rwH, guH, dH, sgmH, supH,
    sdmH, sgvH, nTok, uStep, useGate, label)

  # ── determinism: relaunch the partial round and the merge, the results
  #    must be bit-identical ──
  var savedP = newSeq[float32](nTok * (K + 1) * H)
  var savedOut = newSeq[uint16](nTok * H)
  for jj in 0 ..< nTok * (K + 1) * H:
    savedP[jj] = pH.data[jj]
  for jj in 0 ..< nTok * H:
    savedOut[jj] = cast[uint16](outH.data[jj])
  decodeRound(engine, CaseP, CaseH, CaseS, pH, xH, rwH, guH, dH, sgmH, supH,
    sdmH, sgvH, hH, hsH, scH, nTok, 0, pH, (nTok + 1) * (K + 1) * H)
  mergeRound(engine, outH, pH, nTok)
  for jj in 0 ..< nTok * (K + 1) * H:
    doAssert pH.data[jj] == savedP[jj],
      label & ": partial relaunch mismatch at " & $jj
  for jj in 0 ..< nTok * H:
    doAssert cast[uint16](outH.data[jj]) == savedOut[jj],
      label & ": out relaunch mismatch at " & $jj

  echo "[", label, "] bounds walk, h/hs/partial bars, merge bit-exact, determinism: green"

proc main() =
  var engine = newEngine(runMetal)
  echo "device: ", engine.deviceName()
  runCase(engine, routeProd, routeProdW, decodeProdP, decodeProdH, decodeProdS,
    4, 0xC04D0601'i64, 1.0'f32, true, "prod")
  runCase(engine, routeProd, routeProdW, decodeNogateP, decodeNogateH,
    decodeNogateS, 1, 0xC04D0602'i64, 1.0'f32, false, "nogate")
  runCase(engine, routeScale2, routeScale2W, decodeScale2P, decodeScale2H,
    decodeScale2S, 1, 0xC04D0603'i64, 2.0'f32, true, "scale2")

main()
