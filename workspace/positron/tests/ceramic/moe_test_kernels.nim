# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ──────────────────  moe_test_kernels (shared MoE metal test machinery)  ──────────────────

## Device-entry procs, launch rounds, and f64 judgment templates shared
## by the Metal MoE suites, the fused-forward cases built on this module
## sitting in t_moe_fwd_metal.nim.
##
## - the device entries wrap one kernel launch per readback permutation
## - the launch rounds bridge the host fixtures and copy the readback back
## - the judgment templates compare the readback against the f64 references
##   of `moe_ref_utils` under the derived bars

import workspace/crucible
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_router
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_decode
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_fwd
import moe_ref_utils

# ─── Device entries, one per (binding, readback tensor) permutation ───
#
# Launch mechanics: the machinery consumes the non-output bridge arguments
# (move-only DeviceMem). A tensor the host reads back must therefore lead
# its launch's argument list. Per launch the readback permutation
# reorders the formals: the readback tensor takes the first slot and the buffer
# it replaces takes the displaced formal's slot, since out_r,
# h_scratch, hs_scratch share the fp16 pointer type. Each launch writes
# all three fp16 outputs, so every round produces identical values
# in every buffer.

proc routePreSigmoid*(ids: ptr UncheckedArray[int32],
    rout_w: ptr UncheckedArray[float32],
    x, router_w: ptr UncheckedArray[float16],
    scratch: ptr UncheckedArray[float32],
    num_tokens, hidden, n_routed_experts, top_k: int32, scale: float32) =
  ## Grid (num_tokens, 1), blk (32, 1): the runtime sigmoid route pre-pass.
  ##
  ## Stores:
  ##
  ## - ids: the top-K expert ids (int32)
  ## - rout_w: the fp32 routing weights, which `moe_fwd` recomputes
  ##   identically inside the fused kernel
  let t = int32(threadgroup_position_in_grid.x)
  var idsReg: array[MaxTopK, int32]
  var wReg: array[MaxTopK, float32]
  let
    tok = int(t)
    shpTH: (int, int) = (int(num_tokens), int(hidden))
    shpEH: (int, int) = (int(n_routed_experts), int(hidden))
    shpSc: (int, int) = (int(num_tokens), ScoreChunk * ScoreChunks)
    strTH: (int, int) = (int(hidden), 1)
    strSc: (int, int) = (ScoreChunk * ScoreChunks, 1)
    xV = make_view(x, shpTH, strTH)
    rwV = make_view(router_w, shpEH, strTH)
    scV = make_view(scratch, shpSc, strSc)
  moeRouteSigmoidRT[float16](xV, rwV, t, scV(tok, _), top_k, scale, idsReg, wReg)
  for slot in 0 ..< top_k.int:
    ids[tok * top_k.int + slot] = idsReg[slot]
    rout_w[tok * top_k.int + slot] = wReg[slot]

proc routePreW*(rout_w: ptr UncheckedArray[float32],
    ids: ptr UncheckedArray[int32],
    x, router_w: ptr UncheckedArray[float16],
    scratch: ptr UncheckedArray[float32],
    num_tokens, hidden, n_routed_experts, top_k: int32, scale: float32) =
  routePreSigmoid(ids, rout_w, x, router_w, scratch, num_tokens, hidden,
    n_routed_experts, top_k, scale)

proc fwdGlmO*(out_r, x, router_w, gate_up_w, down_w, sgu, sdw, h,
    hs: ptr UncheckedArray[float16], sc: ptr UncheckedArray[float32],
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts: int32, routed_scaling: float32) =
  moe_fwd(out_r, x, router_w, gate_up_w, down_w, sgu, sdw, h, hs, sc,
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts, routed_scaling, maSilu)

proc fwdGlmH*(h, x, router_w, gate_up_w, down_w, sgu, sdw, out_r,
    hs: ptr UncheckedArray[float16], sc: ptr UncheckedArray[float32],
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts: int32, routed_scaling: float32) =
  moe_fwd(out_r, x, router_w, gate_up_w, down_w, sgu, sdw, h, hs, sc,
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts, routed_scaling, maSilu)

proc fwdGlmS*(hs, x, router_w, gate_up_w, down_w, sgu, sdw, h, out_r:
    ptr UncheckedArray[float16], sc: ptr UncheckedArray[float32],
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts: int32, routed_scaling: float32) =
  moe_fwd(out_r, x, router_w, gate_up_w, down_w, sgu, sdw, h, hs, sc,
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts, routed_scaling, maSilu)

proc fwdRagO*(out_r, x, router_w, gate_up_w, down_w, sgu, sdw, h,
    hs: ptr UncheckedArray[float16], sc: ptr UncheckedArray[float32],
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts: int32, routed_scaling: float32) =
  moe_fwd(out_r, x, router_w, gate_up_w, down_w, sgu, sdw, h, hs, sc,
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts, routed_scaling, maGeluTanh)

proc fwdRagH*(h, x, router_w, gate_up_w, down_w, sgu, sdw, out_r,
    hs: ptr UncheckedArray[float16], sc: ptr UncheckedArray[float32],
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts: int32, routed_scaling: float32) =
  moe_fwd(out_r, x, router_w, gate_up_w, down_w, sgu, sdw, h, hs, sc,
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts, routed_scaling, maGeluTanh)

proc fwdRagS*(hs, x, router_w, gate_up_w, down_w, sgu, sdw, h, out_r:
    ptr UncheckedArray[float16], sc: ptr UncheckedArray[float32],
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts: int32, routed_scaling: float32) =
  moe_fwd(out_r, x, router_w, gate_up_w, down_w, sgu, sdw, h, hs, sc,
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts, routed_scaling, maGeluTanh)

# ─── Cross-path device entries (the decode plus merge route) ──────────

proc decodeCross*(partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores: distinct TensorView) =
  moe_fwd_decode[float16, 2048, 64, 4, 512, 1.0'f32, false](
    partial, x, router_w, gate_up_w, down_w, sgm, sup, sdm, sgv, h, hs, scores)

proc mergeCross*(out_r, partial: distinct TensorView) =
  moe_decode_merge_at[float16, 2048, 4](out_r, partial,
    int32(threadgroup_position_in_grid.x), int32(threadgroup_position_in_grid.y))

proc routeCrossIds*(ids, rout_w, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[float16, 2048, 64, 4, 1.0'f32](ids, rout_w, x, router_w, scratch)

proc routeCrossW*(rout_w, ids, x, router_w, scratch: distinct TensorView) =
  moe_route_fwd[float16, 2048, 64, 4, 1.0'f32](ids, rout_w, x, router_w, scratch)

# ─── Launch rounds ────────────────────────────────────────────────────

template routePreRound*(engine, LIds, LW, idsH, wH, xH, rwH, sH: untyped, nTok, H, E: int, topK: int32, scaleV: float32) =
  ## One routing pre-pass judgment round, two launches:
  ##
  ## - the ids launch first, then the weights launch
  ## - the launch machinery consumes the non-output bridge arguments
  ##   (move-only DeviceMem), so a tensor the host reads back must lead
  ##   its launch's argument list
  ## - the raw-pointer formals take the DeviceMem values (`.data`), never the BridgeTensor wrappers, which bind garbage in that position
  ## - the weights launch rewrites the copied ids with identical values
  block:
    let idsDev = engine.bridge(idsH)
    let wDev = engine.bridge(wH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let sDev = engine.bridge(sH)
    engine.launchDeviceFn LIds(idsDev.data, wDev.data, xDev.data, rwDev.data,
        sDev.data, int32(nTok), int32(H), int32(E), topK, scaleV) do:
      grid = (nTok, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr idsH.data[0], idsDev.data, idsH.len())
  block:
    let idsDev = engine.bridge(idsH)
    let wDev = engine.bridge(wH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let sDev = engine.bridge(sH)
    engine.launchDeviceFn LW(wDev.data, idsDev.data, xDev.data, rwDev.data,
        sDev.data, int32(nTok), int32(H), int32(E), topK, scaleV) do:
      grid = (nTok, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr wH.data[0], wDev.data, wH.len())

template fwdRound*(engine, L, hH, hsH, outH, xH, rwH, guH, dH, sguH, sdwH,
    scH: untyped, nTok, H, E, I, K, S: int, scaleV: float32,
    which: static int, dst: untyped, n: int) =
  ## One fused launch over the bridged fixtures, `which` selecting the readback tensor:
  ## 0 = out_r, 1 = h_scratch, 2 = hs_scratch.
  ##
  ## - the readback tensor takes the launch's first argument slot and the buffer
  ##   it displaces takes the out_r formal's slot (see the device entries' permutation note)
  ## - the raw-pointer formals take the DeviceMem values (`.data`)
  ## - the copy-back sits inside the launch block
  block:
    let oDev = engine.bridge(outH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let guDev = engine.bridge(guH)
    let dDev = engine.bridge(dH)
    let sguDev = engine.bridge(sguH)
    let sdwDev = engine.bridge(sdwH)
    let hDev = engine.bridge(hH)
    let hsDev = engine.bridge(hsH)
    let scDev = engine.bridge(scH)
    when which == 0:
      engine.launchDeviceFn L(oDev.data, xDev.data, rwDev.data, guDev.data,
          dDev.data, sguDev.data, sdwDev.data, hDev.data, hsDev.data,
          scDev.data, int32(nTok), int32(H), int32(E),
          int32(I), int32(K), int32(S), scaleV) do:
        grid = (nTok, 1)
        blk = (32, 1)
      engine.copyToHostFromDevice(addr dst.data[0], oDev.data, n)
    elif which == 1:
      engine.launchDeviceFn L(hDev.data, xDev.data, rwDev.data, guDev.data,
          dDev.data, sguDev.data, sdwDev.data, oDev.data, hsDev.data,
          scDev.data, int32(nTok), int32(H), int32(E),
          int32(I), int32(K), int32(S), scaleV) do:
        grid = (nTok, 1)
        blk = (32, 1)
      engine.copyToHostFromDevice(addr dst.data[0], hDev.data, n)
    else:
      engine.launchDeviceFn L(hsDev.data, xDev.data, rwDev.data, guDev.data,
          dDev.data, sguDev.data, sdwDev.data, hDev.data, oDev.data,
          scDev.data, int32(nTok), int32(H), int32(E),
          int32(I), int32(K), int32(S), scaleV) do:
        grid = (nTok, 1)
        blk = (32, 1)
      engine.copyToHostFromDevice(addr dst.data[0], hsDev.data, n)

# ─── The sigmoid route judgment over the pre-pass ids and weights ─────

template judgeSigmoidRoute*(idsH, wH, xH, rwH: untyped, nTok, H, E, K: int, scaleV: float32, label: string) =
  ## Asserts the pre-pass routing against the f64 sigmoid reference:
  ##
  ## - ids inside [0, E) and distinct
  ## - slot pairs against the f64 top-K under the tie region
  ## - weights over the kernel's own id set under the fp32 bar
  let rawLogits = rawLogitsF64[float16](xH.data, rwH.data, nTok, H, E)
  for tok in 0 ..< nTok:
    let sRef = sigmoidScoresF64(rawLogits[tok * E ..< (tok + 1) * E], E)
    let sel = topKF64(sRef, E, K)
    var gotRow = newSeq[int32](K)
    for slot in 0 ..< K:
      gotRow[slot] = idsH.data[tok * K + slot]
    var slopeDeltaSum = 0.0'f64
    for slot in 0 ..< K:
      slopeDeltaSum += reassocDelta(H, sumAbsF64[float16](xH.data, rwH.data,
          H, tok * H, gotRow[slot].int * H))
    for slot in 0 ..< K:
      let xE = idsH.data[tok * K + slot].int
      doAssert xE >= 0 and xE < E,
        label & ": id " & $xE & " outside [0, " & $E & ") at (tok " & $tok &
          ", slot " & $slot & ")"
      doAssert slot == 0 or idsH.data[tok * K + slot] != idsH.data[tok * K + slot - 1],
        label & ": duplicate id at (tok " & $tok & ", slot " & $slot & ")"
      if xE != sel[slot].int:
        # a tie swap: the swapped pair's score gap must sit within
        # its reassociation and sigmoid slope bound
        doAssert abs(sRef[xE] - sRef[sel[slot].int]) <= sigmoidScoreTieBound(
            reassocDelta(H, sumAbsF64[float16](xH.data, rwH.data, H, tok * H, xE * H)),
            reassocDelta(H, sumAbsF64[float16](xH.data, rwH.data, H, tok * H,
              sel[slot].int * H))),
          label & ": non-tie id mismatch at (tok " & $tok & ", slot " & $slot &
            "): got " & $xE & ", want " & $sel[slot]
      let gotW = widenF64(wH.data[tok * K + slot])
      let wRef = sigmoidWeightsF64(sRef, gotRow, K, scaleV)[slot]
      let bar = sigmoidWeightBar(gotW, K, slopeDeltaSum,
        reassocDelta(H, sumAbsF64[float16](xH.data, rwH.data, H, tok * H, xE * H)))
      doAssert abs(gotW - wRef) <= bar,
        label & ": weight bar at (tok " & $tok & ", slot " & $slot & "): " &
          $abs(gotW - wRef) & " > " & $bar

# ─── The fused judgment: bounds walk, activation bars, f64 out bar ────

template actRefRow*(xArr, gArr, uArr: untyped, xOff, gOff, uOff, H: int, act: MoeAct, slope, uStep: float64): (float64, float64) =
  ## Returns: (h64, bar) for one g/up operand pair, the f64 activation
  ## value in the exact El-round sequence and the matching `hBarF64` bar.
  let g64 = dotOffF64[float16](xArr, gArr, H, xOff, gOff)
  let u64 = dotOffF64[float16](xArr, uArr, H, xOff, uOff)
  let a64 = case act
    of maSilu: siluF64(g64)
    of maGeluTanh: geluTanhF64(g64)
  let h64 = widenF64(hValueF64[float16](g64, u64, act))
  let bar = hBarF64(H, uStep,
    sumAbsF64[float16](xArr, gArr, H, xOff, gOff),
    sumAbsF64[float16](xArr, uArr, H, xOff, uOff),
    a64, u64, h64, slope)
  (h64, bar)

template downDotColumn*(hRefs, hBars, dArr: untyped, hBase, dBase, I: int): (float64, float64) =
  ## Returns: (d64, downBar) for one slot column's down GEMM: the f64 dot
  ## over the slot's reference h values and the down weights, plus the bar
  ## covering the walk's reassociation and the propagated h bar errors.
  var d64 = 0.0'f64
  var sumAbsDown = 0.0'f64
  var hProp = 0.0'f64
  for row in 0 ..< I:
    let hv = hRefs[hBase + row]
    let dv = widenF64(dArr[dBase + row])
    d64 += hv * dv
    sumAbsDown += abs(hv * dv)
    hProp += hBars[hBase + row] * abs(dv)
  (d64, reassocDelta(I, sumAbsDown) + hProp)

template judgeFwd*(hH, hsH, outH, idsH, wH, xH, rwH, guH, dH, sguH, sdwH:
    untyped, nTok, H, E, I, K, S: int, act: MoeAct, slope, uStep: float64,
    label: string) =
  ## Full fused judgment over the read-back scratch and output buffers:
  ##
  ## - the bounds walk: every claimed cell written, plus the spare
  ##   (nTok+1)-th scratch entries untouched
  ## - the routed h_scratch slices and the shared hs_scratch slices
  ##   against the f64 activation chain under the derived bar
  ## - out_r against the f64 chain: per-slot f64 g/up dots then the double
  ##   El round, a f64 down dot over the reference El h values and the fp32
  ##   slot-order weighted accumulation plus the shared sum, one El round
  ## - the bar books each slot's down-walk reassociation and the h/hs bar
  ##   errors propagated through the down weights and the fp32 accumulation
  ##   roundoffs plus the final El round
  ##
  ##   x, weights ──► [ f64 dots, exact El rounds ] ──► h/hs refs
  ##   h/hs refs ──► [ f64 down dots, fp32 slot-order sum ] ──► out ref
  ##
  # ── bounds walk ──
  for jj in 0 ..< nTok * K * I:
    doAssert cast[uint16](hH.data[jj]) != 0xFFFF'u16,
      label & ": h_scratch cell " & $jj & " never written"
  for jj in 0 ..< nTok * S * I:
    doAssert cast[uint16](hsH.data[jj]) != 0xFFFF'u16,
      label & ": hs_scratch cell " & $jj & " never written"
  for jj in 0 ..< nTok * H:
    doAssert cast[uint16](outH.data[jj]) != 0xFFFF'u16,
      label & ": out cell " & $jj & " never written"
  for jj in nTok * K * I ..< (nTok + 1) * K * I:
    doAssert cast[uint16](hH.data[jj]) == 0xFFFF'u16,
      label & ": h_scratch bleed into the spare row at " & $jj
  for jj in nTok * S * I ..< (nTok + 1) * S * I:
    doAssert cast[uint16](hsH.data[jj]) == 0xFFFF'u16,
      label & ": hs_scratch bleed into the spare row at " & $jj
  # ── routed h rows: the f64 chain and the bar per value ──
  var hRefs = newSeq[float64](nTok * K * I)
  var hBars = newSeq[float64](nTok * K * I)
  var hsRefs = newSeq[float64](nTok * S * I)
  var hsBars = newSeq[float64](nTok * S * I)
  for tok in 0 ..< nTok:
    for slot in 0 ..< K:
      let xE = idsH.data[tok * K + slot].int
      for row in 0 ..< I:
        let gOff = (xE * 2) * I * H + row * H
        let uOff = (xE * 2 + 1) * I * H + row * H
        let (h64, bar) = actRefRow(xH.data, guH.data, guH.data, tok * H,
          gOff, uOff, H, act, slope, uStep)
        hRefs[(tok * K + slot) * I + row] = h64
        hBars[(tok * K + slot) * I + row] = bar
        let got = widenF64(hH.data[(tok * K + slot) * I + row])
        doAssert abs(got - h64) <= bar,
          label & ": h bar at (tok " & $tok & ", slot " & $slot & ", row " &
            $row & "): got " & $got & ", want " & $h64 & ", bar " & $bar
  # ── shared hs rows ──
  for tok in 0 ..< nTok:
    for sh in 0 ..< S:
      let sguOff = sh * 2 * I * H
      for row in 0 ..< I:
        let (hs64, bar) = actRefRow(xH.data, sguH.data, sguH.data, tok * H,
          sguOff + row * H, sguOff + (I + row) * H, H, act, slope, uStep)
        hsRefs[(tok * S + sh) * I + row] = hs64
        hsBars[(tok * S + sh) * I + row] = bar
        let got = widenF64(hsH.data[(tok * S + sh) * I + row])
        doAssert abs(got - hs64) <= bar,
          label & ": hs bar at (tok " & $tok & ", sh " & $sh & ", row " &
            $row & "): got " & $got & ", want " & $hs64 & ", bar " & $bar
  # ── out_r against the full f64 chain ──
  for tok in 0 ..< nTok:
    for col in 0 ..< H:
      var want = 0.0'f64
      var bar = 0.0'f64
      for slot in 0 ..< K:
        let xE = idsH.data[tok * K + slot].int
        let wv = widenF64(wH.data[tok * K + slot])
        let (d64, downBar) = downDotColumn(hRefs, hBars, dH.data,
          (tok * K + slot) * I, xE * H * I + col * I, I)
        want += wv * d64
        bar += wv * downBar
      for sh in 0 ..< S:
        let (d64, downBar) = downDotColumn(hsRefs, hsBars, sdwH.data,
          (tok * S + sh) * I, sh * H * I + col * I, I)
        want += d64
        bar += downBar
      bar += (2 * K + 4).float64 * U32 * abs(want) + uStep * abs(want) + FloorSub
      let got = widenF64(outH.data[tok * H + col])
      doAssert abs(got - want) <= bar,
        label & ": out bar at (tok " & $tok & ", col " & $col & "): got " &
          $got & ", want " & $want & ", bar " & $bar

# ─── The decode-path out judgment (the cross check's second side) ─────

template judgeDecodeCross*(outH, idsH, wH, xH, rwH, guH, dH, sgmH, supH,
    sdwH: untyped, nTok, H, E, I, K: int, uStep: float64, label: string) =
  ## Decode-path out judgment: out_r against the f64 expert walk conditioned
  ## on the decode path's own El-rounded routing with the same chain and bar
  ## structure as `judgeFwd`'s out judgment:
  ##
  ## - the silu member's reference h values (the decode path's own rounding contract)
  ## - the shared expert added ungated
  for tok in 0 ..< nTok:
    var hRefs = newSeq[float64](K * I)
    var hBars = newSeq[float64](K * I)
    for slot in 0 ..< K:
      let xE = idsH.data[tok * K + slot].int
      for row in 0 ..< I:
        let (h64, bar) = actRefRow(xH.data, guH.data, guH.data, tok * H,
          (xE * 2) * I * H + row * H, (xE * 2 + 1) * I * H + row * H,
          H, maSilu, SiluSlopeMax.float64, uStep)
        hRefs[slot * I + row] = h64
        hBars[slot * I + row] = bar
    var hsRefs = newSeq[float64](I)
    var hsBars = newSeq[float64](I)
    for row in 0 ..< I:
      let (hs64, bar) = actRefRow(xH.data, sgmH.data, supH.data, tok * H,
        row * H, row * H, H, maSilu, SiluSlopeMax.float64, uStep)
      hsRefs[row] = hs64
      hsBars[row] = bar
    for col in 0 ..< H:
      var want = 0.0'f64
      var bar = 0.0'f64
      for slot in 0 ..< K:
        let xE = idsH.data[tok * K + slot].int
        let wv = widenF64(wH.data[tok * K + slot])
        let (d64, downBar) = downDotColumn(hRefs, hBars, dH.data,
          slot * I, xE * H * I + col * I, I)
        want += wv * d64
        bar += wv * downBar
      let (d64, downBar) = downDotColumn(hsRefs, hsBars, sdwH.data,
        0, col * I, I)
      want += d64
      bar += downBar
      bar += (2 * K + 4).float64 * U32 * abs(want) + uStep * abs(want) + FloorSub
      let got = widenF64(outH.data[tok * H + col])
      doAssert abs(got - want) <= bar,
        label & ": decode out bar at (tok " & $tok & ", col " & $col &
          "): got " & $got & ", want " & $want & ", bar " & $bar
