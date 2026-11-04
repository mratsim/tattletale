# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal fused MoE forward suite, tile-stack megakernel judged against f64 host references
##
## Test that the fused routed-plus-shared MoE forward matches its f64 references
## end to end:
##
## - the sigmoid routing pre-pass: ids, weights and the tie region
## - the h_scratch/hs_scratch activation values under the double El round
##   and the weighted output walk against the full f64 chain
## - the cross-path consistency against the decode plus merge path
## - the config guard with the over-max device drop and the relaunch determinism
##
## Cross-path reasoning:
##
## - the decode path routes through the Qwen softmax form and the megakernel
##   through the GLM sigmoid form, so the paths select different experts
##   on the same fixture
## - each path is judged against the f64 expert walk conditioned on
##   the ids and weights of its own pre-pass: sigmoid for the megakernel,
##   softmax for the decode path
## - the shared-expert bars are routing-independent
## - the decode path only implements the silu member, so the cross-path
##   check runs on silu and the gelu member is judged against its own f64
##   chain in the ragged case
##
## Cases:
##
## - glm47-flash-like silu T=8 and T=4 scale 1.8, qwen36-like silu E=256 K=8
## - ragged-gelu E=70 H=2064 I=550 K=4 S=2, cross-path silu E=64 K=4 I=512
## - the routing scratch poison case, the router NaN poison case
## - a ragged K tail at H=2040, the config guard
##
## Run:
##   nim c -r -d:release --outdir:build/tests --nimcache:nimcache/moe_fwd --hints:off workspace/positron/tests/ceramic/t_moe_fwd_metal.nim

import std/[random, strutils]
import workspace/crucible
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/tests/ceramic_testutils
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_router
import workspace/positron/src/kernels/ceramic/ffn_moe/moe_fwd
import moe_ref_utils
import moe_test_kernels

# ─── Case runner ──────────────────────────────────────────────────────

template runFwdCase(engine, FO, FH, FS: untyped, nTok: int, seed: int64,
    scaleV: float32, act: MoeAct, slope: float64, label: string,
    H, E, I, K, S: int) =
  ## One fused case:
  ##
  ## - fixture fill per the v1 manual test's recipe: x within ±0.3 and weights within ±0.125,
  ##   then the selection scratch poisoned with 1e30, so every scratch cell
  ##   past E must lose the top-K
  ## - the routing pre-pass judged against the f64 sigmoid reference
  ## - three fused readback rounds (h_scratch, hs_scratch, out_r)
  ## - the full fused judgment plus the relaunch determinism check
  let uStep = uStepF64(float16)
  var rng = initRand(seed)
  var xH = makeHostTensor(float16, (nTok, H), (H, 1))
  var rwH = makeHostTensor(float16, (E, H), (H, 1))
  var guH = makeHostTensor(float16, (E, 2, I, H), (2 * I * H, I * H, H, 1))
  var dH = makeHostTensor(float16, (E, H, I), (H * I, I, 1))
  var sguH = makeHostTensor(float16, (S, 2, I, H), (2 * I * H, I * H, H, 1))
  var sdwH = makeHostTensor(float16, (S, H, I), (H * I, I, 1))
  var hH = makeHostTensor(float16, (nTok + 1, K, I), (K * I, I, 1))
  var hsH = makeHostTensor(float16, (nTok + 1, S, I), (S * I, I, 1))
  var outH = makeHostTensor(float16, (nTok, H), (H, 1))
  var idsH = makeHostTensor(int32, (nTok, K), (K, 1))
  var wH = makeHostTensor(float32, (nTok, K), (K, 1))
  var scH = makeHostTensor(float32, (nTok, ScoreChunk * ScoreChunks),
    (ScoreChunk * ScoreChunks, 1))

  for jj in 0 ..< nTok * H:
    xH.data[jj] = narrowEl[float16](rng.rand(-0.3'f32 .. 0.3'f32).float64)
  for jj in 0 ..< E * H:
    rwH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< E * 2 * I * H:
    guH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< E * H * I:
    dH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< S * 2 * I * H:
    sguH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< S * H * I:
    sdwH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< (nTok + 1) * K * I:
    hH.data[jj] = cast[float16](0xFFFF'u16)
  for jj in 0 ..< (nTok + 1) * S * I:
    hsH.data[jj] = cast[float16](0xFFFF'u16)
  for jj in 0 ..< nTok * H:
    outH.data[jj] = cast[float16](0xFFFF'u16)
  for jj in 0 ..< nTok * K:
    idsH.data[jj] = -1
    wH.data[jj] = 0.0'f32
  for jj in 0 ..< nTok * ScoreChunk * ScoreChunks:
    scH.data[jj] = 1.0e30'f32

  routePreRound(engine, routePreSigmoid, routePreW, idsH, wH, xH, rwH, scH,
    nTok, H, E, int32(K), scaleV)
  judgeSigmoidRoute(idsH, wH, xH, rwH, nTok, H, E, K, scaleV, label)

  fwdRound(engine, FH, hH, hsH, outH, xH, rwH, guH, dH, sguH, sdwH, scH,
    nTok, H, E, I, K, S, scaleV, 1, hH, (nTok + 1) * K * I)
  fwdRound(engine, FS, hH, hsH, outH, xH, rwH, guH, dH, sguH, sdwH, scH,
    nTok, H, E, I, K, S, scaleV, 2, hsH, (nTok + 1) * S * I)
  fwdRound(engine, FO, hH, hsH, outH, xH, rwH, guH, dH, sguH, sdwH, scH,
    nTok, H, E, I, K, S, scaleV, 0, outH, nTok * H)

  judgeFwd(hH, hsH, outH, idsH, wH, xH, rwH, guH, dH, sguH, sdwH,
    nTok, H, E, I, K, S, act, slope, uStep, label)

  # ── determinism: relaunch the out round, the result must be bit-identical ──
  var savedOut = newSeq[uint16](nTok * H)
  for jj in 0 ..< nTok * H:
    savedOut[jj] = cast[uint16](outH.data[jj])
  fwdRound(engine, FO, hH, hsH, outH, xH, rwH, guH, dH, sguH, sdwH, scH,
    nTok, H, E, I, K, S, scaleV, 0, outH, nTok * H)
  for jj in 0 ..< nTok * H:
    doAssert cast[uint16](outH.data[jj]) == savedOut[jj],
      label & ": out relaunch mismatch at " & $jj

  echo "[", label, "] route, bounds walk, h/hs bars, out f64 bar, determinism: green"

# ─── The routing scratch poison case ──────────────────────────────────

proc runRoutePoisonCase[E: HwEngine](engine: E) =
  ## E=70 with the selection scratch's cells past E poisoned to 1e30:
  ## the staging must overwrite every walked chunk cell with the real
  ## sigmoid score or the fp32 finite floor, so a poisoned tail cell can
  ## never win the top-K. The id range assert below is the catch.
  const
    T = 1
    H = 64
    E = 70
    K = 4
  let scaleV = 1.0'f32
  var rng = initRand(0xC04D0706'i64)
  var xH = makeHostTensor(float16, (T, H), (H, 1))
  var rwH = makeHostTensor(float16, (E, H), (H, 1))
  var idsH = makeHostTensor(int32, (T, K), (K, 1))
  var wH = makeHostTensor(float32, (T, K), (K, 1))
  var scH = makeHostTensor(float32, (T, ScoreChunk * ScoreChunks),
    (ScoreChunk * ScoreChunks, 1))
  for jj in 0 ..< T * H:
    xH.data[jj] = narrowEl[float16](rng.rand(-0.3'f32 .. 0.3'f32).float64)
  for jj in 0 ..< E * H:
    rwH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< T * K:
    idsH.data[jj] = -1
    wH.data[jj] = 0.0'f32
  for jj in 0 ..< T * ScoreChunk * ScoreChunks:
    scH.data[jj] = 1.0e30'f32
  routePreRound(engine, routePreSigmoid, routePreW, idsH, wH, xH, rwH, scH,
    T, H, E, int32(K), scaleV)
  judgeSigmoidRoute(idsH, wH, xH, rwH, T, H, E, K, scaleV, "route poison")
  echo "[route poison] E=70 tail cells cannot win the top-K: green"

# ─── The router NaN poison case ───────────────────────────────────────

proc runRouteNaNCase[E: HwEngine](engine: E) =
  ## Router weights poisoned with the quiet-NaN fp16 bit pattern:
  ##
  ## - every weight NaN makes every logit and staged score NaN, so no staged
  ##   score matches the NaN group max and every slot takes the unmatched
  ##   route with id eCount−1, zero weight, the ids in range
  ## - the RouteSumEps guard keeps the division finite over the all-zero
  ##   weight sum so no NaN reaches the weights
  ## - one expert poisoned among real scores never matches the group max,
  ##   never wins a slot, and no NaN reaches the weights
  const
    T = 1
    H = 64
    E = 70
    K = 4
  let scaleV = 1.0'f32
  var rng = initRand(0xC04D0708'i64)
  var xH = makeHostTensor(float16, (T, H), (H, 1))
  var rwH = makeHostTensor(float16, (E, H), (H, 1))
  var idsH = makeHostTensor(int32, (T, K), (K, 1))
  var wH = makeHostTensor(float32, (T, K), (K, 1))
  var scH = makeHostTensor(float32, (T, ScoreChunk * ScoreChunks),
    (ScoreChunk * ScoreChunks, 1))
  for jj in 0 ..< T * H:
    xH.data[jj] = narrowEl[float16](rng.rand(-0.3'f32 .. 0.3'f32).float64)
  for jj in 0 ..< T * K:
    idsH.data[jj] = -1
    wH.data[jj] = 0.0'f32
  for jj in 0 ..< T * ScoreChunk * ScoreChunks:
    scH.data[jj] = 1.0e30'f32
  # every router weight NaN: the unmatched route on all slots
  for jj in 0 ..< E * H:
    rwH.data[jj] = cast[float16](0x7E00'u16)
  routePreRound(engine, routePreSigmoid, routePreW, idsH, wH, xH, rwH, scH,
    T, H, E, int32(K), scaleV)
  for slot in 0 ..< K:
    doAssert idsH.data[slot] == int32(E - 1),
      "route NaN: slot " & $slot & " routed to id " & $idsH.data[slot] &
        ", want the unmatched expert " & $(E - 1)
    doAssert cast[uint32](wH.data[slot]) == 0'u32,
      "route NaN: slot " & $slot & " weight bits " &
        $cast[uint32](wH.data[slot]) & ", want zero"
  # one expert's weights NaN among real scores: the poisoned expert
  # never wins a slot
  for jj in 0 ..< E * H:
    rwH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< H:
    rwH.data[jj] = cast[float16](0x7E00'u16)
  for jj in 0 ..< T * K:
    idsH.data[jj] = -1
  routePreRound(engine, routePreSigmoid, routePreW, idsH, wH, xH, rwH, scH,
    T, H, E, int32(K), scaleV)
  for slot in 0 ..< K:
    let xE = idsH.data[slot].int
    doAssert xE >= 1 and xE < E,
      "route NaN mixed: slot " & $slot & " id " & $xE & " outside [1, " &
        $E & ")"
    let bits = cast[uint32](wH.data[slot])
    doAssert ((bits shr 23) and 0xFF'u32) != 0xFF'u32,
      "route NaN mixed: slot " & $slot & " weight is NaN or Inf"
  echo "[route NaN] all-poison unmatched route, one-poison real route: green"

# ─── The cross-path case (megakernel vs decode plus merge) ────────────

proc runCrossSiluCase[E: HwEngine](engine: E) =
  ## Same fixture through both paths, each judged against its own conditioned
  ## f64 chain (the reasoning sits in the file header):
  ##
  ## - the megakernel: sigmoid routing pre-pass, three fused rounds, and the full fused judgment
  ## - the decode path: the softmax routing pre-pass, one decode launch
  ##   (grid (T, K+1), SharedGate off), then the merge launch and the f64
  ##   out bar conditioned on the decode path's own El-rounded routing
  const
    T = 2
    H = 2048
    E = 64
    K = 4
    I = 512
    S = 1
    Scale = 1.0'f32
  let uStep = uStepF64(float16)
  var rng = initRand(0xC04D0707'i64)
  var xH = makeHostTensor(float16, (T, H), (H, 1))
  var rwH = makeHostTensor(float16, (E, H), (H, 1))
  var guH = makeHostTensor(float16, (E, 2, I, H), (2 * I * H, I * H, H, 1))
  var dH = makeHostTensor(float16, (E, H, I), (H * I, I, 1))
  var sguH = makeHostTensor(float16, (S, 2, I, H), (2 * I * H, I * H, H, 1))
  var sdwH = makeHostTensor(float16, (S, H, I), (H * I, I, 1))
  var hH = makeHostTensor(float16, (T + 1, K, I), (K * I, I, 1))
  var hsH = makeHostTensor(float16, (T + 1, S, I), (S * I, I, 1))
  var outH = makeHostTensor(float16, (T, H), (H, 1))
  var idsH = makeHostTensor(int32, (T, K), (K, 1))
  var wH = makeHostTensor(float32, (T, K), (K, 1))
  var scH = makeHostTensor(float32, (T, ScoreChunk * ScoreChunks),
    (ScoreChunk * ScoreChunks, 1))
  # decode-path tensors: separate g/up halves copied from the fused
  # buffer's halves (the same values), plus the decode path's own routing
  # record and the poisoned sgv vector (never read with SharedGate off)
  var sgmH = makeHostTensor(float16, (S, I, H), (I * H, H, 1))
  var supH = makeHostTensor(float16, (S, I, H), (I * H, H, 1))
  var sgvH = makeHostTensor(float16, (S, H), (H, 1))
  var dPartialH = makeHostTensor(float32, (T, K + 1, H), ((K + 1) * H, H, 1))
  var dScH = makeHostTensor(float32, (T, K + 1, E), ((K + 1) * E, E, 1))
  var dIdsH = makeHostTensor(int32, (T, K), (K, 1))
  var dWH = makeHostTensor(float16, (T, K), (K, 1))
  var dRouteScH = makeHostTensor(float32, (T, E), (E, 1))
  var dOutH = makeHostTensor(float16, (T, H), (H, 1))
  var dHH = makeHostTensor(float16, (T, K, I), (K * I, I, 1))
  var dHsH = makeHostTensor(float16, (T, I), (I, 1))

  for jj in 0 ..< T * H:
    xH.data[jj] = narrowEl[float16](rng.rand(-0.3'f32 .. 0.3'f32).float64)
  for jj in 0 ..< E * H:
    rwH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< E * 2 * I * H:
    guH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< E * H * I:
    dH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< S * 2 * I * H:
    sguH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< S * H * I:
    sdwH.data[jj] = narrowEl[float16](rng.rand(-0.125'f32 .. 0.125'f32).float64)
  for jj in 0 ..< (T + 1) * K * I:
    hH.data[jj] = cast[float16](0xFFFF'u16)
  for jj in 0 ..< (T + 1) * S * I:
    hsH.data[jj] = cast[float16](0xFFFF'u16)
  for jj in 0 ..< T * H:
    outH.data[jj] = cast[float16](0xFFFF'u16)
  for jj in 0 ..< T * K:
    idsH.data[jj] = -1
    wH.data[jj] = 0.0'f32
    dIdsH.data[jj] = -1
  for jj in 0 ..< T * ScoreChunk * ScoreChunks:
    scH.data[jj] = 1.0e30'f32
  for jj in 0 ..< S * I * H:
    sgmH.data[jj] = sguH.data[jj]
    supH.data[jj] = sguH.data[I * H + jj]
  for jj in 0 ..< S * H:
    # poisoned sgv vector for SharedGate off, never read. Every element contains the quiet NaN bit pattern
    sgvH.data[jj] = cast[float16](0x7FC0'u16)
  for jj in 0 ..< T * (K + 1) * E:
    dScH.data[jj] = 1.0e30'f32

  # ── the megakernel path ──
  routePreRound(engine, routePreSigmoid, routePreW, idsH, wH, xH, rwH, scH,
    T, H, E, int32(K), Scale)
  judgeSigmoidRoute(idsH, wH, xH, rwH, T, H, E, K, Scale, "cross megakernel")
  fwdRound(engine, fwdGlmH, hH, hsH, outH, xH, rwH, guH, dH, sguH, sdwH,
    scH, T, H, E, I, K, S, Scale, 1, hH, (T + 1) * K * I)
  fwdRound(engine, fwdGlmS, hH, hsH, outH, xH, rwH, guH, dH, sguH, sdwH,
    scH, T, H, E, I, K, S, Scale, 2, hsH, (T + 1) * S * I)
  fwdRound(engine, fwdGlmO, hH, hsH, outH, xH, rwH, guH, dH, sguH, sdwH,
    scH, T, H, E, I, K, S, Scale, 0, outH, T * H)
  judgeFwd(hH, hsH, outH, idsH, wH, xH, rwH, guH, dH, sguH, sdwH,
    T, H, E, I, K, S, maSilu, SiluSlopeMax.float64, uStep, "cross megakernel")

  # ── the decode path: softmax routing pre-pass, decode launch, merge ──
  block:
    let idsDev = engine.bridge(dIdsH)
    let wDev = engine.bridge(dWH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let sDev = engine.bridge(dRouteScH)
    engine.launchDeviceFn routeCrossIds(idsDev, wDev, xDev, rwDev, sDev) do:
      grid = (T, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr dIdsH.data[0], idsDev.data, dIdsH.len())
  block:
    let idsDev = engine.bridge(dIdsH)
    let wDev = engine.bridge(dWH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let sDev = engine.bridge(dRouteScH)
    engine.launchDeviceFn routeCrossW(wDev, idsDev, xDev, rwDev, sDev) do:
      grid = (T, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr dWH.data[0], wDev.data, dWH.len())
  block:
    let pDev = engine.bridge(dPartialH)
    let xDev = engine.bridge(xH)
    let rwDev = engine.bridge(rwH)
    let guDev = engine.bridge(guH)
    let dDev = engine.bridge(dH)
    let sgmDev = engine.bridge(sgmH)
    let supDev = engine.bridge(supH)
    let sdwDev = engine.bridge(sdwH)
    let sgvDev = engine.bridge(sgvH)
    let hDev = engine.bridge(dHH)
    let hsDev = engine.bridge(dHsH)
    let scDev = engine.bridge(dScH)
    engine.launchDeviceFn decodeCross(pDev, xDev, rwDev, guDev, dDev,
        sgmDev, supDev, sdwDev, sgvDev, hDev, hsDev, scDev) do:
      grid = (T, K + 1)
      blk = (32, 1)
    # merge-launch mechanics: the merge launch re-bridges the partials
    # from host memory, so the decode launch copies its fp32 partials
    # back inside its block
    engine.copyToHostFromDevice(addr dPartialH.data[0], pDev.data,
      dPartialH.len())
  block:
    let oDev = engine.bridge(dOutH)
    let pDev = engine.bridge(dPartialH)
    engine.launchDeviceFn mergeCross(oDev, pDev) do:
      grid = (T, H div 32)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr dOutH.data[0], oDev.data, dOutH.len())
  judgeDecodeCross(dOutH, dIdsH, dWH, xH, rwH, guH, dH, sgmH, supH, sdwH,
    T, H, E, I, K, uStep, "cross decode")

  echo "[cross silu] megakernel f64 bar, decode f64 bar, shared bars: green"

# ─── The config guard ─────────────────────────────────────────────────

proc runConfigGuardCase[E: HwEngine](engine: E) =
  ## Host guard coverage: raises naming the offending dim for the v1 cases
  ## and for top_k over n_routed_experts, accepting the GLM and Qwen configs.
  ## The device entry drops an over-max launch with the output prefill untouched.
  for bad in [("n_routed_experts", 8'i32, 2048'i32, 1024'i32, 1536'i32, 4'i32, 1'i32),
              ("top_k", 8, 2048, 64, 1536, 12, 1),
              ("top_k exceeds n_routed_experts", 8, 2048, 2, 1536, 4, 1),
              ("num_tokens", 0, 2048, 64, 1536, 4, 1)]:
    let (dimName, nt, hid, ne, mi, tk, ns) = bad
    var rejected = false
    try:
      moeFwdConfigGuard(nt, hid, ne, mi, tk, ns)
    except AssertionDefect as ex:
      rejected = true
      doAssert dimName in ex.msg,
        "the guard's message lost the dim: " & ex.msg
      echo "  ", dimName, " rejected: ", ex.msg
    doAssert rejected, "the guard accepted the over-max " & dimName & " config"
  moeFwdConfigGuard(8, 2048, 64, 1536, 4, 1)
  moeFwdConfigGuard(2, 2048, 256, 512, 8, 1)
  echo "  accepted the GLM and Qwen rows"
  # device half: an over-max config returns early and the output keeps
  # its prefill values
  var xH = makeHostTensor(float16, (1, 8 * 2048), (8 * 2048, 1))
  var rwH = makeHostTensor(float16, (1, 1024 * 2048), (1024 * 2048, 1))
  var guH = makeHostTensor(float16, (1, 1024 * 3072), (1024 * 3072, 1))
  var dH = makeHostTensor(float16, (1, 1024 * 2048), (1024 * 2048, 1))
  var sguH = makeHostTensor(float16, (1, 3072), (3072, 1))
  var sdwH = makeHostTensor(float16, (1, 2048), (2048, 1))
  var hH = makeHostTensor(float16, (1, 8 * 4 * 1536), (8 * 4 * 1536, 1))
  var hsH = makeHostTensor(float16, (1, 8 * 1536), (8 * 1536, 1))
  var scH = makeHostTensor(float32, (1, 8 * 512), (8 * 512, 1))
  var outH = makeHostTensor(float16, (8, 2048), (2048, 1))
  for jj in 0 ..< 8 * 2048:
    outH.data[jj] = cast[float16](0x7BFF'u16)
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
    engine.launchDeviceFn fwdGlmO(oDev.data, xDev.data, rwDev.data,
        guDev.data, dDev.data, sguDev.data, sdwDev.data, hDev.data,
        hsDev.data, scDev.data, 8'i32, 2048'i32, 1024'i32, 1536'i32,
        4'i32, 1'i32, 1.8'f32) do:
      grid = (8, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr outH.data[0], oDev.data, outH.len())
  for jj in 0 ..< 8 * 2048:
    doAssert cast[uint16](outH.data[jj]) == 0x7BFF'u16,
      "the over-max n_routed_experts launch wrote the output at " & $jj
  echo "[config guard] raises, accepts, over-max launch dropped: green"

# ─── Driver ───────────────────────────────────────────────────────────

proc main() =
  var engine = newEngine(runMetal)
  echo "device: ", engine.deviceName()
  runRoutePoisonCase(engine)
  runRouteNaNCase(engine)
  runFwdCase(engine, fwdGlmO, fwdGlmH, fwdGlmS, 8, 0xC04D0701'i64, 1.8'f32,
    maSilu, SiluSlopeMax.float64, "glm47 T=8", 2048, 64, 1536, 4, 1)
  runFwdCase(engine, fwdGlmO, fwdGlmH, fwdGlmS, 4, 0xC04D0702'i64, 1.8'f32,
    maSilu, SiluSlopeMax.float64, "glm47 T=4", 2048, 64, 1536, 4, 1)
  runFwdCase(engine, fwdGlmO, fwdGlmH, fwdGlmS, 2, 0xC04D0703'i64, 1.0'f32,
    maSilu, SiluSlopeMax.float64, "qwen36", 2048, 256, 512, 8, 1)
  runFwdCase(engine, fwdGlmO, fwdGlmH, fwdGlmS, 2, 0xC04D0709'i64, 1.0'f32,
    maSilu, SiluSlopeMax.float64, "ragged K tail", 2040, 64, 512, 4, 1)
  runFwdCase(engine, fwdRagO, fwdRagH, fwdRagS, 2, 0xC04D0704'i64, 1.0'f32,
    maGeluTanh, GeluSlopeMax, "ragged-gelu", 2064, 70, 550, 4, 2)
  runCrossSiluCase(engine)
  runConfigGuardCase(engine)

main()
