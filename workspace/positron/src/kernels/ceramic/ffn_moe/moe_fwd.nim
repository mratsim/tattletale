# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ──────────────────  moe_fwd (fused MoE forward megakernel)  ──────────────────

## Fused routed-plus-shared MoE forward megakernel on the tile algebra.
## Tile-stack port of the v1 ceramic `ffn_moe/ffn_moe.nim` runtime-config
## entry `moe_fwd`. fp16 storage, fp32 mma over fp16 operands,
## one token per threadgroup of 32 lanes, every dim a runtime value.
##
## Forms:
##
## - the fused megakernel (`moe_fwd`): per token, routing → shared experts →
##   routed slots → one barrier → the output walk.
## - the host config guard (`moeFwdConfigGuard`): raises naming the offending
##   dim (the device entry itself drops configs beyond the compiled-in maxima).
## - the activation chain: the silu member reuses `moe_decode.siluMulEager`,
##   the gelu-tanh member `geluMulEager` below, both with the same double
##   El-round contract so the two paths of the op stay numerically consistent.
##
## Dataflow per token t (one threadgroup):
##
##   x (H) ──► [ moeRouteSigmoidRT ] ──► ids (K), w (K)
##           ──► w = s / (Σs + RouteSumEps) · routed_scaling, fp32, no El round
##
##   x (H) ──► [ (W_g | W_u)[s] @ x ] ──► (g, u) (I) ──► [ act(g)·u ]
##             shared expert s                          ──► El ──► hs_scratch[t, s] (I)
##
##   x (H) ──► [ (W_g | W_u)[e] @ x ] ──► (g, u) (I) ──► [ act(g)·u ]
##             e = ids[slot]                            ──► El ──► h_scratch[t, slot] (I)
##
##   threadgroup barrier: the output walk re-reads the stored scratch entries
##
##   h_scratch[t, slot] ──► [ down_w[ids[slot]] @ · ] ──┐
##                                ──► Σ_slot w[slot]·(down) ──► [ + ]
##   hs_scratch[t, s]   ──► [ shared_down_w[s] @ · ] ──┘          ──► El ──► out_r[t] (H)
##                                ──► Σ_s (down)
##
## Buffers, all fp16 except the fp32 selection scratch, every size a runtime dim:
##
##   | buffer         | shape                                            |
##   | -------------- | ------------------------------------------------ |
##   | out_r, x       | (num_tokens, hidden)                             |
##   | router_w       | (n_routed_experts, hidden)                       |
##   | gate_up_w      | (n_routed_experts, 2·moe_intermediate, hidden)   |
##   | down_w         | (n_routed_experts, hidden, moe_intermediate)     |
##   | shared_*       | (n_shared_experts, ·), the same fusion           |
##   | h_scratch      | (num_tokens, top_k, moe_intermediate), scratch   |
##   | hs_scratch     | (num_tokens, n_shared_experts, moe_intermediate) |
##   | scores_scratch | (num_tokens, ScoreChunk·ScoreChunks), fp32       |

import workspace/crucible
import workspace/math
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/kernels/k_layout_gemm
import ./moe_router
import ./moe_decode

type
  MoeAct* = enum
    ## Expert activation the fused MoE chain applies after the fused
    ## g/up projection, along the model config's activation axis.
    maSilu      ## silu(g) · u
    maGeluTanh  ## gelu_pytorch_tanh(g) · u in the gemma-family form

const
  ScoreChunks* = 8
    ## Compiled-in 64-expert chunk max: an accepted n_routed_experts stays
    ## inside [1, ScoreChunk·ScoreChunks], per the v1 ceramic ffn_moe.nim.

const
  InvSqrt2Pi* = 0.7978845608028654'f32
    ## 1/sqrt(2π) factor of the gelu-tanh inner polynomial.
    ## Value copied from the v1 ceramic math_consts.nim.
  GeluCoef* = 0.044715'f32
    ## Cubic coefficient of the gelu-tanh inner polynomial.
    ## Value copied from the v1 ceramic math_consts.nim.

# ─── Local device extensions: the activation and the bounded store ───

proc geluMulEager[El, Sh, St](dst: var Tile[El, Sh, St], gHalf, uHalf: Tile[float32, Sh, St]) =
  ## `dst[i] = El(geluTanh(gHalf[i]) · uHalf[i])` elementwise over the fp32
  ## g/u accumulator operands. This proc implements the gelu-tanh member
  ## of the `siluMulEager` rounding contract:
  ##
  ## - the gelu result rounds to El (RNE)
  ## - the El-rounded gelu times the fp32 up operand rounds once at the store
  ##
  ## gelu-tanh evaluates in fp32 through the exp2 tanh identity
  ## th = 1 − 2/(exp2(2s·log2e) + 1), stable at both saturation ends,
  ## with s = InvSqrt2Pi·(g + GeluCoef·g³).
  for i in 0 ..< dst.getLayout().size():
    let g = gHalf.data[i]
    let s = InvSqrt2Pi * (g + GeluCoef * g * g * g)
    let th = 1.0'f32 - 2.0'f32 / (exp2(2.0'f32 * s * Log2e) + 1.0'f32)
    let a = 0.5'f32 * g * (1.0'f32 + th)
    dst.data[i] = (a.elRound(El) * uHalf.data[i]).to(El)

template storeRow0ElLim(dst, src, cLayout, regCrd, crdBase: untyped, colBase, colLim: int) =
  ## Stores the m = 0 cells of the El tile `src` into the flat view `dst`
  ## at `colBase + n`, one writer per stored value, columns capped at `colLim`:
  ##
  ## - cells at m > 0 hold the operands' zero fill, an unguarded store
  ##   corrupts neighbouring cells
  ## - a real dim below the 32-wide tile column leaves the tile's tail
  ##   columns outside `colLim`, so the store never touches the next
  ##   stored vector's cells
  for i in 0 ..< cLayout.size():
    let (m, n) = crdBase + regCrd.crd2idx(i)
    if m == 0:
      let col = colBase + n
      if col < colLim:
        dst[col] = src.data[i]

# ─── The compiled-in maxima guard ─────────────────────────────────────

proc moeFwdConfigGuard*(
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts: int32) =
  ## Host-side half of `moe_fwd`'s compiled-in maxima guard.
  ##
  ## Raises with the offending dimension named. The device entry itself
  ## only drops the launch (the GPU has no message channel), so a caller
  ## runs this companion before the launch.
  ##
  ## Accepted config:
  ##   - num_tokens, hidden, moe_intermediate at least 1
  ##   - n_shared_experts at least 0
  ##   - n_routed_experts inside [1, ScoreChunk·ScoreChunks], top_k inside [1, MaxTopK] and at most n_routed_experts
  ##
  ## `moeFwdConfigGuard(8, 2048, 64, 1536, 4, 1)` returns, a config inside every
  ## bound while `moeFwdConfigGuard(8, 2048, 1024, 1536, 4, 1)` raises
  ## AssertionDefect naming n_routed_experts (the expert max times the chunk width).
  doAssert num_tokens >= 1, "num_tokens must be at least 1"
  doAssert hidden >= 1, "hidden must be at least 1"
  doAssert moe_intermediate >= 1, "moe_intermediate must be at least 1"
  doAssert n_shared_experts >= 0, "n_shared_experts must be at least 0"
  doAssert n_routed_experts >= 1 and n_routed_experts <= ScoreChunk * ScoreChunks,
    "n_routed_experts outside [1, " & $(ScoreChunk * ScoreChunks) & "]"
  doAssert top_k >= 1 and top_k <= MaxTopK,
    "top_k outside [1, " & $MaxTopK & "]"
  doAssert top_k <= n_routed_experts,
    "moeFwdConfigGuard: the K slots need K distinct experts, top_k exceeds n_routed_experts"

# ─── The kernel, the runtime-config entry ─────────────────────────────

proc moe_fwd*(
    out_r: ptr UncheckedArray[float16],              # (num_tokens, hidden) fp16 output
    x: ptr UncheckedArray[float16],                  # (num_tokens, hidden) fp16 activations
    router_w: ptr UncheckedArray[float16],           # (n_routed_experts, hidden) fp16 router weight
    gate_up_w: ptr UncheckedArray[float16],          # (n_routed_experts, 2·moe_intermediate, hidden) fp16
    down_w: ptr UncheckedArray[float16],             # (n_routed_experts, hidden, moe_intermediate) fp16
    shared_gate_up_w: ptr UncheckedArray[float16],   # (n_shared_experts, 2·moe_intermediate, hidden) fp16
    shared_down_w: ptr UncheckedArray[float16],      # (n_shared_experts, hidden, moe_intermediate) fp16
    h_scratch: ptr UncheckedArray[float16],          # (num_tokens, top_k, moe_intermediate) fp16 buffer
    hs_scratch: ptr UncheckedArray[float16],         # (num_tokens, n_shared_experts, moe_intermediate) fp16 buffer
    scores_scratch: ptr UncheckedArray[float32],     # (num_tokens, ScoreChunk·ScoreChunks) fp32 selection scratch
    num_tokens, hidden, n_routed_experts, moe_intermediate, top_k,
    n_shared_experts: int32,
    routed_scaling: float32,
    activation: static MoeAct) =
  ## Launch dims (num_tokens, 1, 1), 32 lanes, one token per threadgroup,
  ## the module header's dataflow one full pass per token.
  ##
  ## Parameter comments on the signature name every buffer's shape, and all tensors
  ## are fp16 and last-axis contiguous except the fp32 scores_scratch,
  ## whose token stride is ScoreChunk·ScoreChunks.
  ##
  ##   x (H) ──► [ router_w @ x, ceil(E/64) chunks ] ──► sigmoid ──► scores (E)
  ##           ──► top-K, lowest-index tiebreak ──► ids (K), w (K)
  ##           ──► w = s / (Σs + RouteSumEps) · routed_scaling
  ##
  ##   x (H) ──► [ (W_g | W_u)[e] @ x ] ──► (g, u) (I) ──► [ act(g)·u ]
  ##             e = ids[slot]                          ──► El ──► h_scratch[t, slot]
  ##   x (H) ──► [ (W_g | W_u)[s] @ x ] ──► (g, u) (I) ──► [ act(g)·u ]
  ##             shared s                               ──► El ──► hs_scratch[t, s]
  ##
  ##   h_scratch[t, slot] ──► [ down_w[ids[slot]] @ · ] ──┐
  ##                                ──► Σ_slot w[slot]·(down) ──► [ + ]
  ##   hs_scratch[t, s]   ──► [ shared_down_w[s] @ · ] ──┘          ──► El ──► out_r[t] (H)
  ##                                ──► Σ_s (down)
  ##
  ## Rounding points (the op's consistency contract with the decode path):
  ##
  ## - the routing adds no El round: raw fp32 logits, raw fp32 sigmoid
  ##   scores and w = s/(Σs + RouteSumEps)·routed_scaling all in fp32
  ##   (`moeRouteSigmoidRT`'s contract)
  ## - the h_scratch/hs_scratch values round twice per element
  ##   (`siluMulEager`/`geluMulEager`): the activation rounds to El,
  ##   the El-rounded activation times the fp32 up operand rounds to El
  ##   at the store
  ## - the fp32 weighted down sums accumulate in fp32 and out_r rounds
  ##   once at the store
  ##
  ## Ragged-native over the runtime dims:
  ##
  ## - the K walks bound loads to the raw dim, zero-filled lanes leave
  ##   the mma accumulator untouched, and the stores cap at the real size
  ##   (`storeRow0ElLim`)
  ## - the routing scratch's cells past n_routed_experts stage a finite
  ##   fp32 floor, so they cannot win the top-K
  ## - a config beyond the compiled-in maxima drops the launch:
  ##   n_routed_experts > 512, top_k > MaxTopK, top_k > n_routed_experts,
  ##   or a non-positive dim. `moeFwdConfigGuard` names the offending dim,
  ##   run it before each launch
  ## - register budget: near 2 live 32×32 fp32 accumulators, the gHalf/uHalf
  ##   pair plus transients, one fixed budget independent of the runtime dims
  if num_tokens < 1 or hidden < 1 or moe_intermediate < 1 or
      n_shared_experts < 0 or
      n_routed_experts < 1 or n_routed_experts > ScoreChunk * ScoreChunks or
      top_k < 1 or top_k > MaxTopK or top_k > n_routed_experts:
    return

  let
    t = int32(threadgroup_position_in_grid.x)
    tok = int(t)
    hid = int(hidden)
    inter = int(moe_intermediate)
    shpTH: (int, int) = (int(num_tokens), hid)
    shpEH: (int, int) = (int(n_routed_experts), hid)
    shpGu: (int, int, int, int) = (int(n_routed_experts), 2, inter, hid)
    shpDw: (int, int, int) = (int(n_routed_experts), hid, inter)
    shpSGu: (int, int, int, int) = (int(n_shared_experts), 2, inter, hid)
    shpSDw: (int, int, int) = (int(n_shared_experts), hid, inter)
    shpH: (int, int, int) = (int(num_tokens), int(top_k), inter)
    shpHs: (int, int, int) = (int(num_tokens), int(n_shared_experts), inter)
    strTH: (int, int) = (hid, 1)
    strGu: (int, int, int, int) = (2 * inter * hid, inter * hid, hid, 1)
    strDw: (int, int, int) = (hid * inter, inter, 1)
    strH: (int, int, int) = (int(top_k) * inter, inter, 1)
    strHs: (int, int, int) = (int(n_shared_experts) * inter, inter, 1)
    shpSc: (int, int) = (int(num_tokens), ScoreChunk * ScoreChunks)
    strSc: (int, int) = (ScoreChunk * ScoreChunks, 1)
    xV = make_view(x, shpTH, strTH)
    rwV = make_view(router_w, shpEH, strTH)
    guV = make_view(gate_up_w, shpGu, strGu)
    dwV = make_view(down_w, shpDw, strDw)
    sguV = make_view(shared_gate_up_w, shpSGu, strGu)
    sdwV = make_view(shared_down_w, shpSDw, strDw)
    hV = make_view(h_scratch, shpH, strH)
    hsV = make_view(hs_scratch, shpHs, strHs)
    scV = make_view(scores_scratch, shpSc, strSc)
    outV = make_view(out_r, shpTH, strTH)

  # ── router: logits → sigmoid scores → top-K → weights ──
  var ids: array[MaxTopK, int32]
  var w: array[MaxTopK, float32]
  moeRouteSigmoidRT[float16](xV, rwV, t, scV(tok, _), top_k, routed_scaling,
    ids, w)

  const mma = TiledMma[float32, float16, float16, 8, 32, 16](
    atom: APPLE_8x8x8_F16)
  # BTileN is the mma tile's N width: the accumulator's 32-wide column tile
  const BTileN = 32
  const
    tvmA = mma.getThreadValueMap(opA)
    tvmB = mma.getThreadValueMap(opB)
    tvmC = mma.getThreadValueMap(opC)
    cLayout = mma.getMmaLayout(opC)
  let
    thrID = getLocalThreadID()
    thrCrd = idx2crd(tvmC.getThreadPartitioning(), thrID)
    regCrd = tvmC.layout.slice(((thrCrd, _), _))
    crdBase = tvmC.layout.crd2idx(((thrCrd, _), _))
    kSteps = (hid + 15) div 16
    iHalfTiles = (inter + 31) div 32
    hTiles = (hid + 31) div 32
    iSteps = (inter + 15) div 16
  var
    gHalf: tile(float32, cLayout)
    uHalf: tile(float32, cLayout)
    hT: tile(float16, cLayout)
    d: tile(float32, cLayout)
    routed: tile(float32, cLayout)
    sh: tile(float32, cLayout)
    outT: tile(float16, cLayout)
    a: tile(float16, mma.getMmaLayout(opA))
    bT: tile(float16, mma.getMmaLayout(opB))
  let
    xOp = xV(tok ..< tok + 1, _)

  # ── shared expert activations: hs[s] = act(gs)·us -> hs_scratch[t, s] ──
  for s in 0'i32 ..< n_shared_experts:
    let sgG = sguV(int(s), 0, _, _)
    let sgU = sguV(int(s), 1, _, _)
    let hsRow = hsV(tok, int(s), _)
    for nt in 0 ..< iHalfTiles:
      gHalf.zeros()
      uHalf.zeros()
      for kk in 0 ..< kSteps:
        a.load(xOp, tvmA, thrID, (0, kk))
        bT.load(sgG, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(gHalf, a, bT, gHalf)
        bT.load(sgU, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(uHalf, a, bT, uHalf)
      when activation == maSilu:
        siluMulEager(hT, gHalf, uHalf)
      else:
        geluMulEager(hT, gHalf, uHalf)
      storeRow0ElLim(hsRow, hT, cLayout, regCrd, crdBase, nt * BTileN, inter)

  # ── the top_k routed expert activations -> h_scratch[t, slot] ──
  for slot in 0'i32 ..< top_k:
    let guG = guV(int(ids[slot]), 0, _, _)
    let guU = guV(int(ids[slot]), 1, _, _)
    let hRow = hV(tok, int(slot), _)
    for nt in 0 ..< iHalfTiles:
      gHalf.zeros()
      uHalf.zeros()
      for kk in 0 ..< kSteps:
        a.load(xOp, tvmA, thrID, (0, kk))
        bT.load(guG, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(gHalf, a, bT, gHalf)
        bT.load(guU, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(uHalf, a, bT, uHalf)
      when activation == maSilu:
        siluMulEager(hT, gHalf, uHalf)
      else:
        geluMulEager(hT, gHalf, uHalf)
      storeRow0ElLim(hRow, hT, cLayout, regCrd, crdBase, nt * BTileN, inter)

  # ── threadgroup barrier ──
  # cross-lane read: the output walk re-reads every lane's stored scratch
  # entries from device memory, so the barrier call below orders that read
  # to happen after the stores above
  threadgroup_barrier_device()

  # ── output: routed = Σ w[slot]·down_w[e] @ h, + shared, El store ──
  for nt in 0 ..< hTiles:
    routed.zeros()
    for slot in 0'i32 ..< top_k:
      d.zeros()
      let hOp = hV(tok, int(slot) ..< int(slot) + 1, _)
      let dSlice = dwV(int(ids[slot]), _, _)
      for kk in 0 ..< iSteps:
        a.load(hOp, tvmA, thrID, (0, kk))
        bT.load(dSlice, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(d, a, bT, d)
      for i in 0 ..< cLayout.size():
        routed.data[i] += w[slot] * d.data[i]
    sh.zeros()
    for s in 0'i32 ..< n_shared_experts:
      let hsOp = hsV(tok, int(s) ..< int(s) + 1, _)
      let sdSlice = sdwV(int(s), _, _)
      for kk in 0 ..< iSteps:
        a.load(hsOp, tvmA, thrID, (0, kk))
        bT.load(sdSlice, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(sh, a, bT, sh)
    let outRow = outV(tok, _)
    for i in 0 ..< cLayout.size():
      outT.data[i] = (routed.data[i] + sh.data[i]).to(float16)
    storeRow0ElLim(outRow, outT, cLayout, regCrd, crdBase, nt * BTileN, hid)
