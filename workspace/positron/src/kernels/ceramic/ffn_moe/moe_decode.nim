# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ──────────────────  moe_decode (MoE decode slots on the tile algebra)  ──────────────────

## MoE decode slot walk on the tile algebra: the tile-stack port of the v1
## ceramic `ffn_moe/ffn_moe_decode_single.nim` decode semantics.
## One (token, slot) threadgroup per call, 32 lanes, El storage.
## fp32 mma over El operands.
##
## Forms:
##
## - slot walk (`moe_fwd_decode_at`): routed slots walk the expert GEMMs
##   → silu-mul → down GEMM → fp32 partial, one router rerun per slot.
## - shared slot: the shared-gate weight logit (`sharedGateLogit`) scales
##   the shared expert's walk into partial[t, K].
## - launch-driven form (`moe_fwd_decode`): the grid supplies `t` and `y`,
##   launch dims (num_tokens, K+1, 1), 32 lanes.
## - merge (`moe_decode_merge_at` in `moe_router`): the fp32 partial
##   slices sum slot-order into the El output with one RNE round.

import workspace/crucible
import workspace/math
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_layout_gemm
import ./moe_router

# ─── Local device extensions: the activation and the bounded stores ──

proc siluMulEager*[El, Sh, St](dst: var Tile[El, Sh, St], gHalf, uHalf: Tile[float32, Sh, St]) =
  ## `dst[i] = El(silu(gHalf[i]) · uHalf[i])` elementwise over the fp32 g/u accumulator operands.
  ##
  ## Rounding, per storage element:
  ##
  ## - the silu result rounds to El (RNE)
  ## - the El-rounded silu times the fp32 up operand rounds once at the store
  for i in 0 ..< dst.getLayout().size():
    let g = gHalf.data[i]
    let s = g / (1.0'f32 + exp2(-g * Log2e))
    dst.data[i] = (s.elRound(El) * uHalf.data[i]).to(El)

template storeRow0El(dst, src, cLayout, regCrd, crdBase: untyped, colBase: int) =
  ## Stores the m = 0 cells of the El tile `src` into the flat view `dst`
  ## at `colBase + n`, one writer per stored value. Cells at m > 0 hold
  ## the operands' zero fill, an unguarded store corrupts neighbours.
  for i in 0 ..< cLayout.size():
    let (m, n) = crdBase + regCrd.crd2idx(i)
    if m == 0:
      dst[colBase + n] = src.data[i]

template storeRow0ScaledF32(dst, src, cLayout, regCrd, crdBase: untyped, colBase: int, scale: float32) =
  ## Stores the m = 0 cells of the fp32 tile `src` into the flat fp32 view
  ## `dst` at `colBase + n`, scaled. Partial slices hold raw fp32 values
  ## and the merge applies the single El round.
  for i in 0 ..< cLayout.size():
    let (m, n) = crdBase + regCrd.crd2idx(i)
    if m == 0:
      dst[colBase + n] = scale * src.data[i]

# ─── The decode slot walk ─────────────────────────────────────────────

proc moe_fwd_decode_at*[El: untyped, H, E, K, I: static int, Scale: static float32, SharedGate: static bool](
    partial, x, router_w, gate_up_w, down_w: distinct TensorView,
        # partial (num_tokens, K+1, H) fp32, x (num_tokens, H) El,
        # router_w (E, H) El, gate_up_w (E, 2, I, H) El with the gate
        # weights at half 0 and the up weights at half 1, down_w (E, H, I) El
    shared_gate_w, shared_up_w, shared_down_w: distinct TensorView,
        # (1, I, H), (1, I, H), (1, H, I) El for the shared expert's weights
    shared_gate_vec_w: distinct TensorView,
        # (1, H) El, read only when SharedGate,
        # non-null is the caller's obligation whenever SharedGate is true
    h_scratch, hs_scratch, scores_scratch: distinct TensorView,
        # h_scratch (num_tokens, K, I) El, hs_scratch (num_tokens, I) El,
        # scores_scratch (num_tokens, K+1, E) fp32, one E-slice per slot group,
        # content may be uninitialized
    t, y: int32) =
  ## One (token, slot) pair's decode walk at caller coordinates:
  ## `t` names the token and `y` the slot group. Routed groups cover y < K,
  ## y = K is the shared group. `moe_fwd_decode` is the launch-driven form
  ## and the megakernel inlines this core.
  ##
  ## Routed group y < K, e = ids[y]:
  ##
  ##   x (H) ──► [ router_w @ x, softmax, top-K ] ──► ids (K), w (K)
  ##
  ##   x (H) ──► [ (W_g | W_u)[e] @ x ] ──► (g, u) (I) ──► [ silu(g)·u ]
  ##             one pass over shared x tiles            ──► El ──► h_scratch[t, y]
  ##
  ##   h_scratch[t, y] ──► [ down_w[e] @ · ] ──► w[y]·(down) ──► partial[t, y] (H, fp32)
  ##
  ## Shared group y = K:
  ##
  ##   x (H) ──► [ shared_gate_vec_w @ x ] ──► sigmoid ──► gateVal (El round)
  ##              read only when SharedGate, non-null then is the caller's obligation
  ##
  ##   x (H) ──┬─► [ shared_gate_w @ x ] ──► g (I) ──┐
  ##           └─► [ shared_up_w @ x ]   ──► u (I) ──┴─► [ silu(g)·u ]
  ##                                                    ──► El ──► hs_scratch[t]
  ##
  ##   hs_scratch[t] ──► [ shared_down_w @ · ] ──► gateVal·(down) ──► partial[t, K] (H, fp32)
  ##
  ## Contracts:
  ##
  ## - shapes and dtypes on the parameter comments, statics H/E/K/I,
  ##   Scale and SharedGate compile-time
  ##   (H hidden, E experts, K top-K, I intermediate)
  ## - the silu result rounds to El before the multiply
  ## - the El-widened silu times the fp32 up operand, one more round
  ##   at the h_scratch/hs_scratch store (`siluMulEager`)
  ## - the partial slices hold raw fp32 and the merge launch applies one
  ##   El round
  ##
  ## Each static binding set of this core needs its own call-site line.
  ##
  ## Static asserts (see their messages for the violated-shape failure):
  ##
  ##   | precondition  | a violated shape's failure                                               |
  ##   | ------------- | ------------------------------------------------------------------------ |
  ##   | H mod 16 == 0 | columns silently dropped from every mma dot                              |
  ##   | H mod 32 == 0 | the down walk's 32-wide column tiles silently truncate H                 |
  ##   | I mod 32 == 0 | h_scratch slices left unwritten, stale values re-read on the next launch |
  ##   | E mod 64 == 0 | the 64-expert router chunk mis-tiles                                     |
  static:
    doAssert H mod 16 == 0,
      "moe_fwd_decode_at: H must be a multiple of the 16-wide K step"
    doAssert H mod 32 == 0,
      "moe_fwd_decode_at: H must be a multiple of the 32-wide down walk"
    doAssert I mod 32 == 0,
      "moe_fwd_decode_at: I must be a multiple of the 32-wide output tile"
    doAssert E mod 64 == 0,
      "moe_fwd_decode_at: E must be a multiple of the 64-expert router chunk"

  const mma = tiledMma(float32, El, El, 8, 32, 16)
  # BTileN and KStep are the tiledMma's tile N width and K step
  const
    BTileN = 32
    KStep = 16
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
  var
    gHalf: tile(float32, cLayout)
    uHalf: tile(float32, cLayout)
    hT: tile(El, cLayout)
    d: tile(float32, cLayout)
    a: tile(El, mma.getMmaLayout(opA))
    bT: tile(El, mma.getMmaLayout(opB))
  let
    xOp = x(int(t) ..< int(t) + 1, _)

  if y < K:
    var ids: array[K, int32]
    var w: array[K, float32]
    let scRow = scores_scratch(int(t), int(y), _)
    moeRoute[El, H, E, K, Scale](x, router_w, t, scRow, ids, w)
    # ── gate/up walk for ids[y] -> h_scratch[t, y] ──
    let guG = gate_up_w(int(ids[y]), 0, _, _)
    let guU = gate_up_w(int(ids[y]), 1, _, _)
    let hRow = h_scratch(int(t), int(y), _)
    for nt in 0 ..< I div BTileN:
      gHalf.zeros()
      uHalf.zeros()
      for kk in 0 ..< H div KStep:
        a.load(xOp, tvmA, thrID, (0, kk))
        bT.load(guG, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(gHalf, a, bT, gHalf)
        bT.load(guU, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(uHalf, a, bT, uHalf)
      siluMulEager(hT, gHalf, uHalf)
      storeRow0El(hRow, hT, cLayout, regCrd, crdBase, nt * BTileN)
    # ── threadgroup barrier ──
    # cross-lane read ordering: the down walk re-reads the whole
    # threadgroup's stored scratch slices. The barrier orders that read
    # after the stores
    threadgroup_barrier_device()
    # ── down walk -> partial[t, y] = w[y]·down ──
    let hOp = h_scratch(int(t), int(y) ..< int(y) + 1, _)
    let dSlice = down_w(int(ids[y]), _, _)
    let pRow = partial(int(t), int(y), _)
    for nt in 0 ..< H div BTileN:
      d.zeros()
      for kk in 0 ..< I div KStep:
        a.load(hOp, tvmA, thrID, (0, kk))
        bT.load(dSlice, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(d, a, bT, d)
      storeRow0ScaledF32(pRow, d, cLayout, regCrd, crdBase, nt * BTileN, w[y])
  else:
    # ── shared gate scalar (the moe_fwd chain's rounding form) ──
    var gateVal = 1.0'f32
    when SharedGate:
      let l32 = sharedGateLogit[El, H](x, shared_gate_vec_w, t)
      gateVal = elRound(1.0'f32 / (1.0'f32 + exp2(-l32 * Log2e)), El)
    # ── shared expert activation -> hs_scratch[t] ──
    let sgG = shared_gate_w(0, _, _)
    let sgU = shared_up_w(0, _, _)
    let hsRow = hs_scratch(int(t), _)
    for nt in 0 ..< I div BTileN:
      gHalf.zeros()
      uHalf.zeros()
      for kk in 0 ..< H div KStep:
        a.load(xOp, tvmA, thrID, (0, kk))
        bT.load(sgG, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(gHalf, a, bT, gHalf)
        bT.load(sgU, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(uHalf, a, bT, uHalf)
      siluMulEager(hT, gHalf, uHalf)
      storeRow0El(hsRow, hT, cLayout, regCrd, crdBase, nt * BTileN)
    # ── threadgroup barrier ──
    # cross-lane read ordering: the down walk re-reads the whole
    # threadgroup's stored scratch slices. The barrier orders that read
    # after the stores
    threadgroup_barrier_device()
    # ── shared down walk -> partial[t, K] = gateVal·shared_down ──
    let hsOp = hs_scratch(int(t) ..< int(t) + 1, _)
    let dSlice = shared_down_w(0, _, _)
    let pRow = partial(int(t), int(K), _)
    for nt in 0 ..< H div BTileN:
      d.zeros()
      for kk in 0 ..< I div KStep:
        a.load(hsOp, tvmA, thrID, (0, kk))
        bT.load(dSlice, tvmB, thrID, (nt, kk))
        mma.atom.gemm_tile(d, a, bT, d)
      storeRow0ScaledF32(pRow, d, cLayout, regCrd, crdBase, nt * BTileN, gateVal)

proc moe_fwd_decode*[El: untyped, H, E, K, I: static int, Scale: static float32, SharedGate: static bool](
    partial, x, router_w, gate_up_w, down_w: distinct TensorView,
        # partial (num_tokens, K+1, H) fp32, x (num_tokens, H) El,
        # router_w (E, H) El, gate_up_w (E, 2, I, H) El with the gate
        # weights at half 0 and the up weights at half 1, down_w (E, H, I) El
    shared_gate_w, shared_up_w, shared_down_w: distinct TensorView,
        # (1, I, H), (1, I, H), (1, H, I) El for the shared expert's weights
    shared_gate_vec_w: distinct TensorView,
        # (1, H) El, read only when SharedGate,
        # non-null is the caller's obligation whenever SharedGate is true
    h_scratch, hs_scratch, scores_scratch: distinct TensorView)
        # h_scratch (num_tokens, K, I) El, hs_scratch (num_tokens, I) El,
        # scores_scratch (num_tokens, K+1, E) fp32, one E-slice per slot group,
        # content may be uninitialized
  {.inline.} =
  ## Launch dims (num_tokens, K+1, 1) with 32 lanes, one threadgroup per
  ## (token, slot) pair. The grid supplies the `t` and `y` coordinates.
  ##
  ## Shapes and dtypes on the parameter comments.
  ## H/E/K/I, Scale and SharedGate static compile-time.
  ##
  ## `moe_fwd_decode_at`'s diagram applies per grid point with the score
  ## slices staged at scores_scratch[t, y], one (K+1)·E fp32 slice per
  ## slot-group threadgroup, content may be uninitialized.
  ##
  ## Contracts:
  ##
  ## - produced: partial (K+1 slices per token), h_scratch (K slices),
  ##   hs_scratch (1 slice), scores_scratch
  ## - the merge launch in `moe_router` applies the single El round
  let t = int32(threadgroup_position_in_grid.x)
  let y = int32(threadgroup_position_in_grid.y)
  moe_fwd_decode_at[El, H, E, K, I, Scale, SharedGate](
    partial, x, router_w, gate_up_w, down_w,
    shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
    h_scratch, hs_scratch, scores_scratch, t, y)
