# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ─────────────────────────────────────────────────────────────────────
# ─── MoE decode slot-group walk (moe_fwd_decode_at) ──────────────────
# ─────────────────────────────────────────────────────────────────────

## Decode-regime routed expert-body walk on the ceramic Tile API, one
## (token, slot) threadgroup per call of `moe_fwd_decode_at`. Slot groups
## plus the merge launch (in `moe_router`) compose the full MoE decode
## pass, the megakernel composes this core inline.
##
## Per (token t, slot group y), El storage, fp32 mma over El operands,
## H = hidden, E = experts, K = top-K, I = intermediate:
##
##   routed y < K, e = ids[y]:
##   x (H) ──► [ router_w @ x, sigmoid softmax, top-K ] ──► ids (K), w (K)
##   x (H) ──► [ (W_g | W_u)[e] @ x ] ──► (g, u) (I) ──► [ silu(g)·u ]
##             one pass over shared x tiles            ──► El ──► h_scratch[t, y]
##   h_scratch[t, y] ──► [ down_w[e] @ · ] ──► w[y]·(down) ──► partial[t, y] (H, fp32)
##
##   shared y = K:
##   x (H) ──► [ shared_gate_vec_w @ x ] ──► sigmoid ──► gateVal (El round), SharedGate only
##
##   x (H) ──┬─► [ shared_gate_w @ x ] ──► g (I) ──┐
##           └─► [ shared_up_w @ x ]   ──► u (I) ──┴─► [ silu(g)·u ]
##                                                    ──► El ──► hs_scratch[t]
##   hs_scratch[t] ──► [ shared_down_w @ · ] ──► gateVal·(down) ──► partial[t, K] (H, fp32)
##
## the shared expert keeps gate and up in separate weight tensors, two
## boxes over the shared x tiles (the routed gate_up_w is one fused (E, 2I, H)
## tensor, the g half at 0:I).
##
## Contract:
##   - partial rows: t·(K+1)+y holds w[y]·down(t, y) for y < K,
##     t·(K+1)+K holds gateVal·shared_down, the merge launch applies the
##     single El round to the shared contribution
##   - routing is the `moeRoute` softmax form only, logits round to El,
##     softmax + top-K in fp32, weights round to El
##   - buffers no-copy page-aligned host memory, page-multiple byte lengths
##   - static asserts below: H mod 16 == 0, H mod 32 == 0, I mod 32 == 0,
##     E mod 64 == 0 (see their messages for the violated-shape failure)

import ../math_consts
import workspace/crucible
import workspace/ceramic
import ./moe_router
import ../tile_widen
import ../tile_io_rows

export layout_algebra, tensors, tile_algebra, ptr_arithmetic

# ─── Local device extensions: the activation and partial arithmetic ──

proc siluMulElemEager[El; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[El, R, C, A],
    gHalf, uHalf: RtLeft[float32, R, C, A]) {.device.} =
  ## `dst[r][c] = El(silu(gHalf[r][c]) · uHalf[r][c])` over the fp32 g/u
  ## accumulator operands, the frag walk following the loadTile lane→element
  ## mapping so the operands agree elementwise.
  ##
  ## Distinct from `attn_ssm/gated_delta_net_o_norm.nim`'s `siluMulElem`,
  ## which multiplies the unrounded f32 silu:
  ##
  ##   | proc                      | silu operand at the multiply |
  ##   | ------------------------- | ---------------------------- |
  ##   | `siluMulElemEager` (here) | the El-rounded silu          |
  ##   | gated_delta_net_o_norm    | the unrounded f32 silu       |
  ##
  ## Rounding, per storage element:
  ## - the silu result rounds to El (RNE)
  ## - the El-rounded silu times the fp32 up operand rounds once at the store
  dst.map2(gHalf, uHalf) do:
    let g = x
    let s = g / (1.0'f32 + exp2(-g * Log2e))
    roundToNearestEven[El](roundToNearestEven[El](s).float32 * y)

# tiles-allow storeRowScaledF32 is the row-bounded tile io machinery, it needs
# one bounded-IO tile-io primitive (row-guarded load/store over register tiles)
proc storeRowScaledF32[R, C: static int; A: static MmaAtom](
    dst: ptr UncheckedArray[float32],
    tile: RtLeft[float32, R, C, A],
    rowBase, rowStride, colTile: int32, scale: float32) {.device.} =
  ## Stores accumulator row 0, scaled, at one fp32 partial row:
  ## `dst[rowBase·rowStride + colTile·C + c] = scale · accumulator row 0`.
  ##
  ##   | element (c, v) of tile row 0                     | written when |
  ##   | ------------------------------------------------ | ------------ |
  ##   | dst[rowBase·rowStride + colTile·C + m·N + c + v] | `row == 0`   |
  ##   | stored value                                     | scale·tile   |
  ##
  ## One row per call, the rows above row 0 carry the operands' zero fill
  ## and are not written (callers load with rowLimit = 1). The store guard
  ## `row == 0` keeps exactly one lane per stored element, the same
  ## single-writer spelling as the GDN y store.
  const M = A.getM()
  const N = A.getN()
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = thread_index_in_threadgroup
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let r = cell mod A.getM()
  let c = cell div A.getM()
  if r == 0:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst[int(rowBase) * int(rowStride) + int(colTile) * C +
            m * N + c + v] = scale * tile.frags[0][m].frag[v]

# ─── The decode slot-group walk ───────────────────────────────────────

proc moe_fwd_decode_at*[El; H, E, K, I: static int; Scale: static float32;
    SharedGate: static bool](
    partial: ptr UncheckedArray[float32],  # (num_tokens, K+1, H) fp32 partials
    x, router_w, gate_up_w, down_w: ptr UncheckedArray[El],
    shared_gate_w, shared_up_w, shared_down_w: ptr UncheckedArray[El],
    shared_gate_vec_w: ptr UncheckedArray[El] = nil,
        # (1, H), read only when SharedGate
        # non-null is the caller's obligation whenever SharedGate is true
    h_scratch: ptr UncheckedArray[El],     # (num_tokens, K, I) working buffer
    hs_scratch: ptr UncheckedArray[El],    # (num_tokens, I) working buffer
    scores_scratch: ptr UncheckedArray[float32], # (num_tokens, K+1, E) router selection scratch
    t, y: int32) {.device.} =
  ## One (token, slot) pair's decode walk, `t` the token, `y` the slot
  ## group, routed y < K, the shared group y = K. `moe_fwd_decode` is the
  ## grid-driven form, the megakernel composes this core inline.
  ## Shapes and dtypes on the pointer comments, H/E/K/I/Scale/SharedGate
  ## static compile-time (H hidden, E experts, K top-K, I intermediate).
  ##
  ## Routed group y < K, e = ids[y]:
  ##
  ##   x (H) ──► [ router_w @ x, sigmoid softmax, top-K ] ──► ids (K), w (K)
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
  ## Contract:
  ## - the silu result rounds to El before the multiply, the product rounds
  ##   once more at the h_scratch/hs_scratch store (`siluMulElemEager`)
  ## - the partial rows stay fp32 unrounded, the merge launch applies the
  ##   single El round (the module header's partial-row contract)
  ## - each static binding set of this core needs its own call-site line
  ##
  ## Static asserts (see their messages for the violated-shape failure):
  ##
  ##   | precondition  | a violated shape's failure                                             |
  ##   | ------------- | ---------------------------------------------------------------------- |
  ##   | H mod 16 == 0 | columns silently dropped from every mma dot                            |
  ##   | H mod 32 == 0 | the down walk's 32-wide column tiles silently truncate H               |
  ##   | I mod 32 == 0 | h_scratch rows left unwritten, stale values re-read on the next launch |
  ##   | E mod 64 == 0 | the 64-expert router chunk mis-tiles                                   |
  static:
    doAssert H mod 16 == 0,
      "moe_fwd_decode_at: H must be a multiple of the 16-wide K step"
    doAssert H mod 32 == 0,
      "moe_fwd_decode_at: H must be a multiple of the 32-wide down walk"
    doAssert I mod 32 == 0,
      "moe_fwd_decode_at: I must be a multiple of the 32-wide output tile"
    doAssert E mod 64 == 0,
      "moe_fwd_decode_at: E must be a multiple of the 64-expert router chunk"

  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (H, 0, H, 1))
  let glGu = gate_up_w.gd(shape = (-1, -1, -1, -1), stride = (2 * I * H, 0, H, 1))
  let glDown = down_w.gd(shape = (-1, -1, -1, -1), stride = (H * I, 0, I, 1))
  let glSg = shared_gate_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, H, 1))
  let glSu = shared_up_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, H, 1))
  let glSd = shared_down_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, I, 1))
  let glH = h_scratch.gd(shape = (-1, -1, -1, -1), stride = (I, I, I, 1))
  let glHs = hs_scratch.gd(shape = (-1, -1, -1, -1), stride = (I, 0, I, 1))

  var gHalf: rt_l(float32, 32, 32, getTileConfig(float32, El))
  var uHalf: rt_l(float32, 32, 32, getTileConfig(float32, El))
  var hT: rt_l(El, 32, 32)
  var d: rt_l(float32, 32, 32, getTileConfig(float32, El))
  var a: rt_l(El, 32, 16)
  var bT: rt_r(El, 16, 32)

  if y < K:
    var ids: array[K, int32]
    var w: array[K, float32]
    moeRoute[El, H, E, K, Scale](x, router_w, t,
      scores_scratch +% int32((t * (K + 1) + y) * E), ids, w)
    # ── gate/up walk for ids[y] -> h_scratch[t, y] ──
    for nt in 0'i32 ..< I div 32:
      gHalf.zero()
      uHalf.zero()
      for kk in 0'i32 ..< H div 16:
        a.loadTileRows(glX, (t, 0, 0, kk), 1)
        bT.loadTile(glGu, (ids[y], 0, nt, kk))
        gHalf.mma_AB(a, bT)
        bT.loadTile(glGu, (ids[y], 0, nt + I div 32, kk))
        uHalf.mma_AB(a, bT)
      hT.siluMulElemEager(gHalf, uHalf)
      glH.storeTileRows(hT, (t * K + y, 0, 0, nt), 1)
    # ── threadgroup barrier ──
    # the down walk re-reads the whole threadgroup's stored scratch rows
    # from device memory, the barrier orders that cross-lane read after the stores
    threadgroup_barrier_device()
    # ── down walk -> partial[t, y] = w[y]·down ──
    for nt in 0'i32 ..< H div 32:
      d.zero()
      for kk in 0'i32 ..< I div 16:
        a.loadTileRows(glH, (t * K + y, 0, 0, kk), 1)
        bT.loadTile(glDown, (ids[y], 0, nt, kk))
        d.mma_AB(a, bT)
      storeRowScaledF32(partial, d, int32(t * (K + 1) + y), int32(H), nt, w[y])
  else:
    # ── shared gate scalar (the moe_fwd chain's rounding form) ──
    var gateVal = 1.0'f32
    when SharedGate:
      let l32 = sharedGateLogit[El, H](x, shared_gate_vec_w, t)
      gateVal = roundToNearestEven[El](1.0'f32 / (1.0'f32 +
        exp2(-l32 * Log2e))).float32
    # ── shared expert activation -> hs_scratch[t] ──
    for nt in 0'i32 ..< I div 32:
      gHalf.zero()
      uHalf.zero()
      for kk in 0'i32 ..< H div 16:
        a.loadTileRows(glX, (t, 0, 0, kk), 1)
        bT.loadTile(glSg, (0, 0, nt, kk))
        gHalf.mma_AB(a, bT)
        bT.loadTile(glSu, (0, 0, nt, kk))
        uHalf.mma_AB(a, bT)
      hT.siluMulElemEager(gHalf, uHalf)
      glHs.storeTileRows(hT, (t, 0, 0, nt), 1)
    # ── threadgroup barrier ──
    # the down walk re-reads the whole threadgroup's stored scratch rows
    # from device memory, the barrier orders that cross-lane read after the stores
    threadgroup_barrier_device()
    # ── shared down walk -> partial[t, K] = gateVal·shared_down ──
    for nt in 0'i32 ..< H div 32:
      d.zero()
      for kk in 0'i32 ..< I div 16:
        a.loadTileRows(glHs, (t, 0, 0, kk), 1)
        bT.loadTile(glSd, (0, 0, nt, kk))
        d.mma_AB(a, bT)
      storeRowScaledF32(partial, d, int32(t * (K + 1) + K), int32(H), nt, gateVal)

proc moe_fwd_decode*[El; H, E, K, I: static int; Scale: static float32;
    SharedGate: static bool](
    partial: ptr UncheckedArray[float32],  # (num_tokens, K+1, H) fp32 partials
    x, router_w, gate_up_w, down_w: ptr UncheckedArray[El],
    shared_gate_w, shared_up_w, shared_down_w: ptr UncheckedArray[El],
    shared_gate_vec_w: ptr UncheckedArray[El] = nil,
        # (1, H), read only when SharedGate
        # non-null is the caller's obligation whenever SharedGate is true
    h_scratch: ptr UncheckedArray[El],     # (num_tokens, K, I) working buffer
    hs_scratch: ptr UncheckedArray[El],    # (num_tokens, I) working buffer
    scores_scratch: ptr UncheckedArray[float32]) {.device.} =
      # scores_scratch holds the (num_tokens, K+1, E) router selection scratch
  ## Grid (num_tokens, K+1, 1), 32 lanes, one (token, slot) pair per
  ## threadgroup, `t` and `y` from the grid. Shapes and dtypes on the
  ## pointer comments, H/E/K/I/Scale/SharedGate static compile-time.
  ##
  ## `moe_fwd_decode_at`'s diagram applies per grid point, the score rows
  ## staged at scores_scratch[t, y] (one (K+1)·E fp32 slice per slot-group
  ## threadgroup, content may be uninitialized):
  ##
  ##   x (H) ──► [ moeRoute per slot group ] ──► ids, w ──► partial[t, y] (H, fp32)
  ##
  ## Produced: partial (K+1 rows per token), h_scratch (K rows),
  ## hs_scratch (1 row), scores_scratch. The merge launch in `moe_router`
  ## applies the single El round.
  let t = int32(threadgroup_position_in_grid.x)
  let y = int32(threadgroup_position_in_grid.y)
  moe_fwd_decode_at[El, H, E, K, I, Scale, SharedGate](
    partial, x, router_w, gate_up_w, down_w,
    shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
    h_scratch, hs_scratch, scores_scratch, t, y)
