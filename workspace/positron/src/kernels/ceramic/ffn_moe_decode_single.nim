# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ─────────────────────────────────────────────────────────────────────
# ─── MoE decode slot-group walk (moe_fwd_decode_at) ──────────────────
# ─────────────────────────────────────────────────────────────────────

## Decode-regime routed expert-body walk on the ceramic Tile API,
## one (token, slot) threadgroup recomputing the router in-group,
## the expert's gate_up rows into `h_scratch`, the down projection into the fp32 partial row.
##
## Slot groups plus the merge launch (in `moe_router`) compose
## the full MoE decode pass, the megakernel composes this core inline.
##
## | contract   | value                                                                                              |
## | ---------- | -------------------------------------------------------------------------------------------------- |
## | router     | the `moeRoute` softmax form only, logits round to El, softmax + top-K in fp32, weights round to El |
## | storage    | the element dtype, the decode composition's storage element                                        |
## | partials   | row t·(K+1)+slot holds w[slot]·down(t, slot), slot < K, row t·(K+1)+K holds gateVal·shared_down    |
## | partials 2 | the merge launch applies the single El round to the shared contribution                            |
## | buffers    | no-copy page-aligned host memory with page-multiple byte lengths                                   |

import math_consts
import workspace/crucible
import workspace/ceramic
import ./moe_router
import ./tile_widen
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Local device extensions: the activation and partial arithmetic ──

proc siluMulElemEager[El; R, C: static int; A: static MmaAtom](
    dst: var RtLeft[El, R, C, A],
    gHalf, uHalf: RtLeft[float32, R, C, A]) {.device.} =
  ## Expert activation, `dst[r][c] = bf16(silu(gHalf[r][c]) · uHalf[r][c])` over
  ## the fp32 g/u accumulator operands. The frag walk follows the loadTile
  ## lane→element mapping, the operands agreeing elementwise.
  ##
  ## Eager name separates the two `siluMulElem` contracts.
  ##
## | proc                                                  | silu operand at the multiply |
## | ----------------------------------------------------- | ---------------------------- |
## | `siluMulElemEager` (this module)                      | the bf16-rounded silu        |
## | `attn_ssm/gated_delta_net_o_norm.nim`'s `siluMulElem` | the unrounded f32 silu       |
  ##
  ## Rounding, per storage element:
  ## - the silu result rounds to bf16 (RNE)
  ## - the bf16-rounded silu times the fp32 up operand rounds once at the store
  ##   (the eager chain)
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
  ## Stores the accumulator's value row, scaled, at one fp32 partial row.
  ## The write is `dst[rowBase·rowStride + colTile·C + c] = scale · accumulator row 0`.
  ##
  ## The tile's rows above the value row carry the operand rows' zero fill,
  ## not written, one row per call, no sentinel bookkeeping at the call sites.
  ##
  ## Callers load the operand rows with `rowLimit = 1`, so accumulator row 0
  ## carries the projection's value and the rows above it are exact zeros.
  ##
  ## - the store guard requires `row == 0`, exactly one lane per stored element
  ## - the GDN y store keeps the same single-writer spelling
## | element (c, v) of tile row 0                     | written when     |
## | ------------------------------------------------ | ---------------- |
## | dst[rowBase·rowStride + colTile·C + m·N + c + v] | `row == 0`       |
## | stored value                                     | scale·tile value |
  const M = A.getM()
  const N = A.getN()
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
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
  ## One (token, slot) pair's decode walk, `t` the token, `y` the slot group, routed y < K, the shared group y = K.
  ## `moe_fwd_decode` is the grid-driven wrapper, the megakernel composes this core inline.
  ##
  ## Parameters, pointers naming their dtypes, shapes bound at the call:
  ##
  ## | parameter                     | shape, dtype, layout                                                                                                                                                    | producer                             | unit               |
  ## | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------ |
  ## | partial                       | (num_tokens, K+1, H) f32, row-major, the fp32 partials, the row t·(K+1)+y written by this proc's slot group, the merge launch in moe_router applies the single El round | this proc, the merge launch reads it | f32                |
  ## | x                             | (num_tokens, H) El, row-major, token `t`'s activation row                                                                                                               | host-computed                        | El                 |
  ## | router_w                      | (E, H) El, row-major, the router weight                                                                                                                                 | host-computed                        | El                 |
  ## | gate_up_w                     | (E, 2I, H) El, row-major, fused g/up, the g half at 0:I                                                                                                                 | host-computed                        | El                 |
  ## | down_w                        | (E, H, I) El, row-major, the down projection                                                                                                                            | host-computed                        | El                 |
  ## | shared_gate_w                 | (I, H) El, row-major, the shared gate projection                                                                                                                        | host-computed                        | El                 |
  ## | shared_up_w                   | (I, H) El, row-major, the shared up projection                                                                                                                          | host-computed                        | El                 |
  ## | shared_down_w                 | (H, I) El, row-major, the shared down projection                                                                                                                        | host-computed                        | El                 |
  ## | shared_gate_vec_w             | (1, H) El, row-major, read only when the SharedGate static is true, non-null then is the caller's obligation                                                            | host-computed                        | El                 |
  ## | h_scratch                     | (num_tokens, K, I) El, row-major, working buffer, the routed slot's fp16(act(g)·u) written at h_scratch[t, y]                                                           | this proc                            | El                 |
  ## | hs_scratch                    | (num_tokens, I) El, row-major, working buffer, the shared group writes there                                                                                            | this proc                            | El                 |
  ## | t, y                          | the token index and the slot group, y < K routed, y = K the shared group                                                                                                | device-computed (the wrapper's grid) | tokens / slots     |
  ## | H, E, K, I, Scale, SharedGate | hidden, expert count, top-K, intermediate width, the routing-weight scale and the shared-gate switch, static compile-time, shape preconditions tabled below             | compile-time                         | elements / experts |
  ##
  ## Slot-group behavior by the y index
  ##
  ## | stage        | behavior                                                                                                                                                   |
  ## | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
  ## | routed y < K | recompute the `moeRoute` router in-group, walk expert ids[y]'s gate_up rows into h_scratch[t, y] and the down projection, store the fp32 partial w[y]·down |
  ## | shared y = K | recompute the shared-expert sigmoid scalar, walk the shared expert into hs_scratch[t], store gateVal·shared_down                                           |
  ## | partials     | the module header's partial-buffer contract, the merge launch applies the single El round                                                                  |
  ##
  ## Instantiation contract:
  ## - each static binding set of this core needs its own call-site line
  ##
  ## shape preconditions, each one a static assert below:
  ##
## | precondition  | a violated shape's failure                                             |
## | ------------- | ---------------------------------------------------------------------- |
## | H mod 16 == 0 | columns silently dropped from every mma dot                            |
## | H mod 32 == 0 | the down walk's 32-wide column tiles silently truncate H               |
## | I mod 32 == 0 | h_scratch rows left unwritten, stale values re-read on the next launch |
## | E mod 64 == 0 | the 64-expert router chunk mis-tiles                                   |
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
  ## Grid-driven form of `moe_fwd_decode_at`.
  ## Grid (num_tokens, K+1, 1) at 32 lanes, one (token, slot) pair per threadgroup.
  ##
  ## Parameters, pointers naming their dtypes, shapes bound at the call,
  ## token index and slot group arriving from the grid:
  ##
  ## | parameter                     | shape, dtype, layout                                                                                                                                    | producer      | unit               |
  ## | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------ |
  ## | partial                       | (num_tokens, K+1, H) f32, row-major, the fp32 partials, produced by this kernel's slot groups, the merge launch applies the single El round             | this kernel   | f32                |
  ## | x                             | (num_tokens, H) El, row-major, the layer's activations                                                                                                  | host-computed | El                 |
  ## | router_w                      | (E, H) El, row-major, the router weight                                                                                                                 | host-computed | El                 |
  ## | gate_up_w                     | (E, 2I, H) El, row-major, fused g/up, the g half at 0:I                                                                                                 | host-computed | El                 |
  ## | down_w                        | (E, H, I) El, row-major, the down projection                                                                                                            | host-computed | El                 |
  ## | shared_gate_w                 | (I, H) El, row-major, the shared gate projection                                                                                                        | host-computed | El                 |
  ## | shared_up_w                   | (I, H) El, row-major, the shared up projection                                                                                                          | host-computed | El                 |
  ## | shared_down_w                 | (H, I) El, row-major, the shared down projection                                                                                                        | host-computed | El                 |
  ## | shared_gate_vec_w             | (1, H) El, row-major, read only when the SharedGate static is true, non-null then is the caller's obligation                                            | host-computed | El                 |
  ## | h_scratch                     | (num_tokens, K, I) El, row-major, working buffer, this kernel writes the routed slot activations there                                                  | this kernel   | El                 |
  ## | hs_scratch                    | (num_tokens, I) El, row-major, working buffer, the shared group writes there                                                                            | this kernel   | El                 |
  ## | scores_scratch                | (num_tokens, K+1, E) f32, row-major, the router selection's staged score rows, one E-slice per slot-group threadgroup, the content may be uninitialized | this kernel   | f32                |
  ## | H, E, K, I, Scale, SharedGate | hidden, expert count, top-K, intermediate width, the routing-weight scale and the shared-gate switch, static compile-time                               | compile-time  | elements / experts |
  let t = int32(threadgroup_position_in_grid.x)
  let y = int32(threadgroup_position_in_grid.y)
  moe_fwd_decode_at[El, H, E, K, I, Scale, SharedGate](
    partial, x, router_w, gate_up_w, down_w,
    shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
    h_scratch, hs_scratch, scores_scratch, t, y)
