# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ──────────────────  moe_router (the Qwen softmax routing op)  ──────────────────

## Qwen3.5/3.6 MoE router on the ceramic Tile API, the qwen35_moe mega decode kernel's softmax routing form.
## GLM sigmoid form stays in `moe_fwd.nim`, out of contract here.
##
## | contract    | value                                                                                                             |
## | ----------- | ----------------------------------------------------------------------------------------------------------------- |
## | score chain | logits accumulate fp32 over 16-wide mma chunks, one round to El, widen, fp32 softmax, top-K lowest-index tiebreak |
## | weights     | w = p/(sum p over the K selected)·Scale, one El round per weight at the store                                     |
## | tensors     | x (T, H) El, router_w (E, H) El, ids (T, K) int32, rout_w (T, K) El, partial (T, K+1, H) fp32, out_r (T, H) El    |
## | shapes      | E a multiple of the 64-expert chunk, H a multiple of the 16-wide K step and of the 32-wide merge lane tile        |
## | geometry    | `moe_route_fwd` grid (T, 1, 1) at 32 lanes, `moe_decode_merge` grid (T, H div 32, 1)                              |
##
## Shared internals with `moe_fwd.nim`:
##
## | aspect     | value                                                                                                 |
## | ---------- | ----------------------------------------------------------------------------------------------------- |
## | shared     | the row-0 logit gather, the 5-step `simdShuffleDown` reduction trees, the `own` fragment-cell mapping |
## | extraction | the routers' score chains and atom layouts differ, the reduction trees stay module-local              |
##
import math_consts
import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

const RenormSabotage* {.booldefine.} = false
  ## Compile-time sabotage switch for the poisoned-router fixture, the renorm
  ## half of the pre-fix spelling restored, the unguarded zero-sum 0/0 division.
  ##
  ## Sabotage run:
  ##   nim c -r -d:release -d:RenormSabotage --outdir:build/tests --nimcache:nimcache/red tests/ceramic/t_ceramic_moe_router.nim
  ##
  ## Red outcome:
  ##   the poisoned pass's weight-zero assert fires, slot 0 weight 0x7fc0 (NaN)

const SentinelSabotage* {.booldefine.} = false
  ## Compile-time sabotage switch for the poisoned-router fixture, restoring
  ## the unmatched-branch half of the pre-fix spelling
  ##
  ## - the raw sentinel candidate (1 shl 30) stored as the expert id
  ## - downstream expert-row reads then run off the expert weight's end
  ## - default builds leave both pre-fix spellings out
  ##
  ## Sabotage run:
  ##   nim c -r -d:release -d:SentinelSabotage --outdir:build/tests --nimcache:nimcache/red tests/ceramic/t_ceramic_moe_router.nim
  ##
  ## Red outcome:
  ##   the poisoned pass's expert-id range assert fires, id 1073741824 (1 shl 30)

# ─── Module-local bf16 row-bounded tile load ─────────────────────────
# tile_io_rows ships fp16 variants only, the bf16 guard lives module-local
# (the silu_and_mul and paged_attn precedent)
# The router writes ids and weights elementwise, so it needs no bounded store

proc loadTileRowsBf16[R, C: static int; A: static MmaAtom](
    tile: var RtLeft[bfloat16, R, C, A],
    gl: GlView[bfloat16],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded loadTile for bf16 tiles:
  ## tile plane rows origin[2]·R + r at or above `rowLimit` are zero-filled, not read.
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  let o = (int(origin[0]), int(origin[1]), int(origin[2]), int(origin[3]))
  let src = local_tile_dyn(gl, R, C, o)
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        if int32(origin[2]) * int32(R) + int32(n * M + row) < rowLimit:
          tile.frags[n][m].frag[v] = src[row + n * M, col + m * N + v]
        else:
          tile.frags[n][m].frag[v] = (0.0'f32).bfloat16

proc loadRowsE[El; R, C: static int; A: static MmaAtom](
    tile: var RtLeft[El, R, C, A],
    gl: GlView[El],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded loadTile for the 16-bit storage element El:
  ## the tile_io_rows fp16 proc or the module-local bf16 guard.
  when El is bfloat16:
    loadTileRowsBf16(tile, gl, origin, rowLimit)
  else:
    tile.loadTileRows(gl, origin, rowLimit)

# ─── Local device extensions: the score passes ───────────────────────

proc gatherScores[A, AL: static MmaAtom; F: static int](
    scores: var RtLeft[float32, 8, F, A],
    chunk: RtLeft[float32, 32, 64, AL],
    destFrag: int32) {.device.} =
  ## - Gathers one 64-expert chunk's row-0 logits into the score tile:
  ##   element (r, 8·destFrag + c) receives the raw logit of expert
  ##   64·destFrag + 8r + c, 2 values per lane across all 32 lanes
  ## - the row-0 logits live in the accumulator's lanes {0, 1, 8, 9}, 2 per
  ##   col-frag, destination lane d pulls its pair from source lane `d and 9`
  ## - `d and 9` is the row-0 owner of the same col pair under the universal AC
  ##   layout (row = b1+2b2+4b4, col = 2b0+4b3), a wrong mapping shows up
  ##   as a routing mismatch, never a value error
  static:
    doAssert AL.getM() == A.getM() and AL.getN() == A.getN() and
      AL.getVpt() == A.getVpt(),
      "gatherScores: both atoms must share the lane→fragment cell mapping"
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(AL.getLayoutA(), (lane, 0)).toIntVal()
  let r = cell mod 8          # the destination row = expert div 8
  let srcLane = lane and 9    # the row-0 owner of the lane's col pair
  for m in 0 ..< 8:
    let g0 = simdShuffle(chunk.frags[0][m].frag[0], uint32(srcLane))
    let g1 = simdShuffle(chunk.frags[0][m].frag[1], uint32(srcLane))
    if m == r:
      scores.frags[0][destFrag].frag[0] = g0
      scores.frags[0][destFrag].frag[1] = g1

proc roundScoresFp16[A: static MmaAtom; F: static int](
    scores: var RtLeft[float32, 8, F, A]) {.device.} =
  ## In-place fp16 round of the score tile's raw logits, one RNE round per logit
  ## before the softmax, the softmax form's matmul-output round on fp16
  const colFrags = F div 8
  const vpt = A.getVpt()
  for m in 0 ..< colFrags:
    for v in 0 ..< vpt:
      scores.frags[0][m].frag[v] =
        scores.frags[0][m].frag[v].to(float16).to(float32)

proc roundScoresBf16[A: static MmaAtom; F: static int](
    scores: var RtLeft[float32, 8, F, A]) {.device.} =
  ## In-place bf16 round of the score tile's raw logits, one RNE round per logit
  ## before the softmax, the softmax form's matmul-output round on bf16
  const colFrags = F div 8
  const vpt = A.getVpt()
  for m in 0 ..< colFrags:
    for v in 0 ..< vpt:
      scores.frags[0][m].frag[v] =
        scores.frags[0][m].frag[v].bfloat16.float32

proc softmaxScores[A: static MmaAtom; F: static int](
    scores: var RtLeft[float32, 8, F, A]) {.device.} =
  ## - In-place fp32 softmax over the score tile's El-rounded logits:
  ## - a tile-wide max, the shifted exponentials' sum, then a division
  ##   per element, mirroring the eager softmax's opmath
  ## - the exp is the backend's native exp2, a 1-ulp-class exponential form
  const colFrags = F div 8
  const vpt = A.getVpt()
  var lm = max(scores.frags[0][0].frag[0], scores.frags[0][0].frag[1])
  for m in 1 ..< colFrags:
    lm = max(lm, scores.frags[0][m].frag[0])
    lm = max(lm, scores.frags[0][m].frag[1])
  lm = max(lm, simdShuffleDown(lm, 16'u32))
  lm = max(lm, simdShuffleDown(lm, 8'u32))
  lm = max(lm, simdShuffleDown(lm, 4'u32))
  lm = max(lm, simdShuffleDown(lm, 2'u32))
  lm = max(lm, simdShuffleDown(lm, 1'u32))
  lm = simdShuffle(lm, 0'u32)
  var ls = 0.0'f32
  for m in 0 ..< colFrags:
    for v in 0 ..< vpt:
      ls += exp2((scores.frags[0][m].frag[v] - lm) * Log2e)
  ls += simdShuffleDown(ls, 16'u32)
  ls += simdShuffleDown(ls, 8'u32)
  ls += simdShuffleDown(ls, 4'u32)
  ls += simdShuffleDown(ls, 2'u32)
  ls += simdShuffleDown(ls, 1'u32)
  ls = simdShuffle(ls, 0'u32)
  for m in 0 ..< colFrags:
    for v in 0 ..< vpt:
      scores.frags[0][m].frag[v] =
        exp2((scores.frags[0][m].frag[v] - lm) * Log2e) / ls

proc topkScores[A: static MmaAtom; F, K: static int](
    scores: RtLeft[float32, 8, F, A],
    ids: var array[K, int32],
    w: var array[K, float32]) {.device.} =
  ## - Selects the K largest scores of the (8, F) score tile, the lowest-index
  ##   tiebreak top-K, the weights read back from the original tile
  ## - each selection pass runs on a masked copy, one 5-step
  ##   simdShuffleDown tree (deltas 16, 8, 4, 2, 1) broadcast from lane 0
  ## - the candidate expert indices where score == max (no match = 8·F) get
  ##   reduced by a 5-step min tree, the found expert masked to −float32 max
  ##   in the selection copy
  ##
  ## simdShuffleDown max → candidate min → −float32 max mask
  ##
  ## Expert layout:
  ## - element (r, 8m + c) holds expert 64m + 8r + c, the per-chunk (8, 8)
  ##   mapping at chunk m
  ## - the owner lane of expert e reads back through the same layout:
  ##   the lane's bits b0..b4 give its row and col pair
  ##   (row = b1+2b2+4b4, col = 2b0+4b3)
  var sel: RtLeft[float32, 8, F, A]
  for m in 0 ..< F div 8:
    sel.frags[0][m].frag[0] = scores.frags[0][m].frag[0]
    sel.frags[0][m].frag[1] = scores.frags[0][m].frag[1]
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let r = cell mod 8
  let c0 = cell div 8
  for slot in 0 ..< K:
    var lm = max(sel.frags[0][0].frag[0], sel.frags[0][0].frag[1])
    for m in 1 ..< F div 8:
      lm = max(lm, sel.frags[0][m].frag[0])
      lm = max(lm, sel.frags[0][m].frag[1])
    lm = max(lm, simdShuffleDown(lm, 16'u32))
    lm = max(lm, simdShuffleDown(lm, 8'u32))
    lm = max(lm, simdShuffleDown(lm, 4'u32))
    lm = max(lm, simdShuffleDown(lm, 2'u32))
    lm = max(lm, simdShuffleDown(lm, 1'u32))
    lm = simdShuffle(lm, 0'u32)
    var localCand = int32(1 shl 30)
    for m in 0 ..< F div 8:
      let e0 = int32(64 * m + 8 * r + c0)
      if sel.frags[0][m].frag[0] == lm:
        localCand = min(localCand, e0)
      if sel.frags[0][m].frag[1] == lm:
        localCand = min(localCand, e0 + 1)
    var cand = min(localCand, simdShuffleDown(localCand, 16'u32))
    cand = min(cand, simdShuffleDown(cand, 8'u32))
    cand = min(cand, simdShuffleDown(cand, 4'u32))
    cand = min(cand, simdShuffleDown(cand, 2'u32))
    cand = min(cand, simdShuffleDown(cand, 1'u32))
    cand = simdShuffle(cand, 0'u32)
    if cand >= int32(8 * F):
      # Unmatched top-K candidate:
      # a NaN/Inf-poisoned score pass compares false against NaN everywhere,
      # so no score equals the group max and the reduction keeps the sentinel.
      # - the slot routes to the last expert (8·F − 1) with zero weight
      # - the ids stay in [0, 8·F), the downstream expert-row reads stay in bounds
      # - with every score poisoned all K slots take this branch
      #   and the normalized weights sum to zero
      when SentinelSabotage:
        ids[slot] = cand
      else:
        ids[slot] = int32(8 * F - 1)
      w[slot] = 0.0'f32
    else:
      ids[slot] = cand
    let mSel = cand div 64
    let rest = cand mod 64
    let rw = rest div 8
    let cw = rest mod 8
    let own = (cw div 2 mod 2) + 2 * (rw mod 2) + 4 * ((rw div 2) mod 2) +
              8 * (cw div 4 mod 2) + 16 * ((rw div 4) mod 2)
    if cand < int32(8 * F):
      let w0 = simdShuffle(scores.frags[0][mSel].frag[0], uint32(own))
      let w1 = simdShuffle(scores.frags[0][mSel].frag[1], uint32(own))
      w[slot] = if (cw mod 2) == 0: w0 else: w1
      for m in 0 ..< F div 8:
        let e0 = int32(64 * m + 8 * r + c0)
        if e0 == cand:
          sel.frags[0][m].frag[0] = -3.402823466e38'f32
        if e0 + 1 == cand:
          sel.frags[0][m].frag[1] = -3.402823466e38'f32

# ─── The router core ─────────────────────────────────────────────────

proc moeRoute*[El; H, E, K: static int; Scale: static float32](
    x, router_w: ptr UncheckedArray[El],
    t: int32,
    ids: var array[K, int32],
    w: var array[K, float32]) {.device.} =
  ## One token's top-K expert ids and routing weights, the chunked router
  ## GEMV, the softmax form's score pass, the in-register top-K selection
  ## by lowest index. Register-only, no logits scratch.
  ##
  ## Composed in-group per slot group by the mega kernel.
  ##
  ## Contract:
  ##
  ## - the (8, E div 8) score tile assembles chunk by chunk, chunk cs's 64
  ##   row-0 logits land in col-frag cs
  ## - the logits round to El once, the fp32 softmax runs over the widened values,
  ##   the normalized weights round to El (the eager routing-weights cast)
  ## - a poisoned score pass (NaN/Inf logits) leaves no candidate matching
  ##   the group max, the slot routes to expert E−1 with zero weight
  ##
  ## | poisoned pass | value                                                               |
  ## | ------------- | ------------------------------------------------------------------- |
  ## | ids           | expert E−1, in [0, E), the expert-row reads stay in bounds          |
  ## | weights       | zero and kept zero by the renorm guard, the 0/0 sum never NaNs them |
  const F = E div 8
  static:
    doAssert E mod 64 == 0, "moeRoute: E must be a multiple of the 64-expert chunk"
    doAssert H mod 16 == 0, "moeRoute: H must be a multiple of the 16-wide K step"
    doAssert K <= E, "moeRoute: the K slots need K distinct experts"
  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (H, 0, H, 1))
  let glRouter = router_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, H, 1))

  var scores: rt_l(float32, 8, F)
  var dR: rt_l(float32, 32, 64, getTileConfig(float32, El))
  var a: rt_l(El, 32, 16)
  var b64: rt_r(El, 16, 64)

  for cs in 0'i32 ..< E div 64:
    dR.zero()
    for kk in 0'i32 ..< H div 16:
      a.loadRowsE(glX, (t, 0, 0, kk), 1)
      b64.loadTile(glRouter, (0, 0, cs, kk))
      dR.mma_AB(a, b64)
    scores.gatherScores(dR, cs)
  when El is bfloat16:
    scores.roundScoresBf16()
  else:
    scores.roundScoresFp16()
  scores.softmaxScores()
  scores.topkScores(ids, w)
  var sumW = 0.0'f32
  for slot in 0 ..< K:
    sumW += w[slot]
  for slot in 0 ..< K:
    when RenormSabotage:
      w[slot] = w[slot] / sumW * Scale
    else:
      if sumW > 0.0'f32:
        w[slot] = w[slot] / sumW * Scale
      else:
        # every weight is zero (the poisoned pass) or the sum underflowed,
        # a 0/0 store would NaN the weight and everything downstream
        w[slot] = 0.0'f32
    when El is bfloat16:
      w[slot] = w[slot].bfloat16.float32
    else:
      w[slot] = w[slot].to(float16).to(float32)

# ─── The router-only entry ───────────────────────────────────────────

proc moe_route_fwd*[El; H, E, K: static int; Scale: static float32](
    ids: ptr UncheckedArray[int32],        # (num_tokens, K) expert ids
    rout_w: ptr UncheckedArray[El],        # (num_tokens, K) routing weights
    x: ptr UncheckedArray[El],             # (num_tokens, H) activations
    router_w: ptr UncheckedArray[El],      # (E, H) router weight
    num_tokens: int32) {.device.} =
  ## - Router-only pass, one grid point per token, the launch host's entry
  ## - stores the top-K expert ids and the El-rounded routing weights under
  ##   the `moeRoute` contract, grid (num_tokens, 1, 1) at 32 lanes
  let t = int32(threadgroup_position_in_grid.x)
  var idsReg: array[K, int32]
  var wReg: array[K, float32]
  moeRoute[El, H, E, K, Scale](x, router_w, t, idsReg, wReg)
  for slot in 0 ..< K:
    ids[t * K + slot] = idsReg[slot]
    when El is bfloat16:
      rout_w[t * K + slot] = wReg[slot].bfloat16
    else:
      rout_w[t * K + slot] = wReg[slot].to(float16)

# ─── The shared-expert gate logit ────────────────────────────────────

proc sharedGateLogit*[El; H: static int](
    x, sgw: ptr UncheckedArray[El], t: int32): float32 {.device.} =
  ## Returns the raw fp32 shared-expert scalar logit of token t
  ## - x[t] · shared_gate_vec_w[0], the (1, H) row weight as a one-output GEMV
  ##   over the same 16-wide K steps as the router
  ## - element (0, 0) of the (32, 8) accumulator carries the value on lane 0,
  ##   the caller broadcasting it with simdShuffle from lane 0
  ## - the form's El round, if any, happening after this load
  static:
    doAssert H mod 16 == 0, "sharedGateLogit: H must be a multiple of 16"
  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (H, 0, H, 1))
  let glSgw = sgw.gd(shape = (-1, -1, -1, -1), stride = (1, 0, H, 1))
  var sg: rt_l(float32, 32, 8, getTileConfig(float32, El))
  var a: rt_l(El, 32, 16)
  var b: rt_r(El, 16, 8)
  sg.zero()
  for kk in 0'i32 ..< H div 16:
    a.loadRowsE(glX, (t, 0, 0, kk), 1)
    b.loadTile(glSgw, (0, 0, 0, kk))
    sg.mma_AB(a, b)
  result = simdShuffle(sg.frags[0][0].frag[0], 0'u32)

# ─── The fp32-partial merge (decode regime) ──────────────────────────

proc moe_decode_merge_at*[El; H, K: static int](
    out_r: ptr UncheckedArray[El],         # (num_tokens, H) routed+shared output
    partial: ptr UncheckedArray[float32],  # (num_tokens, K+1, H) fp32 partials
    t, nt: int32) {.device.} =
  ## - Merge walk for one (token, column block) pair at caller coordinates,
  ##   `t` the token and `nt` the H div 32 column block
  ## - sums the token's K+1 fp32 partial rows in slot order, the shared
  ##   contribution last, one El round at the store
  ## - the mega kernel composes this core inline
  ##
  ## producers → (K+1, H) fp32 partials → slot-order sum → El store
  ##
  ## Instantiation contract:
  ## - each static binding set of this core needs a distinct call-site line
  ## - the engine's monomorphization key erases generic static bindings,
  ##   calls that share one call-site line all collapse into a single body
  static:
    doAssert H mod 32 == 0,
      "moe_decode_merge_at: H must be a multiple of the 32-wide lane tile"
  let lane = int32(thread_index_in_threadgroup)
  let col = nt * 32 + lane
  var acc = 0.0'f32
  for y in 0'i32 ..< K + 1:
    acc += partial[int(t * (K + 1) + y) * H + int(col)]
  when El is bfloat16:
    out_r[t * H + col] = acc.bfloat16
  else:
    out_r[t * H + col] = acc.to(float16)

proc moe_decode_merge*[El; H, K: static int](
    out_r: ptr UncheckedArray[El],         # (num_tokens, H) routed+shared output
    partial: ptr UncheckedArray[float32]) {.device.} =
  ## - Grid-driven form of `moe_decode_merge_at`:
  ## - grid (num_tokens, H div 32, 1) at 32 lanes, one output column per lane
  ## - the partial row t·(K+1)+y holds slot y's fp32 contribution, the decode
  ##   regime's partial-buffer contract
  static:
    doAssert H mod 32 == 0,
      "moe_decode_merge: H must be a multiple of the 32-wide lane tile"
  let t = int32(threadgroup_position_in_grid.x)
  let nt = int32(threadgroup_position_in_grid.y)
  moe_decode_merge_at[El, H, K](out_r, partial, t, nt)
